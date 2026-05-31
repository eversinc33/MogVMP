#include "handler_discovery.h"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/DeadStoreElimination.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "llvm_pipeline_utils.h"
#include "passes/concrete_alloca_prop.h"

static void dump_function_snapshot(llvm::Function &fn, const std::string &path)
{
    std::error_code      ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec)
    {
        std::cerr << "Failed to write IR snapshot '" << path << "': " << ec.message() << "\n";
        return;
    }
    fn.print(os, nullptr);
    std::cout << "[*] LLVM IR snapshot written to " << path << "\n";
}

static std::optional<uint32_t> get_constant_i32_return(llvm::Function &fn)
{
    for (auto &bb : fn)
    {
        auto *ret = llvm::dyn_cast<llvm::ReturnInst>(bb.getTerminator());
        if (!ret || ret->getNumOperands() != 1)
            continue;
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(ret->getOperand(0)))
            return static_cast<uint32_t>(ci->getZExtValue());
    }
    return std::nullopt;
}

static std::optional<uint64_t> constant_offset_from_global(llvm::Value *ptr, llvm::GlobalVariable *base)
{
    ptr = ptr->stripPointerCasts();
    if (ptr == base)
        return 0;

    auto *gep = llvm::dyn_cast<llvm::GEPOperator>(ptr);
    if (!gep || gep->getPointerOperand()->stripPointerCasts() != base)
        return std::nullopt;

    const auto &dl = base->getParent()->getDataLayout();
    llvm::APInt off(dl.getIndexSizeInBits(ptr->getType()->getPointerAddressSpace()), 0);
    if (!gep->accumulateConstantOffset(dl, off) || off.isNegative())
        return std::nullopt;
    return off.getZExtValue();
}

static std::optional<uint32_t> constant_i32(llvm::Value *v)
{
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(v))
        return static_cast<uint32_t>(ci->getZExtValue());
    return std::nullopt;
}

static std::optional<PrefixSnapshot> extract_prefix_snapshot(llvm::Function &fn)
{
    auto *module = fn.getParent();
    auto *state_g = module->getGlobalVariable("__vmp_snapshot_state");
    auto *stack_g = module->getGlobalVariable("__vmp_snapshot_stack");
    auto *base_g  = module->getGlobalVariable("__vmp_snapshot_stack_base");
    if (!state_g || !stack_g || !base_g)
        return std::nullopt;

    PrefixSnapshot snap;
    snap.state.assign(4096, std::nullopt);
    snap.stack.assign(512, std::nullopt);
    bool saw_any = false;

    // The snapshot windows are captured as i32 word stores (see build_devirt):
    // decompose each constant word into its 4 little-endian bytes. A non-constant
    // (symbolic) word leaves those bytes unknown. i8 stores are still accepted for
    // robustness against any byte-granular store the optimizer may leave behind.
    auto record_word = [](std::vector<std::optional<uint8_t>> &dst, uint64_t off, llvm::StoreInst *si)
    {
        auto *val = si->getValueOperand();
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(val))
        {
            unsigned bytes = std::max<unsigned>(1, val->getType()->getIntegerBitWidth() / 8);
            uint64_t v     = ci->getZExtValue();
            for (unsigned k = 0; k < bytes && off + k < dst.size(); ++k)
                dst[off + k] = static_cast<uint8_t>(v >> (8u * k));
        }
    };

    for (auto &bb : fn)
    {
        for (auto &inst : bb)
        {
            auto *si = llvm::dyn_cast<llvm::StoreInst>(&inst);
            if (!si)
                continue;

            llvm::Value *ptr = si->getPointerOperand();
            if (auto off = constant_offset_from_global(ptr, state_g))
            {
                if (*off < snap.state.size())
                {
                    record_word(snap.state, *off, si);
                    saw_any = true;
                }
                continue;
            }
            if (auto off = constant_offset_from_global(ptr, stack_g))
            {
                if (*off < snap.stack.size())
                {
                    record_word(snap.stack, *off, si);
                    saw_any = true;
                }
                continue;
            }
            if (ptr->stripPointerCasts() == base_g)
            {
                if (auto v = constant_i32(si->getValueOperand()))
                {
                    snap.stack_base = *v;
                    saw_any = true;
                }
            }
        }
    }

    if (!saw_any || !snap.stack_base)
        return std::nullopt;
    return snap;
}

static llvm::FunctionPassManager build_prefix_discovery_cleanup_pipeline()
{
    llvm::FunctionPassManager cleanup;
    cleanup.addPass(llvm::PromotePass());
    cleanup.addPass(llvm::GVNPass());
    cleanup.addPass(llvm::InstCombinePass());
    cleanup.addPass(llvm::SCCPPass());
    cleanup.addPass(llvm::DSEPass());
    cleanup.addPass(llvm::ADCEPass());
    cleanup.addPass(llvm::SimplifyCFGPass());
    return cleanup;
}

DiscoveryEngine::DiscoveryEngine(
    llvm::Module   &module,
    llvm::Constant *vm_initializer,
    const Memory   &memory,
    bool            save_intermediate,
    unsigned       &step_counter
)
    : module_(module)
    , vm_initializer_(vm_initializer)
    , memory_(memory)
    , save_intermediate_(save_intermediate)
    , step_counter_(step_counter)
{
}

void DiscoveryEngine::run_always_inline()
{
    llvm::ModulePassManager mpm;
    mpm.addPass(llvm::AlwaysInlinerPass());
    helpers::run_module_pipeline(module_, std::move(mpm));
}

static void run_prefix_discovery_cleanup(llvm::Function &fn)
{
    helpers::run_function_pipeline(fn, build_prefix_discovery_cleanup_pipeline());
}

static unsigned                      force_vmp_branch_conditions(llvm::Function &fn, unsigned forced_shift, bool forced_value);
static std::optional<VmpBranchInfo>  detect_vmp_branch(llvm::Function &fn, const Memory &memory);

void DiscoveryEngine::optimize(
    llvm::Function                          &fn,
    llvm::StringRef                          dump_prefix,
    std::optional<std::pair<unsigned, bool>> forced_branch_condition,
    bool                                     full_concretize
)
{
    auto apply_forced_branch = [&]()
    {
        if (!forced_branch_condition)
            return false;
        unsigned forced = force_vmp_branch_conditions(fn, forced_branch_condition->first, forced_branch_condition->second);
        if (!forced)
            return false;
        std::cout << "[*] Forced " << forced << " VMP branch selector(s) to "
                  << (forced_branch_condition->second ? 1 : 0) << "\n";
        run_prefix_discovery_cleanup(fn);
        return true;
    };

    run_always_inline();

    run_prefix_discovery_cleanup(fn);
    apply_forced_branch();
    if (!dump_prefix.empty())
        dump_function_snapshot(fn, (dump_prefix + ".cleanup.ll").str());
    // Normal discovery stops as soon as the next-handler address is known. Fork
    // snapshot capture instead drives folding to a fixed point so the whole
    // register file concretizes, not just the EIP return value.
    if (!full_concretize && get_constant_i32_return(fn))
        return;
    if (detect_vmp_branch(fn, memory_))
        return;

    for (unsigned iter = 0; iter < 100; ++iter)
    {
        // Writable VM-stack window = the synthetic host stack built by
        // build_virtual_memory_layout (ESP=0x18FF00, size 0x10000, base 0x187F00).
        // Outside it the PE image is read-only and keeps folding from the
        // initializer even after computed VM-stack stores. Keep in sync with
        // kSyntheticStackBase/Size in vmp_lifter.cpp.
        constexpr uint64_t kWritableLo = 0x00187F00;
        constexpr uint64_t kWritableHi = 0x00187F00 + 0x10000;
        unsigned folded = propagate_concrete_alloca_constants(fn, vm_initializer_, kWritableLo, kWritableHi);
        if (!dump_prefix.empty())
            dump_function_snapshot(fn, (dump_prefix + ".concrete." + std::to_string(iter) + ".ll").str());

        if (!folded)
            break;
        run_prefix_discovery_cleanup(fn);
        apply_forced_branch();
        if (!dump_prefix.empty())
            dump_function_snapshot(fn, (dump_prefix + ".cleanup." + std::to_string(iter) + ".ll").str());
        if (!full_concretize && get_constant_i32_return(fn))
            break;
        if (detect_vmp_branch(fn, memory_))
            break;
    }
}

// ---------------------------------------------------------------------------
// Helpers for detect_vmp_branch

static std::optional<uint32_t> read_u32le(const Memory &memory, uint32_t addr)
{
    uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i)
    {
        auto it = memory.find(static_cast<uint64_t>(addr) + i);
        if (it == memory.end())
            return std::nullopt;
        v |= static_cast<uint32_t>(it->second) << (i * 8);
    }
    return v;
}

static uint32_t rotl32(uint32_t v, unsigned n)
{
    n &= 31u;
    if (n == 0)
        return v;
    return (v << n) | (v >> (32u - n));
}

static std::optional<uint32_t> decrypt_handler_addr(uint32_t vsp, uint32_t addend, unsigned rot_bits, uint32_t sub_const,
                                                    const Memory &memory)
{
    auto v = read_u32le(memory, vsp - 4);
    if (!v)
        return std::nullopt;
    uint32_t xor_ = vsp ^ *v;
    uint32_t add_ = xor_ + addend;
    uint32_t rot_ = rotl32(add_, rot_bits);
    return sub_const - rot_;
}

// Walk Value users to find a single BinaryOperator with given opcode where one
// operand is `val` and the other is a constant matching `imm` (or any constant
// when `match_any_const` is true). Returns {inst, const_val}.
static std::vector<std::pair<llvm::BinaryOperator *, uint64_t>> find_binop_users(
    llvm::Value *val, llvm::Instruction::BinaryOps opcode,
    bool match_any_const = false, uint64_t imm = 0
)
{
    std::vector<std::pair<llvm::BinaryOperator *, uint64_t>> out;
    for (auto *u : val->users())
    {
        auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
        if (!bo || bo->getOpcode() != opcode)
            continue;
        // Check either operand for the constant
        for (unsigned i = 0; i < 2; ++i)
        {
            auto *ci = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(i));
            if (!ci)
                continue;
            uint64_t cv = ci->getZExtValue();
            if (match_any_const || cv == imm)
                out.push_back({bo, cv});
        }
    }
    return out;
}

static std::pair<llvm::BinaryOperator *, uint64_t> find_binop_user(
    llvm::Value *val, llvm::Instruction::BinaryOps opcode,
    bool match_any_const = false, uint64_t imm = 0
)
{
    auto matches = find_binop_users(val, opcode, match_any_const, imm);
    if (matches.empty())
        return {nullptr, 0};
    return matches.front();
}

static unsigned force_vmp_branch_conditions(llvm::Function &fn, unsigned forced_shift, bool forced_value)
{
    std::vector<llvm::BinaryOperator *> matches;
    for (auto &bb : fn)
    {
        for (auto &inst : bb)
        {
            auto *lshr = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
            if (!lshr || lshr->getOpcode() != llvm::Instruction::LShr || !lshr->isExact() || lshr->use_empty())
                continue;
            auto *shift_ci = llvm::dyn_cast<llvm::ConstantInt>(lshr->getOperand(1));
            if (!shift_ci || shift_ci->getZExtValue() != forced_shift)
                continue;

            llvm::BinaryOperator *neg_mask = nullptr;
            llvm::BinaryOperator *inv_mask = nullptr;
            for (auto *u : lshr->users())
            {
                auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
                if (!bo)
                    continue;
                if (bo->getOpcode() == llvm::Instruction::Sub && bo->hasNoSignedWrap())
                {
                    auto *zero = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(0));
                    if (zero && zero->isZero() && bo->getOperand(1) == lshr)
                        neg_mask = bo;
                }
                else if (bo->getOpcode() == llvm::Instruction::Add && bo->hasNoSignedWrap())
                {
                    auto *ci = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(1));
                    if (bo->getOperand(0) == lshr && ci && ci->isAllOnesValue())
                        inv_mask = bo;
                }
            }
            if (!neg_mask || !inv_mask)
                continue;
            auto [masked_taken, ignored_taken] = find_binop_user(neg_mask, llvm::Instruction::And, true);
            auto [masked_fall, ignored_fall] = find_binop_user(inv_mask, llvm::Instruction::And, true);
            if (!masked_taken || !masked_fall)
                continue;
            bool has_vsp_select = false;
            for (auto *u : masked_taken->users())
            {
                auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
                if (bo && bo->getOpcode() == llvm::Instruction::Add &&
                    (bo->getOperand(0) == masked_fall || bo->getOperand(1) == masked_fall))
                {
                    has_vsp_select = true;
                    break;
                }
            }
            if (has_vsp_select)
                matches.push_back(lshr);
        }
    }

    // In arm-prefix discovery the selector may have been partially optimized
    // before this pass runs, leaving only the isolated flag bit feeding the
    // selected-arm arithmetic. If there is exactly one exact lshr by the branch
    // flag shift, it is the arm condition in the current VMP prefix; force it.
    if (matches.empty())
    {
        std::vector<llvm::BinaryOperator *> exact_shift_matches;
        for (auto &bb : fn)
        {
            for (auto &inst : bb)
            {
                auto *lshr = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
                if (!lshr || lshr->getOpcode() != llvm::Instruction::LShr || !lshr->isExact() || lshr->use_empty())
                    continue;
                auto *shift_ci = llvm::dyn_cast<llvm::ConstantInt>(lshr->getOperand(1));
                if (shift_ci && shift_ci->getZExtValue() == forced_shift)
                    exact_shift_matches.push_back(lshr);
            }
        }
        if (exact_shift_matches.size() == 1)
            matches.push_back(exact_shift_matches.front());
    }

    for (auto *lshr : matches)
    {
        auto *c = llvm::ConstantInt::get(lshr->getType(), forced_value ? 1 : 0);
        lshr->replaceAllUsesWith(c);
    }
    return matches.size();
}

static bool selector_inverts_packed_flag(llvm::Value *selector_source, unsigned flag_shift)
{
    // Common optimized form for testing an unset EFLAGS bit:
    //   %masked_or = or i32 %x, C        ; C has the selected bit clear
    //   %not       = xor i32 %masked_or, -1
    //   %bit       = lshr exact i32 %not, flag_shift
    // In that case the branchless selector is 1 when the packed EFLAGS bit is 0.
    auto *not_op = llvm::dyn_cast<llvm::BinaryOperator>(selector_source);
    if (!not_op || not_op->getOpcode() != llvm::Instruction::Xor)
        return false;
    auto *all_ones = llvm::dyn_cast<llvm::ConstantInt>(not_op->getOperand(1));
    if (!all_ones || !all_ones->isAllOnesValue())
        return false;

    auto *or_op = llvm::dyn_cast<llvm::BinaryOperator>(not_op->getOperand(0));
    if (!or_op || or_op->getOpcode() != llvm::Instruction::Or)
        return false;
    auto *mask = llvm::dyn_cast<llvm::ConstantInt>(or_op->getOperand(1));
    if (!mask)
        return false;

    return !mask->getValue()[flag_shift];
}

std::optional<VmpBranchInfo> detect_vmp_branch(llvm::Function &fn, const Memory &memory)
{
    // Find the return instruction to know which value is ultimately returned.
    llvm::ReturnInst *ret_inst = nullptr;
    for (auto &bb : fn)
    {
        if (auto *ri = llvm::dyn_cast<llvm::ReturnInst>(bb.getTerminator()))
        {
            ret_inst = ri;
            break;
        }
    }
    if (!ret_inst || ret_inst->getNumOperands() == 0)
        return std::nullopt;

    llvm::Value *ret_val = ret_inst->getOperand(0);

    // Walk every instruction looking for `lshr exact i32 X, N`
    for (auto &bb : fn)
    {
        for (auto &inst : bb)
        {
            auto *lshr = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
            if (!lshr || lshr->getOpcode() != llvm::Instruction::LShr || !lshr->isExact())
                continue;
            auto *shift_ci = llvm::dyn_cast<llvm::ConstantInt>(lshr->getOperand(1));
            if (!shift_ci)
                continue;
            unsigned flag_shift = static_cast<unsigned>(shift_ci->getZExtValue());

            // Find `sub nsw i32 0, lshr`  → neg_mask
            llvm::BinaryOperator *neg_mask = nullptr;
            for (auto *u : lshr->users())
            {
                auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
                if (!bo || bo->getOpcode() != llvm::Instruction::Sub || !bo->hasNoSignedWrap())
                    continue;
                auto *zero = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(0));
                if (zero && zero->isZero() && bo->getOperand(1) == lshr)
                {
                    neg_mask = bo;
                    break;
                }
            }
            if (!neg_mask)
                continue;

            // Find `add nsw i32 lshr, -1`  → inv_mask
            llvm::BinaryOperator *inv_mask = nullptr;
            for (auto *u : lshr->users())
            {
                auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
                if (!bo || bo->getOpcode() != llvm::Instruction::Add || !bo->hasNoSignedWrap())
                    continue;
                // operand 0 must be lshr, operand 1 must be -1 (0xFFFFFFFF for i32)
                if (bo->getOperand(0) != lshr)
                    continue;
                auto *ci = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(1));
                if (ci && ci->isAllOnesValue())
                {
                    inv_mask = bo;
                    break;
                }
            }
            if (!inv_mask)
                continue;

            // Find the selected VSP add. The mask values often have many
            // unrelated users from flag maintenance (`and mask, 16`, etc.), so
            // try all constant AND users instead of accepting the first one.
            auto taken_ands = find_binop_users(neg_mask, llvm::Instruction::And, true);
            auto fall_ands  = find_binop_users(inv_mask, llvm::Instruction::And, true);
            for (auto [masked_taken, addr_taken_val] : taken_ands)
            for (auto [masked_fall, addr_fall_val] : fall_ands)
            {
                uint32_t addr_taken = static_cast<uint32_t>(addr_taken_val);
                uint32_t addr_fall  = static_cast<uint32_t>(addr_fall_val);
                if (!memory.count(addr_taken) || !memory.count(addr_fall))
                    continue;

                // Find `add i32 masked_taken, masked_fall`  → vsp_sel
                llvm::BinaryOperator *vsp_sel = nullptr;
                for (auto *u : masked_taken->users())
                {
                    auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
                    if (!bo || bo->getOpcode() != llvm::Instruction::Add)
                        continue;
                    bool has_fall = (bo->getOperand(0) == masked_fall || bo->getOperand(1) == masked_fall);
                    if (has_fall)
                    {
                        vsp_sel = bo;
                        break;
                    }
                }
                if (!vsp_sel)
                    continue;

            // Walk the EIP decrypt chain from vsp_sel:
            // ptr_var  = add i32 vsp_sel, -4
            // gep_var  = getelementptr ... ptr_var
            // load_var = load i32, ptr gep_var
            // xor_var  = xor i32 vsp_sel, load_var
            // add_var  = add i32 xor_var, addend
            // rot_var  = call @llvm.fshl.i32(add_var, add_var, rot_bits)
            // eip_var  = sub i32 sub_const, rot_var
            // verify eip_var == ret_val

            // ptr_var = add i32 vsp_sel, -4
            llvm::BinaryOperator *ptr_var = nullptr;
            for (auto *u : vsp_sel->users())
            {
                auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
                if (!bo || bo->getOpcode() != llvm::Instruction::Add)
                    continue;
                auto *ci = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(1));
                if (ci && ci->getSExtValue() == -4)
                {
                    ptr_var = bo;
                    break;
                }
            }
            if (!ptr_var)
                continue;

            // gep_var using ptr_var
            llvm::GetElementPtrInst *gep_var = nullptr;
            for (auto *u : ptr_var->users())
            {
                if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(u))
                {
                    gep_var = gep;
                    break;
                }
            }
            if (!gep_var)
                continue;

            // load_var = load i32 from gep_var
            llvm::LoadInst *load_var = nullptr;
            for (auto *u : gep_var->users())
            {
                if (auto *li = llvm::dyn_cast<llvm::LoadInst>(u))
                {
                    load_var = li;
                    break;
                }
            }
            if (!load_var)
                continue;

            // xor_var = xor i32 vsp_sel, load_var  (either order)
            llvm::BinaryOperator *xor_var = nullptr;
            for (auto *u : load_var->users())
            {
                auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
                if (!bo || bo->getOpcode() != llvm::Instruction::Xor)
                    continue;
                bool uses_vsp = (bo->getOperand(0) == vsp_sel || bo->getOperand(1) == vsp_sel);
                if (uses_vsp)
                {
                    xor_var = bo;
                    break;
                }
            }
            if (!xor_var)
                continue;

            // add_var = add i32 xor_var, addend
            auto [add_var, addend_val] = find_binop_user(xor_var, llvm::Instruction::Add, true);
            if (!add_var)
                continue;
            uint32_t addend = static_cast<uint32_t>(addend_val);

            // rot_var = call @llvm.fshl.i32(add_var, add_var, rot_bits)
            llvm::CallInst *rot_var = nullptr;
            unsigned        rot_bits = 0;
            for (auto *u : add_var->users())
            {
                auto *ci = llvm::dyn_cast<llvm::CallInst>(u);
                if (!ci)
                    continue;
                auto *fn_called = ci->getCalledFunction();
                if (!fn_called || !fn_called->getName().starts_with("llvm.fshl.i32"))
                    continue;
                // fshl(a, a, n) — both data args should be add_var
                if (ci->getArgOperand(0) != add_var || ci->getArgOperand(1) != add_var)
                    continue;
                auto *n_ci = llvm::dyn_cast<llvm::ConstantInt>(ci->getArgOperand(2));
                if (!n_ci)
                    continue;
                rot_bits = static_cast<unsigned>(n_ci->getZExtValue());
                rot_var  = ci;
                break;
            }
            if (!rot_var)
                continue;

            // eip_var = sub i32 sub_const, rot_var
            llvm::BinaryOperator *eip_var = nullptr;
            uint32_t              sub_const = 0;
            for (auto *u : rot_var->users())
            {
                auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
                if (!bo || bo->getOpcode() != llvm::Instruction::Sub)
                    continue;
                auto *c = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(0));
                if (c && bo->getOperand(1) == rot_var)
                {
                    sub_const = static_cast<uint32_t>(c->getZExtValue());
                    eip_var   = bo;
                    break;
                }
            }
            if (!eip_var)
                continue;

            // Verify this is indeed the returned value
            if (eip_var != ret_val)
                continue;

            // Decrypt both candidate handler addresses
            auto next_taken = decrypt_handler_addr(addr_taken, addend, rot_bits, sub_const, memory);
            auto next_fall  = decrypt_handler_addr(addr_fall,  addend, rot_bits, sub_const, memory);
            if (!next_taken || !next_fall)
                continue;

            // Validate both are in the PE image
            if (!memory.count(*next_taken) || !memory.count(*next_fall))
                continue;

            bool selector_inverts_flag = selector_inverts_packed_flag(lshr->getOperand(0), flag_shift);
            std::cout << "[*] Detected VMP branch: taken=0x" << std::hex << *next_taken
                      << " fall=0x" << *next_fall
                      << " flag_bit=" << std::dec << flag_shift
                      << (selector_inverts_flag ? " inverted" : "") << "\n";

            return VmpBranchInfo{*next_taken, *next_fall, flag_shift, selector_inverts_flag, addr_taken, addr_fall};
            }
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------

PrefixResult DiscoveryEngine::run_step(
    const PrefixDiscoveryBuilder           &build_prefix,
    std::optional<std::pair<unsigned, bool>> forced_branch_condition,
    bool                                     full_concretize
)
{
    unsigned          step = step_counter_++;
    const std::string name = "__prefix_next_" + std::to_string(step);
    auto             *fn   = build_prefix(name);
    if (save_intermediate_)
        dump_function_snapshot(*fn, "out.prefix." + std::to_string(step) + ".before.ll");
    optimize(*fn, save_intermediate_ ? "out.prefix." + std::to_string(step) : "",
             forced_branch_condition, full_concretize);
    if (save_intermediate_)
        dump_function_snapshot(*fn, "out.prefix." + std::to_string(step) + ".after.ll");

    if (auto c = get_constant_i32_return(*fn))
    {
        auto snapshot = extract_prefix_snapshot(*fn);
        fn->dropAllReferences();
        fn->eraseFromParent();
        return {PrefixResult::Kind::Concrete, *c, {}, std::move(snapshot)};
    }

    if (auto b = detect_vmp_branch(*fn, memory_))
    {
        fn->dropAllReferences();
        fn->eraseFromParent();
        return {PrefixResult::Kind::Branch, 0, *b, std::nullopt};
    }

    fn->dropAllReferences();
    fn->eraseFromParent();
    return {};  // Kind::Unknown
}
