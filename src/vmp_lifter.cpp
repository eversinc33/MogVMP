#include "vmp_lifter.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/ModRef.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/ConstraintElimination.h>
#include <llvm/Transforms/Scalar/CorrelatedValuePropagation.h>
#include <llvm/Transforms/Scalar/DeadStoreElimination.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Scalar/JumpThreading.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <remill/Arch/Arch.h>
#include <remill/BC/IntrinsicTable.h>
#include <remill/BC/Lifter.h>
#include <remill/BC/Optimizer.h>
#include <remill/BC/Util.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "handler_discovery.h"
#include "instruction_decoder.h"
#include "llvm_pipeline_utils.h"
#include "passes/concrete_alloca_prop.h"
#include "passes/readable_block_names.h"
#include "passes/readable_register_names.h"

//==---------------------------------------------------------------------------==//
// VmBlock
//==---------------------------------------------------------------------------==//

struct StateInjectPatch
{
    uint32_t offset = 0;
    uint8_t clear_mask = 0xff;
    uint8_t set_bits = 0;
    bool overwrite = true;
};

struct VmBlock
{
    std::vector<uint64_t> handlers;

    bool at_vmexit = false;

    std::unique_ptr<VmBlock> next;

    VmpBranchInfo            branch_info = {};
    std::unique_ptr<VmBlock> taken; // arm when flag bit = 1
    std::unique_ptr<VmBlock> fall; // arm when flag bit = 0

    std::optional<size_t>        inject_before_idx;
    std::vector<StateInjectPatch> state_inject;

    bool is_branch()   const { return taken != nullptr; }
    bool is_terminal() const { return at_vmexit; }
};

//==---------------------------------------------------------------------------==//
// TraceManager
//==---------------------------------------------------------------------------==//

struct TraceManager : remill::TraceManager
{
    const Memory                                  &memory;
    llvm::Module                                  *module;
    const remill::Arch                            *arch;
    std::unordered_set<uint64_t>                   lift_set;
    std::unordered_map<uint64_t, llvm::Function *> traces;

    TraceManager(const Memory &mem, llvm::Module *mod, const remill::Arch *a) : memory(mem), module(mod), arch(a)
    {
    }

    void SetLiftedTraceDefinition(uint64_t addr, llvm::Function *fn) override
    {
        traces[addr] = fn;
    }

    llvm::Function *GetLiftedTraceDefinition(uint64_t addr) override
    {
        if (lift_set.count(addr))
            return nullptr;
        auto  name = TraceName(addr);
        auto *fn = module->getFunction(name);
        if (!fn)
            fn = arch->DeclareLiftedFunction(name, module);
        return fn;
    }

    bool TryReadExecutableByte(uint64_t addr, uint8_t *byte) override
    {
        auto it = memory.find(addr);
        if (it == memory.end())
            return false;
        if (byte)
            *byte = it->second;
        return true;
    }
};

//==---------------------------------------------------------------------------==//

static void dump_module_snapshot(llvm::Module &module, const char *path)
{
    std::error_code      ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec)
    {
        std::cerr << "Failed to write IR snapshot '" << path << "': " << ec.message() << "\n";
        return;
    }
    module.print(os, nullptr);
}

/**
 * Removes the freeze from a 'ret i32 freeze(%x)'
 *
 * @param fn Function
 * @return True if changed
 */
static bool unwrap_return_freezes(llvm::Function &fn)
{
    bool changed = false;
    for (auto &bb : fn)
    {
        auto *ret = llvm::dyn_cast<llvm::ReturnInst>(bb.getTerminator());
        if (!ret || ret->getNumOperands() == 0)
            continue;
        if (auto *freeze = llvm::dyn_cast<llvm::FreezeInst>(ret->getOperand(0)))
        {
            ret->setOperand(0, freeze->getOperand(0));
            changed = true;
        }
    }
    return changed;
}

static void write_u32le(Memory &memory, uint64_t addr, uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i) memory[addr + i] = static_cast<uint8_t>(value >> (i * 8));
}

static Memory patch_push_ret_dispatchers(const Memory &memory, unsigned *patched_count = nullptr)
{
    Memory   patched = memory;
    unsigned count = 0;

    auto it = patched.begin();
    while (it != patched.end())
    {
        const uint64_t pc = it->first;
        auto           inst = InstructionDecoder::decode(patched, pc);
        if (!inst)
        {
            ++it;
            continue;
        }

        if (inst->length == 1)
        {
            auto reg = InstructionDecoder::push_register_index(*inst);
            if (reg)
            {
                auto ret = InstructionDecoder::decode(patched, pc + inst->length);
                if (ret && ret->length == 1 && InstructionDecoder::is_ret(*ret))
                {
                    // push r32; ret -> jmp r32. This keeps the instruction length at two bytes and makes Remill see a normal indirect jump instead of a synthetic stack-based dispatch
                    patched[pc] = 0xFF;
                    patched[pc + 1] = static_cast<uint8_t>(0xE0 + *reg);
                    ++count;
                    it = patched.lower_bound(pc + 2);
                    continue;
                }
            }
        }

        it = patched.lower_bound(pc + inst->length);
    }

    if (patched_count)
        *patched_count = count;
    return patched;
}

/**
 * Finds the first call instruction in a vmentry prelude, following any encountered jmps.
 * This can then later be patched with a jmp, because the junk after the call is never reached
 * in the actual control flow
 *
 * @param memory
 * @param start
 * @return
 */
static std::optional<std::pair<uint64_t, uint64_t>> find_first_direct_call_in_prelude(
    const Memory &memory, uint64_t start
)
{
    uint64_t pc = start;
    for (unsigned scanned = 0; scanned < 0x200;)
    {
        auto inst = InstructionDecoder::decode(memory, pc);
        if (!inst)
            return std::nullopt;

        if (InstructionDecoder::is_direct_call(*inst) && memory.count(*inst->branch_target))
            return std::pair<uint64_t, uint64_t>{pc, *inst->branch_target};

        if (InstructionDecoder::is_unconditional_jmp(*inst) && inst->branch_target && memory.count(*inst->branch_target))
        {
            scanned += inst->length;
            pc = *inst->branch_target;
            continue;
        }

        scanned += inst->length;
        pc += inst->length;
    }
    return std::nullopt;
}

/**
 * Patches calls identified by find_first_direct_call_in_prelude with a jmp
 *
 * @param memory
 * @param vmenter
 * @param patched_count
 * @return
 */
static Memory patch_nonreturn_vmentry_calls(const Memory &memory, uint64_t vmenter, unsigned *patched_count = nullptr)
{
    Memory   patched = memory;
    unsigned count = 0;

    if (auto call = find_first_direct_call_in_prelude(memory, vmenter))
    {
        const auto [call_site, call_target] = *call;
        const auto ret_addr = static_cast<uint32_t>(call_site + 5);
        const auto target = static_cast<uint32_t>(call_target);
        const auto jmp_disp = static_cast<int32_t>(target - static_cast<uint32_t>(ret_addr + 5));

        // VMENTRY calls do not return to the post-call junk in the native wrapper, branch to the VMENTRY body instead
        patched[call_site] = 0x68;  // push imm32
        write_u32le(patched, call_site + 1, ret_addr);
        patched[ret_addr] = 0xE9;  // jmp rel32
        write_u32le(patched, ret_addr + 1, static_cast<uint32_t>(jmp_disp));
        ++count;
    }

    if (patched_count != nullptr)
        *patched_count = count;
    return patched;
}

/**
 * Heuristically identify a vmexit handler by counting the number of pop instructions - if more
 * than 6, it is likely a vmexit
 * @param memory
 * @param addr
 * @return
 */
static bool looks_like_vmexit_restore_handler(const Memory &memory, uint64_t addr)
{
    // The guest-register restore (POPAD-style) does many pops and ends in a `ret`
    // into the native epilogue. VMProtect splits it across several obfuscated
    // blocks linked by unconditional jmps, so follow that jmp chain (not just one
    // hop), accumulating pops, until the terminating `ret`.
    unsigned                     pops = 0;
    bool                         has_ret = false;
    std::unordered_set<uint64_t> seen_blocks;
    uint64_t                     pc = addr;
    constexpr unsigned           kMaxBlocks = 4;

    for (unsigned blocks = 0; blocks < kMaxBlocks && seen_blocks.insert(pc).second; ++blocks)
    {
        bool followed_jmp = false;
        for (uint64_t scanned = 0; scanned < 0x80;)
        {
            auto inst = InstructionDecoder::decode(memory, pc);
            if (!inst)
            {
                ++pc;
                ++scanned;
                continue;
            }
            if (InstructionDecoder::is_pop(*inst))
                ++pops;
            if (InstructionDecoder::is_ret(*inst))
            {
                has_ret = true;
                break;
            }
            if (InstructionDecoder::is_unconditional_jmp(*inst) && inst->branch_target)
            {
                pc            = *inst->branch_target;
                followed_jmp  = true;
                break;
            }
            pc += inst->length;
            scanned += inst->length;
        }
        if (has_ret || !followed_jmp)
            break;
    }

    return has_ret && pops > 6;
}

struct DirectCallSite
{
    uint32_t address = 0;
    uint32_t return_address = 0;
};

struct HostRegisterBinding
{
    std::string reg_name;
    unsigned    arg_index = 0;
};

static std::optional<DirectCallSite> find_unique_direct_call_site(const Memory &memory, uint64_t target)
{
    std::optional<DirectCallSite> call_site;
    for (const auto &[addr, _] : memory)
    {
        auto inst = InstructionDecoder::decode(memory, addr);
        if (!inst || !InstructionDecoder::is_direct_call(*inst) || *inst->branch_target != static_cast<uint32_t>(target))
            continue;

        DirectCallSite candidate{static_cast<uint32_t>(addr), static_cast<uint32_t>(addr + inst->length)};
        if (call_site &&
            (call_site->address != candidate.address || call_site->return_address != candidate.return_address))
            return std::nullopt;
        call_site = candidate;
    }
    return call_site;
}

static const char *x86_reg_name_from_index(unsigned reg_index)
{
    switch (reg_index)
    {
        case 0: return "EAX";
        case 1: return "ECX";
        case 2: return "EDX";
        case 3: return "EBX";
        case 4: return "ESP";
        case 5: return "EBP";
        case 6: return "ESI";
        case 7: return "EDI";
        default: return nullptr;
    }
}

static std::vector<DecodedInstruction> decode_stream_ending_at(
    const Memory &memory, uint32_t window_start, uint32_t end_addr
)
{
    std::vector<DecodedInstruction> best;
    for (uint32_t start = window_start; start < end_addr; ++start)
    {
        std::vector<DecodedInstruction> stream;
        uint32_t                        pc = start;
        bool                            ok = false;
        while (pc < end_addr)
        {
            auto inst = InstructionDecoder::decode(memory, pc);
            if (!inst || inst->length == 0)
                break;
            stream.push_back(*inst);
            pc += inst->length;
            if (pc == end_addr)
            {
                ok = true;
                break;
            }
        }
        if (ok && stream.size() >= best.size())
            best = std::move(stream);
    }
    return best;
}

static std::vector<HostRegisterBinding> infer_argument_register_bindings(
    const Memory &memory, const DirectCallSite &call_site, unsigned param_count
)
{
    std::vector<HostRegisterBinding> bindings;
    if (!param_count)
        return bindings;

    const uint32_t window_start = call_site.address > 0x100 ? call_site.address - 0x100 : 0;
    auto           stream = decode_stream_ending_at(memory, window_start, call_site.address);
    if (stream.empty())
        return bindings;

    unsigned arg_index = 0;
    for (auto it = stream.rbegin(); it != stream.rend() && arg_index < param_count; ++it)
    {
        auto reg_index = InstructionDecoder::push_register_index(*it);
        if (!reg_index)
            break;
        if (const char *reg_name = x86_reg_name_from_index(*reg_index))
            bindings.push_back({reg_name, arg_index});
        ++arg_index;
    }
    return bindings;
}

static unsigned count_argument_pushes(const Memory &memory, const DirectCallSite &call_site)
{
    const uint32_t window_start = call_site.address > 0x100 ? call_site.address - 0x100 : 0;
    auto           stream = decode_stream_ending_at(memory, window_start, call_site.address);
    unsigned       count  = 0;
    for (auto it = stream.rbegin(); it != stream.rend(); ++it)
    {
        if (!InstructionDecoder::push_register_index(*it))
            break;
        ++count;
    }
    return count;
}

//==---------------------------------------------------------------------------==//

static constexpr uint32_t kSyntheticEsp = 0x0018FF00;
static constexpr uint32_t kSyntheticStackSize = 0x10000;
static constexpr uint32_t kSyntheticStackBase = (kSyntheticEsp - (kSyntheticStackSize / 2)) & ~15u;
static constexpr uint32_t kSyntheticParamBase = kSyntheticEsp + 4;

static VirtualMemoryLayout build_virtual_memory_layout(llvm::LLVMContext &ctx, const Memory &memory)
{
    auto *i8Ty = llvm::Type::getInt8Ty(ctx);
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);

    uint64_t pe_top = memory.rbegin()->first + 1;
    uint64_t stack_top = static_cast<uint64_t>(kSyntheticStackBase) + kSyntheticStackSize;

    // Start at zero so low VM/bytecode addresses (e.g. ESI/EDI around 0x100) are represented as ordinary zero-initialized bytes instead of becoming
    // negative GEPs from a higher PE/stack base.
    uint64_t mem_base = 0;
    uint64_t mem_top = std::max<uint64_t>(pe_top, stack_top);
    uint64_t mem_span = mem_top - mem_base;

    std::vector<uint8_t> flat(mem_span, 0);
    for (auto &[addr, byte] : memory) flat[addr - mem_base] = byte;

    auto *array_ty = llvm::ArrayType::get(i8Ty, mem_span);
    auto *init = llvm::ConstantDataArray::get(ctx, llvm::ArrayRef<uint8_t>(flat.data(), flat.size()));

    std::cout << "[*] Virtual memory: 0x" << std::hex << mem_base << " - 0x" << mem_top << " (" << std::dec
              << (mem_span / 1024) << " KB)\n";
    std::cout << "[*] Synthetic stack: ESP=0x" << std::hex << kSyntheticEsp << ", args=0x" << kSyntheticParamBase
              << std::dec << "\n";

    return VirtualMemoryLayout{
        array_ty,
        init,
        llvm::ConstantInt::get(i32Ty, static_cast<uint32_t>(mem_base)),
        llvm::ConstantInt::get(i32Ty, static_cast<uint32_t>(mem_span)),
    };
}

static unsigned erase_unused_internal_functions(llvm::Module &module)
{
    unsigned erased = 0;
    bool     changed = true;
    while (changed)
    {
        changed = false;
        for (auto it = module.begin(), end = module.end(); it != end;)
        {
            llvm::Function &fn = *it++;
            if (fn.isDeclaration() || !fn.use_empty() || !fn.hasInternalLinkage())
                continue;
            fn.eraseFromParent();
            ++erased;
            changed = true;
        }
    }
    return erased;
}

static unsigned erase_unused_undefined8_calls(llvm::Module &module)
{
    auto *fn = module.getFunction("__remill_undefined_8");
    if (!fn)
        return 0;

    llvm::SmallVector<llvm::CallInst *, 64> unused;
    for (auto *user : fn->users())
        if (auto *ci = llvm::dyn_cast<llvm::CallInst>(user))
            if (ci->use_empty())
                unused.push_back(ci);

    for (auto *ci : unused) ci->eraseFromParent();

    if (fn->use_empty())
        fn->eraseFromParent();

    return unused.size();
}

static void neutralize_dispatch_intrinsics(llvm::Module &module)
{
    for (const char *name : {"__remill_jump", "__remill_function_return"})
    {
        auto *fn = module.getFunction(name);
        if (!fn)
            continue;

        llvm::SmallVector<llvm::CallBase *, 64> calls;
        for (auto *user : fn->users())
            if (auto *cb = llvm::dyn_cast<llvm::CallBase>(user))
                calls.push_back(cb);

        for (auto *cb : calls)
        {
            if (cb->getType() != cb->getArgOperand(2)->getType())
                continue;
            cb->replaceAllUsesWith(cb->getArgOperand(2));
            cb->eraseFromParent();
        }
    }
}

static void collect_branch_infos(const VmBlock &block, std::vector<VmpBranchInfo> &out)
{
    if (block.is_branch())
    {
        out.push_back(block.branch_info);
        if (block.taken)
            collect_branch_infos(*block.taken, out);
        if (block.fall)
            collect_branch_infos(*block.fall, out);
    }
    if (block.next)
        collect_branch_infos(*block.next, out);
}

static llvm::ConstantInt *get_i32_const_operand(llvm::Value *v)
{
    return llvm::dyn_cast<llvm::ConstantInt>(v);
}

static bool is_and_with_const(llvm::Value *v, uint32_t c)
{
    auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(v);
    if (!bo || bo->getOpcode() != llvm::Instruction::And)
        return false;
    for (unsigned i = 0; i < 2; ++i)
    {
        auto *ci = get_i32_const_operand(bo->getOperand(i));
        if (ci && static_cast<uint32_t>(ci->getZExtValue()) == c)
            return true;
    }
    return false;
}

static bool replace_uses_in_block(llvm::Value *from, llvm::BasicBlock *bb, llvm::Value *to)
{
    std::vector<llvm::Use *> uses;
    for (auto &use : from->uses())
    {
        auto *user_inst = llvm::dyn_cast<llvm::Instruction>(use.getUser());
        if (user_inst && user_inst->getParent() == bb)
            uses.push_back(&use);
    }
    for (auto *use : uses)
        use->set(to);
    return !uses.empty();
}

static unsigned materialize_branch_vsp_constants(llvm::Function &fn, const VmBlock &root)
{
    std::vector<VmpBranchInfo> branches;
    collect_branch_infos(root, branches);
    if (branches.empty())
        return 0;

    unsigned changed = 0;
    auto    *i32Ty = llvm::Type::getInt32Ty(fn.getContext());

    std::cerr << "[DBG] materialize_branch_vsp_constants: " << branches.size() << " branch(es)\n";
    for (const auto &branch : branches)
    {
        std::cerr << "[DBG]   branch vsp_taken=0x" << std::hex << branch.vsp_taken
                  << " vsp_fall=0x" << branch.vsp_fall << std::dec << "\n";
        llvm::BasicBlock *taken_bb = nullptr;
        llvm::BasicBlock *fall_bb = nullptr;
        for (auto &bb : fn)
        {
            if (bb.getName() == "branch_taken" && !taken_bb)
                taken_bb = &bb;
            else if (bb.getName() == "branch_fall" && !fall_bb)
                fall_bb = &bb;
        }
        if (!taken_bb || !fall_bb)
        {
            std::cerr << "[DBG]   taken_bb=" << (void *)taken_bb << " fall_bb=" << (void *)fall_bb
                      << " (one missing, skipping)\n";
            continue;
        }

        unsigned scanned_adds = 0;
        unsigned matched_adds = 0;
        for (auto &bb : fn)
        {
            for (auto &inst : bb)
            {
                auto *add = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
                if (!add || add->getOpcode() != llvm::Instruction::Add || !add->getType()->isIntegerTy(32))
                    continue;
                ++scanned_adds;

                bool op0_taken = is_and_with_const(add->getOperand(0), branch.vsp_taken);
                bool op1_taken = is_and_with_const(add->getOperand(1), branch.vsp_taken);
                bool op0_fall  = is_and_with_const(add->getOperand(0), branch.vsp_fall);
                bool op1_fall  = is_and_with_const(add->getOperand(1), branch.vsp_fall);
                if (!((op0_taken && op1_fall) || (op0_fall && op1_taken)))
                    continue;
                ++matched_adds;
                std::cerr << "[DBG]   matched selection add in block '" << add->getParent()->getName().str()
                          << "'\n";

                auto *taken_c = llvm::ConstantInt::get(i32Ty, branch.vsp_taken);
                auto *fall_c  = llvm::ConstantInt::get(i32Ty, branch.vsp_fall);
                bool did = false;
                did |= replace_uses_in_block(add, taken_bb, taken_c);
                did |= replace_uses_in_block(add, fall_bb, fall_c);
                if (did)
                {
                    ++changed;
                    std::cout << "[*] Materialized branch VSP constants: taken=0x" << std::hex << branch.vsp_taken
                              << " fall=0x" << branch.vsp_fall << std::dec << "\n";
                }
                else
                {
                    std::cerr << "[DBG]   matched add but replace_uses_in_block changed nothing "
                                 "(add has no uses inside arm blocks)\n";
                }
            }
        }
        std::cerr << "[DBG]   scanned " << scanned_adds << " i32 adds, matched " << matched_adds << "\n";
    }
    return changed;
}

static llvm::Value *find_arm_decrypt_vsp(llvm::BasicBlock *arm_entry, const llvm::DominatorTree &dt)
{
    auto is_add_minus4 = [](llvm::Value *v, llvm::User *u) -> bool
    {
        auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
        if (!bo || bo->getOpcode() != llvm::Instruction::Add)
            return false;
        unsigned other = bo->getOperand(0) == v ? 1 : 0;
        auto *ci = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(other));
        return ci && ci->getSExtValue() == -4;
    };
    auto is_xor_user = [](llvm::Value *v, llvm::User *u) -> bool
    {
        auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(u);
        return bo && bo->getOpcode() == llvm::Instruction::Xor &&
               (bo->getOperand(0) == v || bo->getOperand(1) == v);
    };

    // Detect the VSP value consumed by the branch/dispatch handler's EIP-decrypt chain inside a branch arm

    // Walk the arm entry block  and take the earliest value matching the signature
    // so we concretize the branch handler's VSP, not a later child handler VSP
    for (auto &inst : *arm_entry)
    {
        if (!inst.getType()->isIntegerTy(32))
            continue;
        bool has_addm4 = false, has_xor = false;
        for (auto *u : inst.users())
        {
            if (auto *ui = llvm::dyn_cast<llvm::Instruction>(u); !ui || !dt.dominates(arm_entry, ui->getParent()))
                continue;
            has_addm4 |= is_add_minus4(&inst, u);
            has_xor |= is_xor_user(&inst, u);
        }
        if (has_addm4 && has_xor)
            return &inst;
    }
    return nullptr;
}

/**
 * Replace every load of one of the segment-base offsets with 0 directly
 */
static unsigned flatten_segment_base_loads(llvm::Function &fn, const std::set<uint64_t> &seg_offsets)
{
    if (seg_offsets.empty())
        return 0;

    // Find the 4096-byte remill state alloca.
    llvm::AllocaInst *state_alloca = nullptr;
    for (auto &bb : fn)
        for (auto &inst : bb)
            if (auto *a = llvm::dyn_cast<llvm::AllocaInst>(&inst))
                if (a->getAllocatedType()->isArrayTy() &&
                    a->getAllocatedType()->getArrayNumElements() == 4096)
                {
                    state_alloca = a;
                    break;
                }
    if (!state_alloca)
        return 0;

    const auto &dl = fn.getParent()->getDataLayout();
    unsigned    changed = 0;
    for (auto &bb : fn)
    {
        for (auto &inst : llvm::make_early_inc_range(bb))
        {
            auto *li = llvm::dyn_cast<llvm::LoadInst>(&inst);
            if (!li || !li->getType()->isIntegerTy())
                continue;
            auto *ptr = li->getPointerOperand()->stripPointerCasts();
            auto *gep = llvm::dyn_cast<llvm::GEPOperator>(ptr);
            llvm::Value *base = gep ? gep->getPointerOperand()->stripPointerCasts() : ptr;
            if (base != state_alloca)
                continue;
            llvm::APInt off(dl.getIndexSizeInBits(gep ? gep->getPointerAddressSpace() : 0), 0);
            if (gep && !gep->accumulateConstantOffset(dl, off))
                continue;
            if (!seg_offsets.count(off.getZExtValue()))
                continue;
            li->replaceAllUsesWith(llvm::ConstantInt::get(li->getType(), 0));
            li->eraseFromParent();
            ++changed;
        }
    }
    return changed;
}

static void force_decrypt_vsp_in_arm(llvm::BasicBlock *arm, uint32_t vsp, const llvm::DominatorTree &dt,
                                     llvm::Type *i32Ty, unsigned &changed)
{
    if (!arm || !vsp)
        return;
    llvm::Value *v = find_arm_decrypt_vsp(arm, dt);
    if (!v)
        return;
    v->replaceAllUsesWith(llvm::ConstantInt::get(i32Ty, vsp));
    ++changed;
    std::cout << "[*] Forced branch arm '" << arm->getName().str() << "' decrypt VSP to 0x" << std::hex << vsp
              << std::dec << "\n";
}

static unsigned force_arm_decrypt_vsp(llvm::Function &fn, const VmBlock &root)
{
    std::vector<VmpBranchInfo> branches;
    collect_branch_infos(root, branches);
    if (branches.empty())
        return 0;

    llvm::DominatorTree dt(fn);
    auto               *i32Ty = llvm::Type::getInt32Ty(fn.getContext());
    unsigned            changed = 0;

    // TODO: we do not handle multiple branches yet beause we use hardcoded bb names here FIXME
    llvm::BasicBlock *taken_bb = nullptr;
    llvm::BasicBlock *fall_bb = nullptr;
    for (auto &bb : fn)
    {
        if (bb.getName() == "branch_taken" && !taken_bb)
            taken_bb = &bb;
        else if (bb.getName() == "branch_fall" && !fall_bb)
            fall_bb = &bb;
    }

    for (const auto &branch : branches)
    {
        force_decrypt_vsp_in_arm(taken_bb, branch.vsp_taken, dt, i32Ty, changed);
        force_decrypt_vsp_in_arm(fall_bb, branch.vsp_fall, dt, i32Ty, changed);
    }
    return changed;
}

static uint32_t const_from_md(llvm::Metadata *m)
{
    auto *cm = llvm::cast<llvm::ConstantAsMetadata>(m);
    return static_cast<uint32_t>(llvm::cast<llvm::ConstantInt>(cm->getValue())->getZExtValue());
}

// Run once the selector has been re-assembled into pure SSA (after SROA) but before InstCombine. Matches the same `add (and _,vsp_taken),
// (and _,vsp_fall)` pattern as materialize_branch_vsp_constants()
static unsigned rewrite_branch_conditions_on_selector(llvm::Function &fn)
{
    auto *i32Ty = llvm::Type::getInt32Ty(fn.getContext());
    llvm::DominatorTree dt(fn);
    unsigned changed = 0;

    for (auto &bb : fn)
    {
        auto *br = llvm::dyn_cast<llvm::BranchInst>(bb.getTerminator());
        if (!br || !br->isConditional())
            continue;
        auto *md = br->getMetadata("vmp_branch");
        if (!md || md->getNumOperands() < 2)
            continue;
        const uint32_t vsp_taken = const_from_md(md->getOperand(0));
        const uint32_t vsp_fall  = const_from_md(md->getOperand(1));

        // Find the selector add `(and _, vsp_taken) + (and _, vsp_fall)` that dominates this branch
        llvm::Value *selector = nullptr;
        for (auto &b2 : fn)
        {
            for (auto &inst : b2)
            {
                auto *add = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
                if (!add || add->getOpcode() != llvm::Instruction::Add || !add->getType()->isIntegerTy(32))
                    continue;
                bool op0_taken = is_and_with_const(add->getOperand(0), vsp_taken);
                bool op1_taken = is_and_with_const(add->getOperand(1), vsp_taken);
                bool op0_fall  = is_and_with_const(add->getOperand(0), vsp_fall);
                bool op1_fall  = is_and_with_const(add->getOperand(1), vsp_fall);
                if (!((op0_taken && op1_fall) || (op0_fall && op1_taken)))
                    continue;
                if (dt.dominates(add, br))
                {
                    selector = add;
                    break;
                }
            }
            if (selector)
                break;
        }
        if (!selector)
        {
            std::cerr << "[!] No dominating selector for branch in '" << bb.getName().str() << "'\n";
            continue;
        }

        // successor 0 (taken_bb) is the vsp_taken edge
        llvm::IRBuilder<> b(br);
        auto *cond = b.CreateICmpEQ(selector, llvm::ConstantInt::get(i32Ty, vsp_taken), "branch_on_selector");
        br->setCondition(cond);
        br->setMetadata("vmp_branch", nullptr);
        ++changed;
        std::cout << "[*] Re-keyed VM branch on VSP selector (taken=0x" << std::hex << vsp_taken
                  << " fall=0x" << vsp_fall << std::dec << ")\n";
    }
    return changed;
}

static void force_branch_flag_in_arm(llvm::Function &fn, llvm::BasicBlock *arm, const VmpBranchInfo &branch,
                                     bool forced_value, const llvm::DominatorTree &dt, unsigned &changed)
{
    if (!arm)
        return;
    unsigned bit = (branch.selector_inverts_flag ? !forced_value : forced_value) ? 1u : 0u;
    llvm::SmallVector<llvm::BinaryOperator *, 8> targets;
    for (auto &bb : fn)
    {
        if (!dt.dominates(arm, &bb))
            continue;
        for (auto &inst : bb)
        {
            auto *lshr = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
            if (!lshr || lshr->getOpcode() != llvm::Instruction::LShr || !lshr->isExact() ||
                !lshr->getType()->isIntegerTy(32))
                continue;
            auto *sh = llvm::dyn_cast<llvm::ConstantInt>(lshr->getOperand(1));
            if (sh && sh->getZExtValue() == branch.flag_shift)
                targets.push_back(lshr);
        }
    }
    for (auto *lshr : targets)
    {
        lshr->replaceAllUsesWith(llvm::ConstantInt::get(lshr->getType(), bit));
        ++changed;
    }
    if (!targets.empty())
        std::cout << "[*] Forced branch arm '" << arm->getName().str() << "' flag bit " << branch.flag_shift
                  << " to " << bit << " (" << targets.size() << " site(s))\n";
}

static unsigned force_arm_branch_flags(llvm::Function &fn, const VmBlock &root)
{
    std::vector<VmpBranchInfo> branches;
    collect_branch_infos(root, branches);
    if (branches.empty())
        return 0;

    llvm::DominatorTree dt(fn);
    unsigned            changed = 0;

    llvm::BasicBlock *taken_bb = nullptr;
    llvm::BasicBlock *fall_bb = nullptr;
    for (auto &bb : fn)
    {
        // TODO: fixme same as above - hardcoded names
        if (bb.getName() == "branch_taken" && !taken_bb)
            taken_bb = &bb;
        else if (bb.getName() == "branch_fall" && !fall_bb)
            fall_bb = &bb;
    }

    for (const auto &branch : branches)
    {
        force_branch_flag_in_arm(fn, taken_bb, branch, true, dt, changed);
        force_branch_flag_in_arm(fn, fall_bb, branch, false, dt, changed);
    }
    return changed;
}

static llvm::FunctionPassManager build_pre_concrete_cleanup_pipeline()
{
    llvm::FunctionPassManager cleanup;
    cleanup.addPass(llvm::PromotePass());
    cleanup.addPass(llvm::GVNPass(llvm::GVNOptions().setPRE(false).setLoadPRE(false)));
    cleanup.addPass(llvm::InstCombinePass());
    cleanup.addPass(llvm::SCCPPass());
    cleanup.addPass(llvm::DSEPass());
    cleanup.addPass(llvm::ADCEPass());
    return cleanup;
}

static llvm::FunctionPassManager build_post_concrete_cleanup_pipeline(bool include_sroa)
{
    llvm::FunctionPassManager cleanup;
    if (include_sroa)
        cleanup.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
    cleanup.addPass(llvm::PromotePass());
    cleanup.addPass(llvm::GVNPass(llvm::GVNOptions().setPRE(false).setLoadPRE(false)));
    cleanup.addPass(llvm::InstCombinePass());
    cleanup.addPass(llvm::SCCPPass());
    cleanup.addPass(llvm::CorrelatedValuePropagationPass());
    cleanup.addPass(llvm::JumpThreadingPass());
    cleanup.addPass(llvm::InstCombinePass());
    cleanup.addPass(llvm::DSEPass());
    cleanup.addPass(llvm::ADCEPass());
    cleanup.addPass(llvm::SimplifyCFGPass());
    return cleanup;
}

static void run_always_inline(llvm::Module &module)
{
    llvm::ModulePassManager mpm;
    mpm.addPass(llvm::AlwaysInlinerPass());
    helpers::run_module_pipeline(module, std::move(mpm));
}

static void run_pre_concrete_cleanup(llvm::Module &module)
{
    helpers::run_function_pipeline_on_module(module, build_pre_concrete_cleanup_pipeline());
}

static void run_post_concrete_cleanup(llvm::Module &module, bool include_sroa)
{
    helpers::run_function_pipeline_on_module(module, build_post_concrete_cleanup_pipeline(include_sroa));
}

static void run_sroa(llvm::Module &module)
{
    llvm::FunctionPassManager fpm;
    fpm.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
    helpers::run_function_pipeline_on_module(module, std::move(fpm));
}

static void run_freeze_cleanup(llvm::Module &module)
{
    llvm::FunctionPassManager cleanup;
    cleanup.addPass(llvm::ADCEPass());
    cleanup.addPass(llvm::SimplifyCFGPass());
    helpers::run_function_pipeline_on_module(module, std::move(cleanup));
}

static llvm::FunctionPassManager build_handler_local_cleanup_pipeline()
{
    llvm::FunctionPassManager cleanup;
    cleanup.addPass(llvm::PromotePass());
    cleanup.addPass(llvm::EarlyCSEPass());
    cleanup.addPass(llvm::InstCombinePass());
    cleanup.addPass(llvm::SCCPPass());
    cleanup.addPass(llvm::ADCEPass());
    cleanup.addPass(llvm::SimplifyCFGPass());
    return cleanup;
}

static void run_handler_local_cleanup(llvm::Function &fn)
{
    helpers::run_function_pipeline(fn, build_handler_local_cleanup_pipeline());
}

//==---------------------------------------------------------------------------==//
// LiftingContext
//==---------------------------------------------------------------------------==//

using ArmInject = std::optional<std::vector<StateInjectPatch>>;
using ArmForce  = std::optional<std::pair<unsigned, bool>>;

constexpr uint32_t kStateBytes           = 4096;
constexpr uint32_t kVmpStackWindowBytes  = 512;
constexpr uint32_t kVmpStackWindowRadius = kVmpStackWindowBytes / 2;
constexpr uint32_t kESIOffset            = 2280;
constexpr uint32_t kSnapshotStateBegin   = 2048;
constexpr uint32_t kSnapshotStateEnd     = 2560;

struct DevirtEmit
{
    llvm::Module    &target;
    llvm::Function  *devirt;
    llvm::Value     *state;
    llvm::Value     *ret_ptr;
    llvm::ArrayType *stateTy;
    llvm::StringRef  return_reg_name;
    bool             capture_snapshot;
};

struct LiftingContext
{
    llvm::LLVMContext                &ctx;
    const Memory                     &original_memory;
    remill::Arch::ArchPtr             arch;
    std::unique_ptr<llvm::Module>     sem_module;
    Memory                            patched_memory;
    TraceManager                      manager;
    remill::TraceLifter               lifter;
    VirtualMemoryLayout               vm;
    VmpTrace                          trace;
    unsigned                          param_count;
    bool                              save_intermediate;
    uint32_t                          host_return_address;
    std::vector<HostRegisterBinding>  host_reg_bindings;

    // Discovery state
    std::unordered_set<uint64_t>      global_seen;
    unsigned                          step_counter = 0;
    DiscoveryEngine                   discovery;
    static constexpr unsigned         kMaxHandlers = 1024;

    LiftingContext(
        llvm::LLVMContext                   &ctx,
        const Memory                        &original_memory,
        remill::Arch::ArchPtr                arch,
        std::unique_ptr<llvm::Module>        sem_module,
        Memory                               patched_memory,
        VirtualMemoryLayout                  vm,
        VmpTrace                             trace,
        unsigned                             param_count,
        bool                                 save_intermediate,
        uint32_t                             host_return_address,
        std::vector<HostRegisterBinding>     host_reg_bindings
    )
        : ctx(ctx)
        , original_memory(original_memory)
        , arch(std::move(arch))
        , sem_module(std::move(sem_module))
        , patched_memory(std::move(patched_memory))
        , manager(this->patched_memory, this->sem_module.get(), this->arch.get())
        , lifter(this->arch.get(), manager)
        , vm(std::move(vm))
        , trace(trace)
        , param_count(param_count)
        , save_intermediate(save_intermediate)
        , host_return_address(host_return_address)
        , host_reg_bindings(std::move(host_reg_bindings))
        , discovery(*this->sem_module, this->vm.initializer, this->original_memory, this->save_intermediate, this->step_counter)
    {
        manager.lift_set.insert(trace.vmenter);
    }

    std::unique_ptr<llvm::Module> execute(llvm::ArrayRef<uint64_t> replay_handlers);

    void            prepare_lifted_handler(uint64_t handler_addr);
    void            lift_handler(uint64_t addr);
    bool            discover_block(VmBlock &block, uint64_t start_addr,
                                   std::vector<uint64_t> prefix_path,
                                   ArmInject arm_inject, ArmForce arm_force);
    llvm::Function *build_devirt(llvm::Module &target, const VmBlock &root,
                                 llvm::StringRef function_name    = "devirt",
                                 llvm::StringRef return_reg_name  = "EAX",
                                 uint32_t        initial_eflags   = 0x202,
                                 const PrefixSnapshot *initial_snapshot = nullptr,
                                 bool            capture_snapshot = false);
    void setup_remill_intrinsics(llvm::Module &module) const;
    void build_remill_read(llvm::Module &module, const char *name, llvm::Type *valTy, llvm::Align align) const;
    void build_remill_write(llvm::Module &module, const char *name, llvm::Type *valTy, llvm::Align align) const;
    void store_known_or_unknown_byte(llvm::IRBuilder<> &b, llvm::GlobalVariable *unknown_byte_g,
                                     llvm::Value *ptr, const std::optional<uint8_t> &byte) const;
    void seed_window(llvm::IRBuilder<> &b, llvm::GlobalVariable *unknown_byte_g, llvm::Value *dst,
                     uint32_t dst_base, const std::vector<std::optional<uint8_t>> &bytes,
                     uint32_t begin, uint32_t end) const;
    void store_reg_value(llvm::IRBuilder<> &b, llvm::Value *state, const char *name, llvm::Value *value) const;
    void store_reg(llvm::IRBuilder<> &b, llvm::Value *state, const char *name, uint64_t value) const;
    void store_aflag_bit(llvm::IRBuilder<> &b, llvm::Value *state, uint32_t value,
                         uint32_t bit, uint32_t byte_off, const char *flag_name) const;
    void store_flags(llvm::IRBuilder<> &b, llvm::Value *state, uint32_t value) const;
    void apply_state_patches(llvm::IRBuilder<> &bldr, llvm::Value *state,
                             llvm::ArrayRef<StateInjectPatch> patches) const;
    llvm::Value *emit_handler_call(llvm::IRBuilder<> &bldr, llvm::Module &target, llvm::Value *state,
                                   uint64_t addr, llvm::Value *mem);
    void emit_forced_arm(const DevirtEmit &E, const VmBlock &child, llvm::BasicBlock *arm_bb, bool forced_value,
                         uint64_t branch_handler_addr, const VmpBranchInfo &branch_info, llvm::Value *mem);
    void emit_block(const DevirtEmit &E, const VmBlock &block, llvm::BasicBlock *cur_bb, llvm::Value *mem,
                    const std::vector<StateInjectPatch> *arm_patches);
    void neutralize_external_lifted_calls(llvm::Module &module) const;
    void stub_and_inline_flag_compare_helpers(llvm::Module &module) const;

    static ArmInject                    flag_inject_for_branch(const VmpBranchInfo &branch, bool forced_selector);
    static std::vector<StateInjectPatch> make_flag_patches(const VmpBranchInfo &branch, bool forced_selector);
    static unsigned                     count_block_handlers(const VmBlock &block);
};

void LiftingContext::prepare_lifted_handler(uint64_t handler_addr)
{
    auto it = manager.traces.find(handler_addr);
    if (it == manager.traces.end())
        return;
    auto *fn = it->second;
    if (!fn || fn->isDeclaration())
        return;
    if (fn->arg_size() >= 1)
        fn->getArg(0)->addAttr(llvm::Attribute::NoAlias);
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    run_handler_local_cleanup(*fn);
}

void LiftingContext::lift_handler(uint64_t addr)
{
    manager.lift_set.insert(addr);
    if (!manager.traces.count(addr))
        lifter.Lift(addr);
    prepare_lifted_handler(addr);
}

ArmInject LiftingContext::flag_inject_for_branch(const VmpBranchInfo &branch, bool forced_selector)
{
    // Map flag_shift to aflag byte offset in the state alloca
    static const uint8_t kFlagBitToByte[12] = {1,0,3,0,5,0,7,9,0,0,11,13};
    unsigned fs = branch.flag_shift;
    if (fs >= 12 || kFlagBitToByte[fs] == 0)
        return std::nullopt;

    bool raw_flag_value = branch.selector_inverts_flag ? !forced_selector : forced_selector;
    const uint8_t bit = static_cast<uint8_t>(1u << (fs & 7u));
    std::vector<StateInjectPatch> patches;

    // Remill arithmetic flag byte view
    patches.push_back(StateInjectPatch{
        kAFlagOffset + kFlagBitToByte[fs], 0xff, static_cast<uint8_t>(raw_flag_value ? 1u : 0u), true});

    // Packed RFLAGS view. Some lifted branch handlers read the packed EFLAGS byte directly, so preserve unrelated flag bits and force only the requested condition bit.
    patches.push_back(StateInjectPatch{
        kRFlagOffset + (fs / 8u), static_cast<uint8_t>(~bit), static_cast<uint8_t>(raw_flag_value ? bit : 0u), false});
    return ArmInject{std::move(patches)};
}

std::vector<StateInjectPatch> LiftingContext::make_flag_patches(const VmpBranchInfo &branch, bool forced_selector)
{
    static const uint8_t kFlagBitToByte[12] = {1,0,3,0,5,0,7,9,0,0,11,13};
    std::vector<StateInjectPatch> patches;
    unsigned fs = branch.flag_shift;
    if (fs >= 12 || kFlagBitToByte[fs] == 0)
        return patches;
    bool raw_flag_value = branch.selector_inverts_flag ? !forced_selector : forced_selector;
    const uint8_t bit = static_cast<uint8_t>(1u << (fs & 7u));
    patches.push_back(StateInjectPatch{
        kAFlagOffset + kFlagBitToByte[fs], 0xff, static_cast<uint8_t>(raw_flag_value ? 1u : 0u), true});
    patches.push_back(StateInjectPatch{
        kRFlagOffset + (fs / 8u), static_cast<uint8_t>(~bit), static_cast<uint8_t>(raw_flag_value ? bit : 0u), false});
    return patches;
}

unsigned LiftingContext::count_block_handlers(const VmBlock &block)
{
    auto total = static_cast<unsigned>(block.handlers.size());
    if (block.next)   total += count_block_handlers(*block.next);
    if (block.taken)  total += count_block_handlers(*block.taken);
    if (block.fall)   total += count_block_handlers(*block.fall);
    return total;
}

static void prepare_remill_intrinsic_fn(llvm::Function *fn, bool read_only)
{
    fn->removeFnAttr(llvm::Attribute::NoInline);
    fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    fn->removeFnAttr(llvm::Attribute::NoDuplicate);
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::WillReturn);
    fn->addFnAttr(llvm::Attribute::NoSync);
    if (read_only)
        fn->setMemoryEffects(llvm::MemoryEffects::readOnly());
}

void LiftingContext::build_remill_read(llvm::Module &module, const char *name, llvm::Type *valTy, llvm::Align align) const
{
    auto *i8Ty = llvm::Type::getInt8Ty(ctx);
    auto *fn = module.getFunction(name);
    if (!fn || !fn->isDeclaration())
        return;
    prepare_remill_intrinsic_fn(fn, /*read_only=*/true);
    auto *addr = fn->getArg(1);

    auto *mem = fn->getArg(0);
    auto *entryBB = llvm::BasicBlock::Create(ctx, "entry", fn);

    llvm::IRBuilder<> b(entryBB);
    auto             *off = b.CreateSub(addr, vm.base, "vmem_off");
    auto             *ptr = b.CreateGEP(i8Ty, mem, off, "vmem_ptr");
    b.CreateRet(b.CreateAlignedLoad(valTy, ptr, align));
}

void LiftingContext::build_remill_write(llvm::Module &module, const char *name, llvm::Type *valTy, llvm::Align align) const
{
    auto *i8Ty = llvm::Type::getInt8Ty(ctx);
    auto *fn = module.getFunction(name);
    if (!fn || !fn->isDeclaration())
        return;
    prepare_remill_intrinsic_fn(fn, /*read_only=*/false);
    fn->setMemoryEffects(
        llvm::MemoryEffects(llvm::MemoryEffects::Location::InaccessibleMem, llvm::ModRefInfo::ModRef)
    );

    auto *mem  = fn->getArg(0);
    auto *addr = fn->getArg(1);
    auto *val  = fn->getArg(2);

    auto *entryBB = llvm::BasicBlock::Create(ctx, "entry", fn);

    llvm::IRBuilder<> b(entryBB);
    auto             *off = b.CreateSub(addr, vm.base, "vmem_off");
    auto             *ptr = b.CreateGEP(i8Ty, mem, off, "vmem_ptr");
    b.CreateAlignedStore(val, ptr, align);
    b.CreateRet(mem);
}

void LiftingContext::setup_remill_intrinsics(llvm::Module &module) const
{
    auto *i8Ty  = llvm::Type::getInt8Ty(ctx);
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);

    build_remill_read(module, "__remill_read_memory_8",  i8Ty,                          llvm::Align(1));
    build_remill_read(module, "__remill_read_memory_16", llvm::Type::getInt16Ty(ctx),   llvm::Align(1));
    build_remill_read(module, "__remill_read_memory_32", i32Ty,                         llvm::Align(1));

    build_remill_write(module, "__remill_write_memory_32", i32Ty,                       llvm::Align(1));
    build_remill_write(module, "__remill_write_memory_16", llvm::Type::getInt16Ty(ctx), llvm::Align(1));
    build_remill_write(module, "__remill_write_memory_8",  i8Ty,                        llvm::Align(1));

    {
        auto *fn = module.getFunction("__remill_undefined_8");
        if (fn && fn->isDeclaration())
        {
            // stub with ret 0
            fn->removeFnAttr(llvm::Attribute::NoInline);
            fn->removeFnAttr(llvm::Attribute::OptimizeNone);
            fn->removeFnAttr(llvm::Attribute::NoDuplicate);
            fn->addFnAttr(llvm::Attribute::AlwaysInline);
            fn->addFnAttr(llvm::Attribute::WillReturn);
            fn->addFnAttr(llvm::Attribute::NoSync);
            fn->setMemoryEffects(llvm::MemoryEffects::none());
            auto             *bb = llvm::BasicBlock::Create(ctx, "entry", fn);
            llvm::IRBuilder<> b(bb);
            b.CreateRet(llvm::ConstantInt::get(i8Ty, 0));
        }
    }

    {
        auto *fn = module.getFunction("__remill_jump");
        if (fn && fn->isDeclaration())
        {
            fn->removeFnAttr(llvm::Attribute::NoInline);
            fn->removeFnAttr(llvm::Attribute::OptimizeNone);
            fn->addFnAttr(llvm::Attribute::AlwaysInline);
            fn->addFnAttr(llvm::Attribute::WillReturn);
            fn->addFnAttr(llvm::Attribute::NoSync);
            fn->setMemoryEffects(llvm::MemoryEffects::none());
            auto             *bb = llvm::BasicBlock::Create(ctx, "entry", fn);
            llvm::IRBuilder<> b(bb);
            b.CreateRet(fn->getArg(2));
        }
    }
}

void LiftingContext::neutralize_external_lifted_calls(llvm::Module &module) const
{
    for (auto &fn : module)
    {
        if (!fn.isDeclaration() || !fn.getName().starts_with("sub_"))
            continue;
        if (!fn.getReturnType()->isPointerTy() || fn.arg_size() < 3 || !fn.getArg(2)->getType()->isPointerTy())
            continue;

        fn.removeFnAttr(llvm::Attribute::NoInline);
        fn.removeFnAttr(llvm::Attribute::OptimizeNone);
        fn.addFnAttr(llvm::Attribute::AlwaysInline);
        fn.addFnAttr(llvm::Attribute::WillReturn);
        fn.addFnAttr(llvm::Attribute::NoSync);
        fn.setMemoryEffects(llvm::MemoryEffects::none());

        auto             *bb = llvm::BasicBlock::Create(ctx, "entry", &fn);
        llvm::IRBuilder<> b(bb);
        b.CreateRet(fn.getArg(2));
    }
}

void LiftingContext::stub_and_inline_flag_compare_helpers(llvm::Module &module) const
{
    for (auto &fn : module)
    {
        // these do not matter and we can stub them
        if (!fn.getName().starts_with("__remill_flag_computation_") && !fn.getName().starts_with("__remill_compare_"))
            continue;
        fn.addFnAttr(llvm::Attribute::AlwaysInline);
        if (fn.isDeclaration())
        {
            fn.setLinkage(llvm::GlobalValue::InternalLinkage);
            auto             *bb = llvm::BasicBlock::Create(ctx, "entry", &fn);
            llvm::IRBuilder<> b(bb);
            b.CreateRet(fn.getArg(0));
        }
        llvm::SmallVector<llvm::CallBase *, 16> calls;
        for (auto *user : fn.users())
            if (auto *cb = llvm::dyn_cast<llvm::CallBase>(user))
                calls.push_back(cb);
        for (auto *cb : calls)
        {
            llvm::InlineFunctionInfo ifi;
            llvm::InlineFunction(*cb, ifi);
        }
    }
}

void LiftingContext::store_known_or_unknown_byte(llvm::IRBuilder<> &b, llvm::GlobalVariable *unknown_byte_g,
                                                 llvm::Value *ptr, const std::optional<uint8_t> &byte) const
{
    auto *i8Ty = llvm::Type::getInt8Ty(ctx);
    if (byte)
        b.CreateStore(llvm::ConstantInt::get(i8Ty, *byte), ptr, false);
    else
        b.CreateStore(b.CreateLoad(i8Ty, unknown_byte_g, "snapshot_unknown_byte"), ptr, false);
}

// Seed a snapshot byte window
void LiftingContext::seed_window(llvm::IRBuilder<> &b, llvm::GlobalVariable *unknown_byte_g, llvm::Value *dst,
                                 uint32_t dst_base, const std::vector<std::optional<uint8_t>> &bytes,
                                 uint32_t begin, uint32_t end) const
{
    auto *i8Ty  = llvm::Type::getInt8Ty(ctx);
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);
    auto i = begin;
    for (; i + 4 <= end; i += 4)
    {
        if (bytes[i] && bytes[i + 1] && bytes[i + 2] && bytes[i + 3])
        {
            uint32_t w = static_cast<uint32_t>(*bytes[i]) | (static_cast<uint32_t>(*bytes[i + 1]) << 8) |
                         (static_cast<uint32_t>(*bytes[i + 2]) << 16) | (static_cast<uint32_t>(*bytes[i + 3]) << 24);
            auto *ptr = b.CreateConstGEP1_32(i8Ty, dst, dst_base + i, "snapshot_init_word_ptr");
            b.CreateStore(llvm::ConstantInt::get(i32Ty, w), ptr, false);
        }
        else
        {
            for (uint32_t k = 0; k < 4; ++k)
            {
                auto *ptr = b.CreateConstGEP1_32(i8Ty, dst, dst_base + i + k, "snapshot_init_byte_ptr");
                store_known_or_unknown_byte(b, unknown_byte_g, ptr, bytes[i + k]);
            }
        }
    }
    for (; i < end; ++i)
    {
        auto *ptr = b.CreateConstGEP1_32(i8Ty, dst, dst_base + i, "snapshot_init_byte_ptr");
        store_known_or_unknown_byte(b, unknown_byte_g, ptr, bytes[i]);
    }
}

void LiftingContext::store_reg_value(llvm::IRBuilder<> &b, llvm::Value *state, const char *name, llvm::Value *value) const
{
    auto *i8Ty = llvm::Type::getInt8Ty(ctx);
    const auto *reg = arch->RegisterByName(name);
    if (!reg)
    {
        std::cerr << "[!] store_reg_value: unknown register '" << name << "' — initialization skipped\n";
        return;
    }
    auto *ptr =
        b.CreateConstGEP1_32(i8Ty, state, static_cast<uint32_t>(reg->offset), std::string(name) + "_ptr");
    b.CreateStore(value, ptr);
}

void LiftingContext::store_reg(llvm::IRBuilder<> &b, llvm::Value *state, const char *name, uint64_t value) const
{
    if (const auto *reg = arch->RegisterByName(name))
        store_reg_value(b, state, name, llvm::ConstantInt::get(reg->type, value));
    else
        std::cerr << "[!] store_reg: unknown register '" << name << "' — initialization skipped\n";
}

void LiftingContext::store_aflag_bit(llvm::IRBuilder<> &b, llvm::Value *state, uint32_t value,
                                     uint32_t bit, uint32_t byte_off, const char *flag_name) const
{
    auto *i8Ty = llvm::Type::getInt8Ty(ctx);
    auto *ptr = b.CreateConstGEP1_32(i8Ty, state, kAFlagOffset + byte_off, flag_name);
    b.CreateStore(llvm::ConstantInt::get(i8Ty, (value >> bit) & 1u), ptr);
}

void LiftingContext::store_flags(llvm::IRBuilder<> &b, llvm::Value *state, uint32_t value) const
{
    auto *i8Ty = llvm::Type::getInt8Ty(ctx);
    // Initialize arithmetic flags (aflag) separately from the packed architectural flags "rflag"
    // Writing only to eflgas as a register is not enough for jcc instructions
    constexpr uint32_t kAFlagOffset = 16 + 2048;
    constexpr uint32_t kRFlagOffset = kAFlagOffset + 16;

    auto *rflag_ptr = b.CreateConstGEP1_32(i8Ty, state, kRFlagOffset, "rflag_ptr");
    b.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), value), rflag_ptr);

    store_aflag_bit(b, state, value, 0, 1, "aflag_cf_ptr");
    store_aflag_bit(b, state, value, 2, 3, "aflag_pf_ptr");
    store_aflag_bit(b, state, value, 4, 5, "aflag_af_ptr");
    store_aflag_bit(b, state, value, 6, 7, "aflag_zf_ptr");
    store_aflag_bit(b, state, value, 7, 9, "aflag_sf_ptr");
    store_aflag_bit(b, state, value, 10, 11, "aflag_df_ptr");
    store_aflag_bit(b, state, value, 11, 13, "aflag_of_ptr");
}

// helper for applying state patches to branches
void LiftingContext::apply_state_patches(llvm::IRBuilder<> &bldr, llvm::Value *state,
                                         llvm::ArrayRef<StateInjectPatch> patches) const
{
    auto *i8Ty = llvm::Type::getInt8Ty(ctx);
    for (const auto &patch : patches)
    {
        auto *ptr = bldr.CreateConstGEP1_32(i8Ty, state, patch.offset, "forced_flag_ptr");
        llvm::Value *val = llvm::ConstantInt::get(i8Ty, patch.set_bits);
        if (!patch.overwrite)
        {
            auto *old     = bldr.CreateLoad(i8Ty, ptr, "forced_flag_old");
            auto *cleared = bldr.CreateAnd(old, llvm::ConstantInt::get(i8Ty, patch.clear_mask), "forced_flag_clear");
            val = bldr.CreateOr(cleared, llvm::ConstantInt::get(i8Ty, patch.set_bits), "forced_flag_set");
        }
        bldr.CreateStore(val, ptr);
    }
}

// Helper to emit a call to a handler
llvm::Value *LiftingContext::emit_handler_call(llvm::IRBuilder<> &bldr, llvm::Module &target, llvm::Value *state,
                                               uint64_t addr, llvm::Value *mem)
{
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);
    auto  fn_name = manager.TraceName(addr);
    auto *handler = target.getFunction(fn_name);
    if (!handler)
    {
        std::cerr << "[!] handler " << fn_name << " not found in module\n";
        exit(1);
    }
    auto *pc = llvm::ConstantInt::get(i32Ty, static_cast<uint32_t>(addr));
    return bldr.CreateCall(handler, {state, pc, mem}, "mem");
}

static llvm::GlobalVariable *get_or_create_global(llvm::Module &target, llvm::StringRef name, llvm::Type *ty)
{
    if (auto *g = target.getGlobalVariable(name))
        return g;
    return new llvm::GlobalVariable(
        target, ty, false, llvm::GlobalValue::ExternalLinkage, nullptr, name);
}

static llvm::ConstantAsMetadata *md_i32(llvm::LLVMContext &ctx, uint32_t v)
{
    return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), v));
}

void LiftingContext::emit_forced_arm(const DevirtEmit &E, const VmBlock &child, llvm::BasicBlock *arm_bb,
                                     bool forced_value, uint64_t branch_handler_addr,
                                     const VmpBranchInfo &branch_info, llvm::Value *mem)
{
    llvm::IRBuilder<> arm_bldr(arm_bb);
    auto patches = make_flag_patches(branch_info, forced_value);
    apply_state_patches(arm_bldr, E.state, patches);

    auto *arm_mem = emit_handler_call(arm_bldr, E.target, E.state, branch_handler_addr, mem);

    emit_block(E, child, arm_bb, arm_mem, &patches);
}

// Recursive helper: emit all handlers in `block` into `cur_bb`, then emit the appropriate terminator (ret or condbr into child blocks)
// Pass the memory token explicitly
void LiftingContext::emit_block(const DevirtEmit &E, const VmBlock &block, llvm::BasicBlock *cur_bb,
                                llvm::Value *mem, const std::vector<StateInjectPatch> *arm_patches)
{
    auto *i8Ty  = llvm::Type::getInt8Ty(ctx);
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);
    llvm::IRBuilder<> bldr(cur_bb);

    const bool   split_before_last_handler  = block.is_branch() && !block.handlers.empty();
    const size_t linear_handler_count = split_before_last_handler ? block.handlers.size() - 1 : block.handlers.size();

    for (size_t idx = 0; idx < linear_handler_count; ++idx)
    {
        if (block.inject_before_idx && idx == *block.inject_before_idx)
            apply_state_patches(bldr, E.state, block.state_inject); // Inject forced state bytes before this handler
        if (arm_patches)
            apply_state_patches(bldr, E.state, *arm_patches);

        mem = emit_handler_call(bldr, E.target, E.state, block.handlers[idx], mem);
    }

    if (block.is_terminal())
    {
        auto *ret_val = bldr.CreateLoad(i32Ty, E.ret_ptr, E.return_reg_name.str() + "_out");
        if (E.capture_snapshot)
        {
            // Discovery's cached-snapshot optimization: at a prefix VMEXIT, put the concrete VM register window + data-stack window into
            // globals that run_step/extract_prefix_snapshot reads back, so the next discovery step can resume from concrete state
            auto *state_g = get_or_create_global(E.target, "__vmp_snapshot_state", E.stateTy);
            auto *stackTy = llvm::ArrayType::get(i8Ty, kVmpStackWindowBytes);
            auto *stack_g = get_or_create_global(E.target, "__vmp_snapshot_stack", stackTy);
            auto *base_g  = get_or_create_global(E.target, "__vmp_snapshot_stack_base", i32Ty);

            auto *esi_ptr   = bldr.CreateConstGEP1_32(i8Ty, E.state, kESIOffset, "snapshot_esi_ptr");
            auto *vsp       = bldr.CreateLoad(i32Ty, esi_ptr, "snapshot_vsp");
            auto *stack_base = bldr.CreateSub(vsp, llvm::ConstantInt::get(i32Ty, kVmpStackWindowRadius), "snapshot_stack_base");
            bldr.CreateStore(stack_base, base_g, false);

            // Capture the register/stack
            for (uint32_t w = kSnapshotStateBegin; w < kSnapshotStateEnd; w += 4)
            {
                auto *wptr = bldr.CreateConstGEP1_32(i8Ty, E.state, w, "snapshot_state_word_ptr");
                auto *word = bldr.CreateLoad(i32Ty, wptr, "snapshot_state_word");
                auto *dst  = bldr.CreateConstGEP2_32(E.stateTy, state_g, 0, w, "snapshot_state_dst");
                bldr.CreateStore(word, dst, false);
            }
            for (uint32_t i = 0; i < kVmpStackWindowBytes; i += 4)
            {
                auto *addr = bldr.CreateAdd(stack_base, llvm::ConstantInt::get(i32Ty, i), "snapshot_stack_addr");
                auto *off  = bldr.CreateSub(addr, vm.base, "snapshot_stack_off");
                auto *src  = bldr.CreateGEP(i8Ty, mem, off, "snapshot_stack_src");
                auto *word = bldr.CreateLoad(i32Ty, src, "snapshot_stack_word");
                auto *dst  = bldr.CreateConstGEP2_32(stackTy, stack_g, 0, i, "snapshot_stack_dst");
                bldr.CreateStore(word, dst, false);
            }
        }
        bldr.CreateRet(ret_val);
        return;
    }

    if (block.next)
    {
        emit_block(E, *block.next, cur_bb, mem, arm_patches);
        return;
    }

    if (block.is_branch())
    {
        const unsigned fs = block.branch_info.flag_shift;
        llvm::Value *flag_cond = nullptr;
        if (fs < 32)
        {
            auto *rflags_ptr = bldr.CreateConstGEP1_32(i8Ty, E.state, kRFlagOffset, "rflags_ptr");
            auto *rflags_i32 = bldr.CreateLoad(i32Ty, rflags_ptr, "rflags");
            auto *masked     = bldr.CreateAnd(rflags_i32, llvm::ConstantInt::get(i32Ty, 1u << fs), "branch_flag_masked");
            auto *raw_cond   = bldr.CreateICmpNE(masked, llvm::ConstantInt::get(i32Ty, 0), "branch_raw_flag");
            flag_cond = block.branch_info.selector_inverts_flag ? bldr.CreateNot(raw_cond, "branch_selector") : raw_cond;
        }
        else
        {
            bldr.CreateUnreachable();
            return;
        }

        if (!split_before_last_handler)
        {
            bldr.CreateUnreachable();
            return;
        }
        const uint64_t branch_handler_addr = block.handlers.back();

        auto *taken_bb = llvm::BasicBlock::Create(ctx, "branch_taken", E.devirt); // TODO: hardcoded fixme same as above
        auto *fall_bb  = llvm::BasicBlock::Create(ctx, "branch_fall",  E.devirt);
        auto *cond_br  = bldr.CreateCondBr(flag_cond, taken_bb, fall_bb);
        // Tag the branch so the post-concrete pass can re-key it on the branchless VSP selector
        {
            cond_br->setMetadata(
                "vmp_branch",
                llvm::MDNode::get(ctx, {md_i32(ctx, block.branch_info.vsp_taken), md_i32(ctx, block.branch_info.vsp_fall)})
            );
        }

        emit_forced_arm(E, *block.taken, taken_bb, true,  branch_handler_addr, block.branch_info, mem);
        emit_forced_arm(E, *block.fall,  fall_bb,  false, branch_handler_addr, block.branch_info, mem);
        return;
    }

    // No terminator, emit unreachable
    bldr.CreateUnreachable();
}

llvm::Function *LiftingContext::build_devirt(
    llvm::Module &target, const VmBlock &root,
    llvm::StringRef function_name, llvm::StringRef return_reg_name,
    uint32_t initial_eflags, const PrefixSnapshot *initial_snapshot,
    bool capture_snapshot
)
{
    // Setup function signature
    auto                     *i8Ty  = llvm::Type::getInt8Ty(ctx);
    auto                     *i32Ty = llvm::Type::getInt32Ty(ctx);
    std::vector<llvm::Type *> param_types(param_count, i32Ty);
    auto                     *fn_ty = llvm::FunctionType::get(i32Ty, param_types, false);
    auto *devirt = llvm::Function::Create(fn_ty, llvm::GlobalValue::ExternalLinkage, function_name, &target);
    for (unsigned i = 0; i < param_count; ++i) devirt->getArg(i)->setName("arg" + std::to_string(i));

    // Create first basic block
    auto             *bb = llvm::BasicBlock::Create(ctx, "entry", devirt);
    llvm::IRBuilder<> b(bb);

    // Create remill state
    auto              *stateTy = llvm::ArrayType::get(i8Ty, kStateBytes);
    auto              *state   = b.CreateAlloca(stateTy, nullptr, "state");
    state->setAlignment(llvm::Align(64));

    b.CreateStore(llvm::ConstantAggregateZero::get(stateTy), state);

    auto *unknown_byte_g = new llvm::GlobalVariable(
        target, i8Ty, false, llvm::GlobalValue::ExternalLinkage, nullptr, "__vmp_snapshot_unknown_byte");

    // Setup initial state snapshot if supplied (for handler discovery)
    if (initial_snapshot && initial_snapshot->state.size() == kStateBytes)
        seed_window(b, unknown_byte_g, state, 0, initial_snapshot->state, kSnapshotStateBegin, kSnapshotStateEnd);

    // Actual run - setup initial state to concretize as much as possible
    if (!initial_snapshot)
    {
        // while args are symbolic, we can still model a proper host function call context
        store_reg(b, state, "EAX", 0); // these should not matter much anyway so set to 0, if it is an arg its overwritten below
        store_reg(b, state, "EBX", 0);
        store_reg(b, state, "ECX", 0);
        store_reg(b, state, "EDX", 0);
        store_reg(b, state, "ESI", 1);
        store_reg(b, state, "EDI", 0);
        store_reg(b, state, "EBP", kSyntheticEsp + 0xC0);
        store_reg(b, state, "ESP", kSyntheticEsp);
        store_reg(b, state, "EIP", root.handlers.empty() ? 0 : root.handlers.front());
        store_flags(b, state, initial_eflags);
        // flat memory model
        store_reg(b, state, "SSBASE", 0);
        store_reg(b, state, "DSBASE", 0);
        store_reg(b, state, "ESBASE", 0);
        store_reg(b, state, "CSBASE", 0);
        store_reg(b, state, "FSBASE", 0);
        store_reg(b, state, "GSBASE", 0);

        // set up args
        for (const auto &binding : host_reg_bindings)
        {
            if (binding.arg_index < param_count)
                store_reg_value(b, state, binding.reg_name.c_str(), devirt->getArg(binding.arg_index));
        }
    }

    // Create virtual memory alloca
    auto *vmem = b.CreateAlloca(vm.array_ty, nullptr, "virtual_memory");
    vmem->setAlignment(llvm::Align(16));
    b.CreateStore(vm.initializer, vmem);

    // Setup initial stack snapshot if supplied (for handler discovery)
    if (initial_snapshot && initial_snapshot->stack.size() == kVmpStackWindowBytes)
        seed_window(b, unknown_byte_g, vmem, initial_snapshot->stack_base, initial_snapshot->stack, 0, kVmpStackWindowBytes);
    else
    {
        auto *ret_addr_ptr = b.CreateConstGEP1_32(i8Ty, vmem, kSyntheticEsp, "return_address_ptr");
        b.CreateStore(llvm::ConstantInt::get(i32Ty, host_return_address), ret_addr_ptr, false);

        // Make the requested original function parameters symbolic
        for (unsigned i = 0; i < param_count; ++i)
        {
            // At VMENTER the first argument lives at ESP+4
            auto *arg_ptr =
                b.CreateConstGEP1_32(i8Ty, vmem, kSyntheticParamBase + (i * 4), "arg" + std::to_string(i) + "_ptr");
            b.CreateStore(devirt->getArg(i), arg_ptr, false);
        }
    }

    // Resolve return register pointer once (used in every terminal block)
    const auto *ret_reg = arch->RegisterByName(return_reg_name.str());
    if (!ret_reg)
    {
        std::cerr << "[!] return register " << return_reg_name.str() << " not found\n";
        exit(1);
    }
    auto *ret_ptr = b.CreateConstGEP1_32(i8Ty, state, (uint32_t)ret_reg->offset, return_reg_name.str() + "_out_ptr");

    DevirtEmit E{target, devirt, state, ret_ptr, stateTy, return_reg_name, capture_snapshot};
    emit_block(E, root, bb, vmem, nullptr);

    return devirt;
}

bool LiftingContext::discover_block(
    VmBlock &block, uint64_t start_addr,
    std::vector<uint64_t> prefix_path, ArmInject arm_inject, ArmForce arm_force
)
{
    uint64_t current_addr = start_addr;
    std::optional<PrefixSnapshot> cached_snapshot;

    while (step_counter < kMaxHandlers)
    {
        // Lift if not already lifted
        if (!global_seen.count(current_addr))
        {
            lift_handler(current_addr);
            global_seen.insert(current_addr);
        }
        block.handlers.push_back(current_addr);
        std::cout << "[*] Lifting handler 0x" << std::hex << current_addr << std::dec << "\n";

        // Heuristic check for vmexit, but we lift the native resume stub after VMEXIT as well
        const bool is_vmexit_restore = looks_like_vmexit_restore_handler(patched_memory, current_addr);
        if (is_vmexit_restore)
            std::cout << "[*] Potential VMEXIT at 0x" << std::hex << current_addr << std::dec << "\n";

        unsigned discovery_step       = step_counter;
        bool     used_cached_snapshot = cached_snapshot && discovery_step >= 4; // fully build first 4 steps, need a full VMENTER for the snapshot

        // Build prefix function: builds the current handler either from a snapshot or as a full prefix from the entry to recover next handler (EIP)
        auto build_prefix = [&](llvm::StringRef fn_name) -> llvm::Function *
        {
            VmBlock               probe;
            const PrefixSnapshot *seed = nullptr;
            if (used_cached_snapshot)
            {
                // Fast path: run just this handler, seeded from the cached snapshot.
                probe.handlers.push_back(current_addr);
                seed = &*cached_snapshot;
            }
            else
            {
                // Slow path: re-lift the full prefix (root -> ... -> this handler).
                probe.handlers = prefix_path;
                probe.handlers.insert(
                    probe.handlers.end(), block.handlers.begin(), block.handlers.end());

                // Force the branch flag concrete just before the branch handler.
                if (arm_inject && !prefix_path.empty())
                {
                    probe.inject_before_idx = prefix_path.size() - 1;
                    probe.state_inject      = *arm_inject;
                }
            }
            probe.at_vmexit = true;  // terminal block: emit `ret EIP` (+ snapshot capture)

            // Return EIP (not EAX) so the probe yields the next handler address.
            return build_devirt(
                *sem_module, probe, fn_name, /*return_reg_name=*/"EIP",
                /*initial_eflags=*/0x202, /*initial_snapshot=*/seed,
                /*capture_snapshot=*/true);
        };

        PrefixResult result;
        while (true)
        {
            // we want to fully concretize the prefix, so that the next iteration can benefit from the whole state snapshot
            result = discovery.run_step(build_prefix, arm_force, /*full_concretize=*/true);

            bool retry_full_prefix = false; // fallback to full prefix if the result is bogus or unknown
            if (used_cached_snapshot)
            {
                if (result.kind == PrefixResult::Kind::Unknown)
                    retry_full_prefix = true;
                else if (result.kind == PrefixResult::Kind::Concrete)
                {
                    if (result.next == static_cast<uint32_t>(current_addr) ||
                        result.next == 0 ||
                        result.next == host_return_address ||
                        !original_memory.count(result.next))
                    {
                        retry_full_prefix = true;
                    }
                }
            }
            if (!retry_full_prefix)
                break;

            std::cout << "[*] Snapshot discovery failed after 0x" << std::hex << current_addr
                      << "; retrying full prefix" << std::dec << "\n";
            used_cached_snapshot = false;
            cached_snapshot.reset();
        }

        if (result.kind == PrefixResult::Kind::Concrete)
        {
            uint32_t next = result.next;
            if (result.snapshot)
                cached_snapshot = std::move(result.snapshot);
            std::cout << "[*] Calculated next handler: 0x" << std::hex << next << std::dec << "\n";

            if (is_vmexit_restore)
            {
                // Lift the result-finalizer stub the restore handler returns int and then end the block at the VM exit
                if (next != 0 && original_memory.count(next))
                {
                    std::cout << "[*] VMEXIT: lifting result-finalizer stub 0x" << std::hex << next << std::dec
                              << "\n";
                    lift_handler(next);
                    block.handlers.push_back(next);
                }
                block.at_vmexit = true;
                return true;
            }

            if (next == 0 || next == host_return_address)
            {
                std::cout << "[!] Handler discovery stopped at terminal address 0x" << std::hex << next
                          << std::dec << "\n";
                return false;
            }
            if (!original_memory.count(next))
            {
                std::cout << "[!] Handler discovery produced non-image address 0x" << std::hex << next
                          << std::dec << "\n";
                return false;
            }
            current_addr = next;
            continue;
        }

        if (result.kind == PrefixResult::Kind::Branch)
        {
            block.branch_info = result.branch;
            block.taken       = std::make_unique<VmBlock>();
            block.fall        = std::make_unique<VmBlock>();

            // Pass the full path (prefix_path + this block) to each arm and compute state_offset and forced_byte for the aflag used by this branch
            std::vector<uint64_t> arm_prefix = prefix_path;
            arm_prefix.insert(arm_prefix.end(), block.handlers.begin(), block.handlers.end());

            ArmInject taken_inject = flag_inject_for_branch(result.branch, true);
            ArmInject fall_inject  = flag_inject_for_branch(result.branch, false);

            bool ok_taken = discover_block(
                *block.taken, result.branch.addr_taken, arm_prefix, taken_inject,
                ArmForce{{result.branch.flag_shift, true}});
            bool ok_fall  = discover_block(
                *block.fall,  result.branch.addr_fall,  arm_prefix, fall_inject,
                ArmForce{{result.branch.flag_shift, false}});
            return ok_taken && ok_fall;
        }

        // if Kind::Unknown
        std::cout << "[!] Could not calculate next handler after 0x" << std::hex << current_addr
                  << std::dec << "\n";
        return false;
    }
    std::cout << "[!] Handler discovery hit limit\n";
    return false;
}

std::unique_ptr<llvm::Module> LiftingContext::execute(llvm::ArrayRef<uint64_t> replay_handlers)
{
    // Setup remill intrinsics and normalize functions
    setup_remill_intrinsics(*sem_module);
    neutralize_dispatch_intrinsics(*sem_module);
    neutralize_external_lifted_calls(*sem_module);
    stub_and_inline_flag_compare_helpers(*sem_module);

    // -----------------------------------------------------------------------
    // Build root VmBlock
    auto root_block = std::make_unique<VmBlock>();
    bool recovered_to_vmexit = false;

    // Replay mode - build a flat linear VmBlock from the replay list
    if (!replay_handlers.empty())
    {
        std::vector<uint64_t> replay_trace;
        if (replay_handlers.front() != trace.vmenter)
            replay_trace.push_back(trace.vmenter);
        replay_trace.insert(replay_trace.end(), replay_handlers.begin(), replay_handlers.end());

        std::cout << "[*] Replaying " << std::dec << replay_trace.size() << " trace entries; skipping discovery\n";
        for (uint64_t addr : replay_trace)
            lift_handler(addr);

        root_block->handlers  = replay_trace;
        root_block->at_vmexit = true;
        recovered_to_vmexit   = true;
    }
    else
    {
        // Discovery mode - lift handlers one by one and fork on branch
        std::cout << "[*] Starting handler discovery at 0x" << std::hex << trace.vmenter << std::dec << "\n";
        recovered_to_vmexit = discover_block(*root_block, trace.vmenter, {}, std::nullopt, std::nullopt);
    }

    if (!recovered_to_vmexit)
        std::cerr << "[!] Failed to recover a complete trace to VMEXIT; emitting partially devirtualized IR\n";

    std::cout << "[*] Incrementally recovered " << count_block_handlers(*root_block) << " trace entries\n";
    std::cout << "[*] Lifted " << manager.traces.size() << " handlers\n";

    // Mark handlers as always inline
    for (auto &[addr, fn] : manager.traces)
    {
        if (fn && !fn->isDeclaration() && fn->arg_size() >= 1)
        {
            fn->getArg(0)->addAttr(llvm::Attribute::NoAlias);
            fn->addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }

    remill::OptimizationGuide guide{};
    remill::OptimizeModule(arch, sem_module, manager.traces, guide);

    // -----------------------------------------------------------------------

    // Build output module
    auto out_module = std::make_unique<llvm::Module>("devirt", ctx);
    arch->PrepareModuleDataLayout(out_module.get());

    // Move handlers into output module
    for (auto &[addr, fn] : manager.traces)
        remill::MoveFunctionIntoModule(fn, out_module.get());

    // Setup remill intrinsics
    setup_remill_intrinsics(*out_module);

    // Cleanup dispatches and remaining handler calls
    neutralize_dispatch_intrinsics(*out_module);
    neutralize_external_lifted_calls(*out_module);

    // Build main function
    build_devirt(*out_module, *root_block);

    // Stub and inline flag/compare helpers
    stub_and_inline_flag_compare_helpers(*out_module);

    // Optimize individual handlers
    for (auto &[addr, fn] : manager.traces)
    {
        llvm::LoopAnalysisManager     lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager    cam;
        llvm::ModuleAnalysisManager   mam;
        llvm::PassBuilder             pb;
        pb.registerModuleAnalyses(mam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.registerCGSCCAnalyses(cam);
        pb.crossRegisterProxies(lam, fam, cam, mam);
        llvm::FunctionPassManager fpm0;
        
        // Phase 1: Initial Scalarization & Dead Code
        fpm0.addPass(llvm::SROAPass({}));
        fpm0.addPass(llvm::SCCPPass());
        fpm0.addPass(llvm::ADCEPass());

        // Phase 2: Simplification & Memory Promotion
        fpm0.addPass(llvm::SimplifyCFGPass());
        fpm0.addPass(llvm::EarlyCSEPass(true));
        fpm0.addPass(llvm::PromotePass()); // Convert allocas to SSA

        // Phase 3: Value Numbering & Elimination
        fpm0.addPass(llvm::InstCombinePass());
        fpm0.addPass(llvm::GVNPass(llvm::GVNOptions()));
        fpm0.addPass(llvm::SCCPPass());
        fpm0.addPass(llvm::ADCEPass());

        // Phase 4: Final Cleanup
        fpm0.addPass(llvm::InstCombinePass());
        fpm0.addPass(llvm::SROAPass({}));
        fpm0.addPass(llvm::ADCEPass());
        fpm0.addPass(llvm::DSEPass());
        fpm0.addPass(llvm::SimplifyCFGPass());
        fpm0.run(*fn, fam);
    }

    if (save_intermediate)
        dump_module_snapshot(*out_module, "out.before_cleanup.ll");

    // Inline all functions
    run_always_inline(*out_module);

    // concretize segments to 0 so they can be folded
    std::set<uint64_t> seg_offsets;
    for (const char *n : {"SSBASE", "DSBASE", "ESBASE", "CSBASE", "FSBASE", "GSBASE"})
        if (const auto *r = arch->RegisterByName(n))
            seg_offsets.insert(static_cast<uint64_t>(r->offset));

    if (auto *devirt_fn = out_module->getFunction("devirt"))
    {
        // Concretize the VM condition flag inside each branch arm
        force_arm_branch_flags(*devirt_fn, *root_block);


        unsigned flat = flatten_segment_base_loads(*devirt_fn, seg_offsets);
        if (flat)
            std::cout << "[*] Flattened " << flat << " segment-base load(s) to 0\n";
    }
    run_pre_concrete_cleanup(*out_module);

    // Mark handler functions internal so they can be pruned after inlining.
    for (auto &[addr, fn] : manager.traces)
    {
        if (fn && !fn->isDeclaration())
            fn->setLinkage(llvm::GlobalValue::InternalLinkage);
    }

    // Concretize the dispatch handler's VSP in each branch arm so the EIP-decrypt
    // chain (and every VM-stack access keyed off it) folds. The legacy
    // masked-select matcher (materialize_branch_vsp_constants) only fires when the
    // branchless selection survives as `add (and..),(and..)`; after full inlining
    // that arithmetic has usually collapsed into a VM-stack push/pop, so the
    // dominator-scoped decrypt-signature pass is the one that actually applies.
    if (auto *devirt_fn = out_module->getFunction("devirt"))
    {
        unsigned forced = materialize_branch_vsp_constants(*devirt_fn, *root_block);
        forced += force_arm_decrypt_vsp(*devirt_fn, *root_block);
        if (forced)
            run_post_concrete_cleanup(*out_module, /*include_sroa=*/false);
    }

    dump_module_snapshot(*out_module, "out.DBG_before_concrete.ll");
    if (save_intermediate)
        dump_module_snapshot(*out_module, "out.before_concrete.ll");

    // Iteratively optimize and propagate constants
    unsigned           total_folded = 0;
    constexpr unsigned kMaxConcreteIters = 300;
    auto *devirt_fn = out_module->getFunction("devirt");
    for (unsigned iter = 0; iter < kMaxConcreteIters; ++iter)
    {
        // GVN/PRE in the cleanup keeps re-materializing segment-base loads from
        // the state alloca; re-flatten them each iteration so register-file
        // addresses stay foldable.
        unsigned flat = flatten_segment_base_loads(*devirt_fn, seg_offsets);
        unsigned folded = propagate_concrete_alloca_constants(
            *devirt_fn, vm.initializer, kSyntheticStackBase,
            static_cast<uint64_t>(kSyntheticStackBase) + kSyntheticStackSize);
        total_folded += folded;
        if (!folded && !flat)
            break;

        run_post_concrete_cleanup(*out_module, /*include_sroa=*/false);
        if (iter < 6)
            dump_module_snapshot(*out_module, ("out.DBG_iter" + std::to_string(iter) + ".ll").c_str());
    }

    flatten_segment_base_loads(*devirt_fn, seg_offsets);
    // Expose the branchless VSP selector as pure SSA (SROA only, no folding), then
    // re-key the injected branch on it
    // Only then run the simplifying cleanup, which now folds the VM layer while
    // preserving the branch as select(arg-condition, ...)
    // FIXME: this is a bit brittle maybe...
    run_sroa(*out_module);
    if (rewrite_branch_conditions_on_selector(*devirt_fn))
        run_post_concrete_cleanup(*out_module, /*include_sroa=*/false);
    run_post_concrete_cleanup(*out_module, /*include_sroa=*/true);
    if (unwrap_return_freezes(*devirt_fn))
        run_freeze_cleanup(*out_module);

    std::cout << "[*] Concrete state/virtual_memory loads folded after scalar cleanup: " << std::dec << total_folded << "\n";
    dump_module_snapshot(*out_module, "out.DBG_after_concrete_prop.ll");
    if (save_intermediate)
        dump_module_snapshot(*out_module, "out.after_concrete_prop.ll");

    // Final optimization
    helpers::run_default_o3_pipeline(*out_module);
    if (save_intermediate)
        dump_module_snapshot(*out_module, "out.after_final_o3.ll");

    // Remove unused internal functions and undefined8 calls
    erase_unused_internal_functions(*out_module);
    erase_unused_undefined8_calls(*out_module);

    // Final readability-only passes: rename SSA regs, then basic blocks.
    llvm::ModulePassManager readable_names;
    readable_names.addPass(ReadableRegisterNamesPass());
    readable_names.addPass(ReadableBlockNamesPass());
    helpers::run_module_pipeline(*out_module, std::move(readable_names));

    // Return the module
    return out_module;
}

//==---------------------------------------------------------------------------==//

LiftResult VmpLifter::run(
    const Memory &memory, VmpTrace trace, std::optional<unsigned> param_count, bool save_intermediate,
    llvm::ArrayRef<uint64_t> replay_handlers
)
{
    auto  ctx_owned = std::make_unique<llvm::LLVMContext>();
    auto &ctx       = *ctx_owned;

    // Only supporting 32bit
    auto arch = remill::Arch::Get(ctx, "windows", "x86");
    if (!arch)
    {
        std::cerr << "Failed to create remill arch for x86/windows\n";
        return {};
    }

    std::unique_ptr<llvm::Module> sem_module(remill::LoadArchSemantics(arch.get()));

    // ---------------------------------------------------------------
    // Normalization 
    
    // Normalize VMENTRY wrapper calls so it doesn't fall through into junk code
    unsigned patched_vmentry_calls = 0;
    auto     call_patched_memory = patch_nonreturn_vmentry_calls(memory, trace.vmenter, &patched_vmentry_calls);
    uint32_t host_return_address = 0;
    unsigned resolved_param_count = param_count.value_or(0);
    std::vector<HostRegisterBinding> host_reg_bindings;
    if (auto call_site = find_unique_direct_call_site(memory, trace.vmenter))
    {
        host_return_address = call_site->return_address;
        if (!param_count)
        {
            resolved_param_count = count_argument_pushes(memory, *call_site);
            std::cout << "    argc inferred: " << std::dec << resolved_param_count << "\n";
        }
        host_reg_bindings = infer_argument_register_bindings(memory, *call_site, resolved_param_count);
    }
    std::cout << "    Synthetic host return address: 0x" << std::hex << host_return_address << std::dec << "\n";
    for (const auto &binding : host_reg_bindings)
        std::cout << "    Inferred arg" << binding.arg_index << " pushed from " << binding.reg_name << "\n";
    std::cout << "[*] Normalized " << std::dec << patched_vmentry_calls << " non-returning VMENTRY call(s)\n";

    // Patch register-based push/ret dispatchers into equivalent indirect jumps
    unsigned patched_dispatchers = 0;
    auto     patched_memory = patch_push_ret_dispatchers(call_patched_memory, &patched_dispatchers);
    std::cout << "[*] Normalized " << std::dec << patched_dispatchers << " *push r32; ret* dispatchers\n";

    // Set up virtual memory
    auto vm = build_virtual_memory_layout(ctx, memory);

    // ---------------------------------------------------------------
    // Lifting
    
    LiftingContext lifting_context(
        ctx, memory, std::move(arch), std::move(sem_module), std::move(patched_memory),
        std::move(vm), trace, resolved_param_count, save_intermediate,
        host_return_address, std::move(host_reg_bindings)
    );

    auto out_module = lifting_context.execute(replay_handlers);
    return LiftResult{std::move(ctx_owned), std::move(out_module)};
}
