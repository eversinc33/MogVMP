#include "concrete_alloca_prop.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/Casting.h>
#include <llvm/TargetParser/Triple.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

std::optional<uint8_t> read_initializer_byte(llvm::Constant *init, uint64_t off)
{
    if (auto *cda = llvm::dyn_cast<llvm::ConstantDataArray>(init))
    {
        if (off >= cda->getNumElements())
            return std::nullopt;
        return static_cast<uint8_t>(cda->getElementAsInteger(off));
    }
    if (llvm::isa<llvm::ConstantAggregateZero>(init))
        return 0;
    if (auto *ca = llvm::dyn_cast<llvm::ConstantArray>(init))
    {
        if (off >= ca->getNumOperands())
            return std::nullopt;
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(ca->getOperand(off)))
            return static_cast<uint8_t>(ci->getZExtValue());
    }
    return std::nullopt;
}

std::optional<uint64_t> constant_offset_from_alloca(
    llvm::Value *ptr, llvm::AllocaInst *base, const llvm::DataLayout &dl
)
{
    ptr = ptr->stripPointerCasts();
    if (ptr == base)
        return 0;

    auto *gep = llvm::dyn_cast<llvm::GEPOperator>(ptr);
    if (!gep || gep->getPointerOperand()->stripPointerCasts() != base)
        return std::nullopt;

    llvm::APInt off(dl.getIndexTypeSizeInBits(ptr->getType()), 0);
    if (!gep->accumulateConstantOffset(dl, off))
        return std::nullopt;
    if (off.isNegative())
        return std::nullopt;
    return off.getZExtValue();
}

bool constant_to_bytes(llvm::Constant *c, uint64_t width, bool little_endian, llvm::SmallVectorImpl<uint8_t> &bytes)
{
    if (auto *ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(c))
    {
        auto v = ci->getValue();
        bytes.clear();
        bytes.reserve(width);
        for (uint64_t i = 0; i < width; ++i)
        {
            uint64_t shift = little_endian ? i * 8 : (width - 1 - i) * 8;
            bytes.push_back(static_cast<uint8_t>((v.lshr(shift).getLimitedValue()) & 0xffu));
        }
        return true;
    }
    if (auto *cda = llvm::dyn_cast_or_null<llvm::ConstantDataArray>(c))
    {
        if (cda->getNumElements() != width)
            return false;
        bytes.clear();
        bytes.reserve(width);
        for (uint64_t i = 0; i < width; ++i)
            bytes.push_back(static_cast<uint8_t>(cda->getElementAsInteger(i)));
        return true;
    }
    if (auto *ca = llvm::dyn_cast_or_null<llvm::ConstantArray>(c))
    {
        if (ca->getNumOperands() != width)
            return false;
        bytes.clear();
        bytes.reserve(width);
        for (uint64_t i = 0; i < width; ++i)
        {
            auto *ci = llvm::dyn_cast<llvm::ConstantInt>(ca->getOperand(i));
            if (!ci)
                return false;
            bytes.push_back(static_cast<uint8_t>(ci->getZExtValue()));
        }
        return true;
    }
    return false;
}

struct ConcreteAllocaDomain
{
    llvm::AllocaInst *alloca = nullptr;
    llvm::StoreInst  *init_store = nullptr;
    llvm::Constant   *initializer = nullptr;
    bool              zero_default = false;
    // Writable window [writable_lo, writable_hi). When set (hi > lo), bytes
    // OUTSIDE this window are treated as read-only: a non-constant (computed)
    // store can only target the VM's writable scratch (its data stack, inside the
    // window), so it cannot alias the read-only PE image (code/bytecode) outside
    // it. Read-only bytes therefore keep reading the initializer even after a
    // symbolic store invalidates default_valid for the writable region.
    uint64_t          writable_lo = 0;
    uint64_t          writable_hi = 0;
};

using FactMap = llvm::DenseMap<uint64_t, int>;  // -1 => unknown, 0..255 => known

struct DomainState
{
    // If true, missing facts read from the domain initializer/zero default. If
    // false, a missing fact is unknown because a non-constant store may have
    // clobbered an arbitrary byte in the domain.
    bool    default_valid = true;
    FactMap facts;
};

using ProgramState = llvm::SmallVector<DomainState, 2>;

std::optional<uint8_t> read_domain_default(const ConcreteAllocaDomain &domain, uint64_t off)
{
    if (domain.initializer)
        return read_initializer_byte(domain.initializer, off);
    if (domain.zero_default)
        return 0;
    return std::nullopt;
}

bool is_read_only_offset(const ConcreteAllocaDomain &domain, uint64_t off)
{
    // A writable window of (lo,hi) marks everything outside it read-only.
    return domain.writable_hi > domain.writable_lo &&
           (off < domain.writable_lo || off >= domain.writable_hi);
}

int effective_byte(const ConcreteAllocaDomain &domain, const DomainState &state, uint64_t off)
{
    auto it = state.facts.find(off);
    if (it != state.facts.end())
        return it->second;
    // Read the initializer when the default is still valid, OR when this byte is
    // read-only (outside the writable window) and therefore cannot have been
    // clobbered by the non-constant store that invalidated default_valid.
    if (state.default_valid || is_read_only_offset(domain, off))
        if (auto b = read_domain_default(domain, off))
            return *b;
    return -1;
}

std::optional<uint8_t> read_fact_byte(const ConcreteAllocaDomain &domain, const DomainState &state, uint64_t off)
{
    int byte = effective_byte(domain, state, off);
    if (byte < 0)
        return std::nullopt;
    return static_cast<uint8_t>(byte);
}

bool is_based_on_alloca(llvm::Value *ptr, llvm::AllocaInst *base)
{
    ptr = ptr->stripPointerCasts();
    if (ptr == base)
        return true;
    if (auto *gep = llvm::dyn_cast<llvm::GEPOperator>(ptr))
        return is_based_on_alloca(gep->getPointerOperand(), base);
    return false;
}

std::optional<unsigned> find_domain_base_for_ptr(llvm::Value *ptr, llvm::ArrayRef<ConcreteAllocaDomain> domains)
{
    for (unsigned i = 0; i < domains.size(); ++i)
        if (domains[i].alloca && is_based_on_alloca(ptr, domains[i].alloca))
            return i;
    return std::nullopt;
}

std::optional<unsigned> find_domain_index_for_ptr(
    llvm::Value *ptr, llvm::ArrayRef<ConcreteAllocaDomain> domains, const llvm::DataLayout &dl, uint64_t &off_out
)
{
    for (unsigned i = 0; i < domains.size(); ++i)
    {
        if (!domains[i].alloca)
            continue;
        if (auto off = constant_offset_from_alloca(ptr, domains[i].alloca, dl))
        {
            off_out = *off;
            return i;
        }
    }
    return std::nullopt;
}

// Maps each store whose pointer is a non-constant offset into a domain alloca to
// the [lo, hi) byte range it can touch, when that range is provably bounded.
// Stores absent from the map have an unknown range and must be treated as a
// whole-domain clobber.
using BoundedStoreRanges = llvm::DenseMap<llvm::StoreInst *, std::pair<uint64_t, uint64_t>>;

// Use ScalarEvolution to bound the byte range of strided (loop-induction) stores
// into a domain alloca. VMProtect leans on REP MOVS / context copies whose
// destination index is an affine IV; without a bound the dataflow pass clobbers
// the entire 4 GiB domain, erasing facts (e.g. the next-handler EIP word) that
// the copy provably does not overlap. A bounded range lets us invalidate only the
// touched bytes. Anything we cannot bound is left out of the map (full clobber).
void compute_bounded_store_ranges(
    llvm::Function &fn, llvm::ArrayRef<ConcreteAllocaDomain> domains, const llvm::DataLayout &dl,
    BoundedStoreRanges &out
)
{
    // Cap the range we are willing to materialize as explicit unknown facts; an
    // enormous bound is no better than a full clobber and would bloat the maps.
    constexpr uint64_t kMaxRangeBytes = 1u << 16;

    llvm::DominatorTree         dt(fn);
    llvm::LoopInfo              li(dt);
    llvm::AssumptionCache       ac(fn);
    llvm::TargetLibraryInfoImpl tlii{llvm::Triple(fn.getParent()->getTargetTriple())};
    llvm::TargetLibraryInfo     tli(tlii);
    llvm::ScalarEvolution       se(fn, tli, ac, dt, li);

    for (auto &bb : fn)
    {
        for (auto &inst : bb)
        {
            auto *si = llvm::dyn_cast<llvm::StoreInst>(&inst);
            if (!si)
                continue;
            llvm::Value *ptr = si->getPointerOperand();

            // Constant-offset stores are already handled precisely; only the
            // non-constant ones into a domain alloca need bounding.
            uint64_t const_off = 0;
            if (find_domain_index_for_ptr(ptr, domains, dl, const_off))
                continue;
            auto base_domain = find_domain_base_for_ptr(ptr, domains);
            if (!base_domain)
                continue;

            // Offset (in bytes) of the store pointer relative to the alloca base.
            const llvm::SCEV *off_scev =
                se.getMinusSCEV(se.getSCEV(ptr), se.getSCEV(domains[*base_domain].alloca));
            if (llvm::isa<llvm::SCEVCouldNotCompute>(off_scev))
                continue;

            llvm::ConstantRange range = se.getUnsignedRange(off_scev);
            if (range.isFullSet() || range.isWrappedSet())
                continue;

            llvm::APInt min_off = range.getUnsignedMin();
            llvm::APInt max_off = range.getUnsignedMax();
            if (min_off.getActiveBits() > 64 || max_off.getActiveBits() > 64)
                continue;

            uint64_t width = dl.getTypeStoreSize(si->getValueOperand()->getType());
            if (!width)
                continue;
            uint64_t lo = min_off.getZExtValue();
            uint64_t hi = max_off.getZExtValue() + width;  // exclusive
            if (hi <= lo || hi - lo > kMaxRangeBytes)
                continue;

            out[si] = {lo, hi};
        }
    }
}

bool same_state(const ProgramState &a, const ProgramState &b)
{
    if (a.size() != b.size())
        return false;
    for (unsigned i = 0; i < a.size(); ++i)
    {
        if (a[i].default_valid != b[i].default_valid || a[i].facts.size() != b[i].facts.size())
            return false;
        for (const auto &kv : a[i].facts)
        {
            auto it = b[i].facts.find(kv.first);
            if (it == b[i].facts.end() || it->second != kv.second)
                return false;
        }
    }
    return true;
}

bool returns_only_constants(llvm::Function &fn)
{
    bool saw_ret = false;
    for (auto &bb : fn)
    {
        if (auto *ret = llvm::dyn_cast<llvm::ReturnInst>(bb.getTerminator()))
        {
            saw_ret = true;
            if (ret->getReturnValue() && !llvm::isa<llvm::Constant>(ret->getReturnValue()))
                return false;
        }
    }
    return saw_ret;
}

unsigned prune_dead_tracked_memory_after_constant_return(
    llvm::Function &fn, llvm::ArrayRef<ConcreteAllocaDomain> domains, const llvm::DataLayout &dl
)
{
    if (!returns_only_constants(fn))
        return 0;

    unsigned                                    changed = 0;
    llvm::SmallVector<llvm::Instruction *, 128> erase;
    for (auto &bb : fn)
    {
        for (auto &inst : bb)
        {
            if (auto *si = llvm::dyn_cast<llvm::StoreInst>(&inst))
            {
                uint64_t off = 0;
                if (find_domain_index_for_ptr(si->getPointerOperand(), domains, dl, off))
                    erase.push_back(si);
            }
        }
    }

    for (auto *i : erase)
    {
        i->eraseFromParent();
        ++changed;
    }

    bool local_changed = true;
    while (local_changed)
    {
        local_changed = false;
        erase.clear();
        for (auto &bb : fn)
            for (auto &inst : bb)
                if (inst.use_empty() && !inst.isTerminator() && !inst.mayHaveSideEffects())
                    erase.push_back(&inst);
        for (auto *i : erase)
        {
            i->eraseFromParent();
            ++changed;
            local_changed = true;
        }
    }

    return changed;
}

ProgramState meet_predecessors(
    llvm::BasicBlock &bb, llvm::DenseMap<llvm::BasicBlock *, ProgramState> &out_state,
    llvm::DenseMap<llvm::BasicBlock *, llvm::DenseSet<llvm::BasicBlock *>> &executable_preds,
    llvm::ArrayRef<ConcreteAllocaDomain>                                    domains
)
{
    ProgramState result;
    result.resize(domains.size());

    auto exec_it = executable_preds.find(&bb);
    if (exec_it == executable_preds.end())
        return result;

    bool first_pred = true;
    for (auto *pred : exec_it->second)
    {
        auto it = out_state.find(pred);
        if (it == out_state.end())
            continue;

        if (first_pred)
        {
            result = it->second;
            first_pred = false;
            continue;
        }

        const ProgramState &pred_state = it->second;
        for (unsigned d = 0; d < domains.size(); ++d)
        {
            llvm::DenseSet<uint64_t> keys;
            for (const auto &kv : result[d].facts) keys.insert(kv.first);
            for (const auto &kv : pred_state[d].facts) keys.insert(kv.first);

            DomainState merged;
            merged.default_valid = result[d].default_valid && pred_state[d].default_valid;
            for (uint64_t off : keys)
            {
                int a = effective_byte(domains[d], result[d], off);
                int b = effective_byte(domains[d], pred_state[d], off);
                int v = (a == b) ? a : -1;

                auto def = read_domain_default(domains[d], off);
                if (v >= 0 && merged.default_valid && def && *def == static_cast<uint8_t>(v))
                    continue;  // default already represents this fact
                merged.facts[off] = v;
            }
            result[d] = std::move(merged);
        }
    }

    return result;
}

void add_executable_edge(
    llvm::BasicBlock *from, llvm::BasicBlock *to, llvm::DenseSet<llvm::BasicBlock *> &executable_blocks,
    llvm::DenseMap<llvm::BasicBlock *, llvm::DenseSet<llvm::BasicBlock *>> &executable_preds, bool &changed
)
{
    if (executable_blocks.insert(to).second)
        changed = true;
    if (executable_preds[to].insert(from).second)
        changed = true;
}

llvm::ConstantInt *known_const(llvm::Value *v, llvm::DenseMap<llvm::Value *, llvm::ConstantInt *> &env)
{
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(v))
        return ci;
    auto it = env.find(v);
    return it == env.end() ? nullptr : it->second;
}

struct SymFact
{
    unsigned     domain;
    uint64_t     off;
    uint64_t     width;
    llvm::Value *value;
};

ProgramState transfer_block(
    llvm::BasicBlock &bb, ProgramState state, llvm::ArrayRef<ConcreteAllocaDomain> domains, const llvm::DataLayout &dl,
    bool little_endian, bool rewrite, unsigned &changed,
    llvm::SmallVectorImpl<llvm::BasicBlock *> *known_successors = nullptr,
    llvm::SmallVectorImpl<SymFact>            *persistent_sym_facts = nullptr,
    const BoundedStoreRanges                  *bounded_store_ranges = nullptr
)
{
    llvm::SmallVector<llvm::Instruction *, 32>         erase;
    llvm::DenseMap<llvm::Value *, llvm::ConstantInt *> env;
    llvm::SmallVector<SymFact, 32>                     local_sym_facts;
    auto &sym_facts = persistent_sym_facts ? *persistent_sym_facts : local_sym_facts;

    auto overlaps = [](uint64_t a_off, uint64_t a_width, uint64_t b_off, uint64_t b_width)
    { return a_off < b_off + b_width && b_off < a_off + a_width; };
    auto kill_sym = [&](unsigned domain, uint64_t off, uint64_t width)
    {
        llvm::erase_if(
            sym_facts, [&](const SymFact &f) { return f.domain == domain && overlaps(f.off, f.width, off, width); }
        );
    };
    auto kill_domain_sym = [&](unsigned domain)
    { llvm::erase_if(sym_facts, [&](const SymFact &f) { return f.domain == domain; }); };
    auto find_sym = [&](unsigned domain, uint64_t off, uint64_t width) -> llvm::Value *
    {
        for (const auto &f : sym_facts)
            if (f.domain == domain && f.off == off && f.width == width)
                return f.value;
        return nullptr;
    };

    for (auto &inst : bb)
    {
        bool is_init_store = false;
        for (const auto &domain : domains) is_init_store |= (&inst == domain.init_store);
        if (is_init_store)
            continue;

        if (auto *li = llvm::dyn_cast<llvm::LoadInst>(&inst))
        {
            uint64_t off = 0;
            auto     domain_idx = find_domain_index_for_ptr(li->getPointerOperand(), domains, dl, off);
            if (!domain_idx)
                continue;

            uint64_t width = dl.getTypeStoreSize(li->getType());
            if (!width || !li->getType()->isIntegerTy())
                continue;

            if (rewrite)
            {
                if (auto *sym = find_sym(*domain_idx, off, width))
                {
                    if (sym->getType() == li->getType())
                    {
                        li->replaceAllUsesWith(sym);
                        erase.push_back(li);
                        ++changed;
                        continue;
                    }
                }
            }

            llvm::APInt value(li->getType()->getIntegerBitWidth(), 0);
            bool        ok = true;
            for (uint64_t i = 0; i < width; ++i)
            {
                auto byte = read_fact_byte(domains[*domain_idx], state[*domain_idx], off + i);
                if (!byte)
                {
                    ok = false;
                    break;
                }

                uint64_t shift = little_endian ? i * 8 : (width - 1 - i) * 8;
                value |= llvm::APInt(value.getBitWidth(), static_cast<uint64_t>(*byte)) << shift;
            }
            if (!ok)
                continue;

            auto *ci = llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(li->getType(), value));
            env[li] = ci;
            if (rewrite)
            {
                li->replaceAllUsesWith(ci);
                erase.push_back(li);
                ++changed;
            }
            continue;
        }

        if (auto *si = llvm::dyn_cast<llvm::StoreInst>(&inst))
        {
            uint64_t off = 0;
            auto     domain_idx = find_domain_index_for_ptr(si->getPointerOperand(), domains, dl, off);
            if (!domain_idx)
            {
                if (auto base_domain = find_domain_base_for_ptr(si->getPointerOperand(), domains))
                {
                    unsigned d = static_cast<unsigned>(*base_domain);
                    // A store through a non-constant GEP may alias any previously
                    // tracked byte/symbolic fact in this alloca.  Keeping old facts
                    // here is unsound; VMProtect uses computed VM-stack addresses
                    // heavily, and stale forwarded facts can change program semantics.
                    // When ScalarEvolution could bound the strided store to a byte
                    // range, invalidate only that range (keeping default_valid and
                    // every fact outside it) instead of clobbering the whole domain.
                    auto range_it = bounded_store_ranges ? bounded_store_ranges->find(si)
                                                          : BoundedStoreRanges::const_iterator{};
                    if (bounded_store_ranges && range_it != bounded_store_ranges->end())
                    {
                        auto [lo, hi] = range_it->second;
                        for (uint64_t o = lo; o < hi; ++o)
                            state[d].facts[o] = -1;
                        kill_sym(d, lo, hi - lo);
                    }
                    else
                    {
                        state[d].facts.clear();
                        state[d].default_valid = false;
                        kill_domain_sym(d);
                    }
                }
                continue;
            }

            uint64_t width = dl.getTypeStoreSize(si->getValueOperand()->getType());
            if (!width)
                continue;

            kill_sym(*domain_idx, off, width);

            llvm::SmallVector<uint8_t, 16> bytes;
            if (constant_to_bytes(
                    llvm::dyn_cast<llvm::Constant>(si->getValueOperand()), width, little_endian, bytes
                ))
            {
                for (uint64_t i = 0; i < width; ++i)
                {
                    auto def = read_domain_default(domains[*domain_idx], off + i);
                    if (state[*domain_idx].default_valid && def && *def == bytes[i])
                        state[*domain_idx].facts.erase(off + i);
                    else
                        state[*domain_idx].facts[off + i] = bytes[i];
                }
            }
            else
            {
                for (uint64_t i = 0; i < width; ++i) state[*domain_idx].facts[off + i] = -1;
                if (rewrite && si->getValueOperand()->getType()->isIntegerTy())
                    sym_facts.push_back({static_cast<unsigned>(*domain_idx), off, width, si->getValueOperand()});
            }
            continue;
        }

        if (auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(&inst))
        {
            auto *a = known_const(bo->getOperand(0), env);
            auto *b = known_const(bo->getOperand(1), env);
            if (a && b)
            {
                const llvm::APInt &av = a->getValue();
                const llvm::APInt &bv = b->getValue();
                llvm::APInt        r(av.getBitWidth(), 0);
                switch (bo->getOpcode())
                {
                    case llvm::Instruction::Add: r = av + bv; break;
                    case llvm::Instruction::Sub: r = av - bv; break;
                    case llvm::Instruction::Mul: r = av * bv; break;
                    case llvm::Instruction::And: r = av & bv; break;
                    case llvm::Instruction::Or: r = av | bv; break;
                    case llvm::Instruction::Xor: r = av ^ bv; break;
                    case llvm::Instruction::Shl: r = av.shl(bv); break;
                    case llvm::Instruction::LShr: r = av.lshr(bv); break;
                    case llvm::Instruction::AShr: r = av.ashr(bv); break;
                    default: continue;
                }
                auto *ci = llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(bo->getType(), r));
                env[&inst] = ci;
                if (rewrite)
                {
                    inst.replaceAllUsesWith(ci);
                    erase.push_back(&inst);
                    ++changed;
                }
            }
            continue;
        }

        if (auto *ci = llvm::dyn_cast<llvm::CastInst>(&inst))
        {
            if (auto *v = known_const(ci->getOperand(0), env))
            {
                auto        bw = ci->getType()->getIntegerBitWidth();
                llvm::APInt r = v->getValue();
                if (r.getBitWidth() < bw)
                    r = ci->getOpcode() == llvm::Instruction::SExt ? r.sext(bw) : r.zext(bw);
                else if (r.getBitWidth() > bw)
                    r = r.trunc(bw);
                auto *kc = llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(ci->getType(), r));
                env[&inst] = kc;
                if (rewrite)
                {
                    inst.replaceAllUsesWith(kc);
                    erase.push_back(&inst);
                    ++changed;
                }
            }
            continue;
        }

        if (auto *ev = llvm::dyn_cast<llvm::ExtractValueInst>(&inst))
        {
            if (auto *agg = llvm::dyn_cast<llvm::ConstantAggregate>(ev->getAggregateOperand()))
            {
                auto idx = ev->getIndices()[0];
                if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(agg->getOperand(idx)))
                {
                    env[&inst] = ci;
                    if (rewrite)
                    {
                        inst.replaceAllUsesWith(ci);
                        erase.push_back(&inst);
                        ++changed;
                    }
                }
            }
            continue;
        }

        if (auto *icmp = llvm::dyn_cast<llvm::ICmpInst>(&inst))
        {
            auto *a = known_const(icmp->getOperand(0), env);
            auto *b = known_const(icmp->getOperand(1), env);
            if (a && b)
            {
                bool r = false;
                switch (icmp->getPredicate())
                {
                    case llvm::CmpInst::ICMP_EQ: r = a->getValue().eq(b->getValue()); break;
                    case llvm::CmpInst::ICMP_NE: r = a->getValue().ne(b->getValue()); break;
                    case llvm::CmpInst::ICMP_ULT: r = a->getValue().ult(b->getValue()); break;
                    case llvm::CmpInst::ICMP_ULE: r = a->getValue().ule(b->getValue()); break;
                    case llvm::CmpInst::ICMP_UGT: r = a->getValue().ugt(b->getValue()); break;
                    case llvm::CmpInst::ICMP_UGE: r = a->getValue().uge(b->getValue()); break;
                    case llvm::CmpInst::ICMP_SLT: r = a->getValue().slt(b->getValue()); break;
                    case llvm::CmpInst::ICMP_SLE: r = a->getValue().sle(b->getValue()); break;
                    case llvm::CmpInst::ICMP_SGT: r = a->getValue().sgt(b->getValue()); break;
                    case llvm::CmpInst::ICMP_SGE: r = a->getValue().sge(b->getValue()); break;
                    default: break;
                }
                auto *ci = llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(icmp->getType(), r));
                env[&inst] = ci;
                if (rewrite)
                {
                    inst.replaceAllUsesWith(ci);
                    erase.push_back(&inst);
                    ++changed;
                }
            }
            continue;
        }

        if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&inst))
        {
            auto *callee = cb->getCalledFunction();
            if (!callee)
                continue;
            auto name = callee->getName();
            if ((name == "__remill_atomic_begin" || name == "__remill_barrier_store_load" ||
                 name == "__remill_atomic_end") &&
                cb->arg_size() >= 1)
            {
                if (rewrite)
                {
                    inst.replaceAllUsesWith(cb->getArgOperand(0));
                    erase.push_back(&inst);
                    ++changed;
                }
                continue;
            }
            if (name == "llvm.bswap.i32" && cb->arg_size() == 1)
                if (auto *a = known_const(cb->getArgOperand(0), env))
                {
                    auto *ci =
                        llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(cb->getType(), a->getValue().byteSwap()));
                    env[&inst] = ci;
                    if (rewrite)
                    {
                        inst.replaceAllUsesWith(ci);
                        erase.push_back(&inst);
                        ++changed;
                    }
                }
            if ((name == "llvm.fshl.i8" || name == "llvm.fshr.i32" || name == "llvm.fshl.i32") && cb->arg_size() == 3)
            {
                auto *a = known_const(cb->getArgOperand(0), env);
                auto *b = known_const(cb->getArgOperand(1), env);
                auto *c = known_const(cb->getArgOperand(2), env);
                if (a && b && c)
                {
                    if (a == b)
                    {
                        bool     left = name.contains("fshl");
                        unsigned sh = c->getZExtValue() % a->getBitWidth();
                        auto     r = left ? a->getValue().rotl(sh) : a->getValue().rotr(sh);
                        auto    *ci = llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(cb->getType(), r));
                        env[&inst] = ci;
                        if (rewrite)
                        {
                            inst.replaceAllUsesWith(ci);
                            erase.push_back(&inst);
                            ++changed;
                        }
                    }
                }
            }
            continue;
        }
    }

    if (known_successors)
    {
        auto *term = bb.getTerminator();
        if (auto *br = llvm::dyn_cast<llvm::BranchInst>(term); br && br->isConditional())
        {
            if (auto *c = known_const(br->getCondition(), env))
                known_successors->push_back(br->getSuccessor(c->isZero() ? 1 : 0));
        }
        else if (auto *sw = llvm::dyn_cast<llvm::SwitchInst>(term))
        {
            if (auto *c = known_const(sw->getCondition(), env))
                known_successors->push_back(sw->findCaseValue(c)->getCaseSuccessor());
        }
        if (known_successors->empty())
            for (unsigned i = 0; i < term->getNumSuccessors(); ++i) known_successors->push_back(term->getSuccessor(i));
    }

    for (auto *i : erase) i->eraseFromParent();

    return state;
}

unsigned propagate_concrete_alloca_constants(
    llvm::Function &fn, llvm::Constant *virtual_memory_initializer,
    uint64_t vmem_writable_lo, uint64_t vmem_writable_hi
)
{
    ConcreteAllocaDomain state_domain;
    ConcreteAllocaDomain vmem_domain;

    for (auto &bb : fn)
        for (auto &inst : bb)
            if (auto *ai = llvm::dyn_cast<llvm::AllocaInst>(&inst))
            {
                if (ai->getName() == "state")
                {
                    state_domain.alloca = ai;
                    state_domain.zero_default = true;
                }
                else if (ai->getName() == "virtual_memory")
                {
                    vmem_domain.alloca = ai;
                    vmem_domain.initializer = virtual_memory_initializer;
                    // Outside the writable VM-stack window the PE image is read-only
                    // and keeps folding from the initializer even after computed
                    // (symbolic) stores invalidate the writable region.
                    vmem_domain.writable_lo = vmem_writable_lo;
                    vmem_domain.writable_hi = vmem_writable_hi;
                }
            }

    for (auto &bb : fn)
        for (auto &inst : bb)
            if (auto *si = llvm::dyn_cast<llvm::StoreInst>(&inst))
            {
                if (state_domain.alloca && si->getPointerOperand()->stripPointerCasts() == state_domain.alloca &&
                    llvm::isa<llvm::ConstantAggregateZero>(si->getValueOperand()))
                    state_domain.init_store = si;
                if (vmem_domain.alloca && si->getPointerOperand()->stripPointerCasts() == vmem_domain.alloca &&
                    si->getValueOperand() == virtual_memory_initializer)
                    vmem_domain.init_store = si;
            }

    llvm::SmallVector<ConcreteAllocaDomain, 2> domains;
    if (state_domain.alloca)
        domains.push_back(state_domain);
    if (vmem_domain.alloca)
        domains.push_back(vmem_domain);
    if (domains.empty())
        return 0;

    const auto &dl = fn.getParent()->getDataLayout();
    const bool  little_endian = dl.isLittleEndian();

    // Bound strided (induction-variable) stores so they invalidate only the bytes
    // they can touch instead of the whole domain. Computed once on the original IR;
    // the store keys survive the rewrite phase (stores are never folded away here),
    // and any store whose index later folds to a constant is handled by the precise
    // constant-offset path before the clobber branch is reached.
    BoundedStoreRanges bounded_store_ranges;
    compute_bounded_store_ranges(fn, domains, dl, bounded_store_ranges);

    llvm::DenseMap<llvm::BasicBlock *, ProgramState>                       in_state;
    llvm::DenseMap<llvm::BasicBlock *, ProgramState>                       out_state;
    llvm::DenseSet<llvm::BasicBlock *>                                     executable_blocks;
    llvm::DenseMap<llvm::BasicBlock *, llvm::DenseSet<llvm::BasicBlock *>> executable_preds;

    ProgramState empty;
    empty.resize(domains.size());
    executable_blocks.insert(&fn.getEntryBlock());

    bool changed_analysis = true;
    while (changed_analysis)
    {
        changed_analysis = false;
        for (auto &bb : fn)
        {
            if (!executable_blocks.contains(&bb))
                continue;

            ProgramState in =
                (&bb == &fn.getEntryBlock()) ? empty : meet_predecessors(bb, out_state, executable_preds, domains);
            unsigned                                 dummy_changed = 0;
            llvm::SmallVector<llvm::BasicBlock *, 2> known_successors;
            ProgramState                             out =
                transfer_block(bb, in, domains, dl, little_endian, /*rewrite=*/false, dummy_changed, &known_successors,
                               /*persistent_sym_facts=*/nullptr, &bounded_store_ranges);

            auto old_in = in_state.find(&bb);
            if (old_in == in_state.end() || !same_state(old_in->second, in))
            {
                in_state[&bb] = in;
                changed_analysis = true;
            }
            auto old_out = out_state.find(&bb);
            if (old_out == out_state.end() || !same_state(old_out->second, out))
            {
                out_state[&bb] = std::move(out);
                changed_analysis = true;
            }

            for (auto *succ : known_successors)
                add_executable_edge(&bb, succ, executable_blocks, executable_preds, changed_analysis);
        }
    }

    unsigned changed = 0;
    for (auto &bb : fn)
    {
        if (!executable_blocks.contains(&bb))
            continue;
        auto         it = in_state.find(&bb);
        ProgramState in = (it == in_state.end()) ? empty : it->second;
        (void)transfer_block(bb, in, domains, dl, little_endian, /*rewrite=*/true, changed,
                             /*known_successors=*/nullptr, /*persistent_sym_facts=*/nullptr, &bounded_store_ranges);
    }

    // After CFG cleanup the devirt function often becomes a single straight-line
    // block. If so, do one linear sweep to pick up facts that were conservatively
    // killed while the CFG still had joins in previous outer iterations.
    bool straight_line = true;
    for (auto &bb : fn)
    {
        auto *term = bb.getTerminator();
        if (term->getNumSuccessors() > 1)
        {
            straight_line = false;
            break;
        }
    }
    if (straight_line)
    {
        ProgramState                   linear = empty;
        llvm::SmallVector<SymFact, 32> linear_sym_facts;
        for (auto &bb : fn)
            linear = transfer_block(
                bb, linear, domains, dl, little_endian, /*rewrite=*/true, changed, nullptr, &linear_sym_facts,
                &bounded_store_ranges
            );
    }

    changed += prune_dead_tracked_memory_after_constant_return(fn, domains, dl);

    return changed;
}
