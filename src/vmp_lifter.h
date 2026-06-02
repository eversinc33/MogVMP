#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "peload.h"

constexpr uint32_t kRFlagOffset = 2080;
constexpr uint32_t kAFlagOffset = 2064;

struct LiftResult
{
    std::unique_ptr<llvm::LLVMContext> ctx;
    std::unique_ptr<llvm::Module>      module;
    explicit                           operator bool() const
    {
        return module != nullptr;
    }
};

struct VmpTrace
{
    uint64_t vmenter = 0;
    uint64_t image_base = 0;  // Optional runtime image base (--imagebase).
};

// One --continue entry: where to resume after a VMEXIT, plus an optional explicit
// call target for that exit when the VM computes it as base+RVA (which doesn't
// fold to a constant in the prefix probe, e.g. an internal sub_XXXX call). When
// call_target is 0 the call is auto-resolved from the exit EIP (import thunks do
// fold). Syntax: "reentry" or "call@reentry".
struct ContinueEntry
{
    uint64_t reentry = 0;
    uint64_t call_target = 0;  // 0 = auto-resolve from the exit EIP
    unsigned call_argc = 0;    // stack args for this exit's call (0 = none/known)
};

class VmpLifter
{
   public:
    // Recover and lift the VM handler sequence starting at trace.vmenter until
    // the VMEXIT register-restore handler (detected heuristically).
    // Targets x86-32/Windows only. Returns a LiftResult (falsy on failure).
    // param_count: number of symbolic function arguments. Pass std::nullopt to
    // infer it from the `push <reg>` sequence ahead of the VMENTER call.
    LiftResult run(
        const Memory &memory, const PEInfo &pe_info, VmpTrace trace, std::optional<unsigned> param_count,
        bool save_intermediate = false, bool follow_vmexit = false,
        llvm::ArrayRef<ContinueEntry> continue_vmentries = {}, llvm::ArrayRef<uint64_t> replay_handlers = {}
    );
};
