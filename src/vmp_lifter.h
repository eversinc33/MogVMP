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

class VmpLifter
{
   public:
    // Recover and lift the VM handler sequence starting at trace.vmenter until
    // the VMEXIT register-restore handler (detected heuristically).
    // Targets x86-32/Windows only. Returns a LiftResult (falsy on failure).
    // param_count: number of symbolic function arguments. Pass std::nullopt to
    // infer it from the `push <reg>` sequence ahead of the VMENTER call.
    // continue_vmentries: VMENTER addresses of the VMs reached after each VMEXIT,
    // in order. When non-empty the lifter follows VMEXITs (emitting the external
    // call the VM exited to) and resumes at the next listed VM; empty means lift a
    // single VM and stop at its first VMEXIT.
    LiftResult run(
        const Memory &memory, const PEInfo &pe_info, VmpTrace trace, std::optional<unsigned> param_count,
        bool save_intermediate = false, llvm::ArrayRef<uint64_t> continue_vmentries = {}
    );
};
