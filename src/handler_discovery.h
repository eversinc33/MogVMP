#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "peload.h"

struct VirtualMemoryLayout
{
    llvm::ArrayType   *array_ty;
    llvm::Constant    *initializer;
    llvm::ConstantInt *base;
    llvm::ConstantInt *span;
};

// Describes the two successors of a VMP branchless conditional jump.
// "taken"/"fall" are named from the branchless selector's perspective:
// selector bit 1 chooses addr_taken/vsp_taken, selector bit 0 chooses
// addr_fall/vsp_fall. If selector_inverts_flag is true, selector bit 1 means
// the raw packed EFLAGS bit is 0 rather than 1.
struct VmpBranchInfo
{
    uint32_t addr_taken;   // next handler address when selector bit = 1
    uint32_t addr_fall;    // next handler address when selector bit = 0
    unsigned flag_shift;   // packed EFLAGS bit index feeding the selector
    bool selector_inverts_flag = false; // selector bit is !packed_EFLAGS[flag_shift]
    uint32_t vsp_taken = 0; // selected virtual-SP value when selector bit = 1
    uint32_t vsp_fall = 0;  // selected virtual-SP value when selector bit = 0
};

struct PrefixSnapshot
{
    uint32_t stack_base = 0;  // base VA of the VSP(ESI)-centered window (0 if not concretized)
    std::vector<std::optional<uint8_t>> state;
    std::vector<std::optional<uint8_t>> stack;

    // ESP-centered window, needed at a VMEXIT where the VM stack pointer (ESI) is
    // restored to a non-constant native value but ESP (and the native return
    // addresses / pushed call target on the pivoted frame) is concrete. Seeded back
    // by build_devirt so the exit fast-path can read the call target.
    uint32_t esp_stack_base = 0;  // base VA of the ESP-centered window (0 if absent)
    std::vector<std::optional<uint8_t>> esp_stack;
};

// Result of one prefix-discovery step.
struct PrefixResult
{
    enum class Kind { Concrete, Branch, Unknown };
    Kind                          kind   = Kind::Unknown;
    uint32_t                      next   = 0;   // valid when kind == Concrete
    VmpBranchInfo                 branch = {};  // valid when kind == Branch
    std::optional<PrefixSnapshot> snapshot;
};

using PrefixDiscoveryBuilder = std::function<llvm::Function *(llvm::StringRef function_name)>;

class DiscoveryEngine
{
public:
    DiscoveryEngine(
        llvm::Module   &module,
        llvm::Constant *vm_initializer,
        const Memory   &memory,
        bool            save_intermediate,
        unsigned       &step_counter
    );

    // Run one prefix-discovery step. Increments the step counter and returns
    // a Concrete, Branch, or Unknown result.
    // full_concretize: keep folding until a fixed point even after the return
    // value is already constant. Normal discovery stops as soon as the next
    // handler address is known; fork-snapshot capture instead needs the whole
    // VM register file (VSP/VIP/rolling key/...) concretized, not just the EIP.
    PrefixResult run_step(
        const PrefixDiscoveryBuilder           &build_prefix,
        std::optional<std::pair<unsigned, bool>> forced_branch_condition = std::nullopt,
        bool                                     full_concretize = false
    );

private:
    llvm::Module    &module_;
    llvm::Constant  *vm_initializer_;
    const Memory    &memory_;
    bool             save_intermediate_;
    unsigned        &step_counter_;

    void optimize(
        llvm::Function                          &fn,
        llvm::StringRef                          dump_prefix,
        std::optional<std::pair<unsigned, bool>> forced_branch_condition,
        bool                                     full_concretize = false
    );
    void run_always_inline();
};
