#pragma once

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>

// vmem_writable_lo/hi bound the VM's writable scratch (its data stack). Bytes of
// the virtual_memory alloca outside that window are treated as read-only PE image
// (code/bytecode) and keep folding from the initializer even after a non-constant
// store invalidates the writable region. Pass (0,0) to disable (all writable).
unsigned propagate_concrete_alloca_constants(
    llvm::Function &fn, llvm::Constant *virtual_memory_initializer,
    uint64_t vmem_writable_lo = 0, uint64_t vmem_writable_hi = 0
);
