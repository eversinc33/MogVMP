#pragma once

#include <llvm/IR/PassManager.h>

// Renames basic blocks to a readable sequence: the entry block becomes "entry"
// and the remaining blocks become "bb1", "bb2", ... in layout order. Run after
// ReadableRegisterNamesPass so register names are already settled.
struct ReadableBlockNamesPass : llvm::PassInfoMixin<ReadableBlockNamesPass>
{
    llvm::PreservedAnalyses run(llvm::Module &module, llvm::ModuleAnalysisManager &);
};
