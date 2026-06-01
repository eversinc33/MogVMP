#pragma once

#include <llvm/IR/PassManager.h>

struct StripRdtscSideEffectsPass : llvm::PassInfoMixin<StripRdtscSideEffectsPass>
{
    llvm::PreservedAnalyses run(llvm::Module &module, llvm::ModuleAnalysisManager &);
};
