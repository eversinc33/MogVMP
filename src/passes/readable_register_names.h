#pragma once

#include <llvm/IR/PassManager.h>

struct ReadableRegisterNamesPass : llvm::PassInfoMixin<ReadableRegisterNamesPass>
{
    llvm::PreservedAnalyses run(llvm::Module &module, llvm::ModuleAnalysisManager &);
};
