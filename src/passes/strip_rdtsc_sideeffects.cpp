#include "strip_rdtsc_sideeffects.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

llvm::PreservedAnalyses StripRdtscSideEffectsPass::run(llvm::Module &module, llvm::ModuleAnalysisManager &)
{
    bool                         changed = false;
    llvm::SmallVector<llvm::Instruction *, 16> dead_rdtsc_calls;

    for (auto &fn : module)
    {
        if (fn.isDeclaration())
            continue;

        for (auto &bb : fn)
        {
            for (auto &inst : bb)
            {
                auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
                if (!call)
                    continue;

                auto *asm_value = llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand()->stripPointerCasts());
                if (!asm_value || llvm::StringRef(asm_value->getAsmString()).trim() != "rdtsc")
                    continue;

                // if the outputs are unused, we can delete the call
                if (call->use_empty())
                    dead_rdtsc_calls.push_back(call);
            }
        }
    }

    for (auto *inst : dead_rdtsc_calls)
    {
        inst->eraseFromParent();
        changed = true;
    }

    return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}
