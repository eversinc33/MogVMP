#include "readable_block_names.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <string>

namespace
{

void rename_function_blocks(llvm::Function &fn)
{
    // Move every block to a unique temporary name first so setting "entry"/"bbN"
    // below can never collide with a block that still holds the target name and
    // force LLVM to uniquify it (e.g. into "entry1").
    std::size_t tmp = 0;
    for (auto &bb : fn)
        bb.setName("__bb_tmp_" + std::to_string(tmp++));

    std::size_t index = 0;
    for (auto &bb : fn)
    {
        bb.setName(index == 0 ? "entry" : ("bb" + std::to_string(index)));
        ++index;
    }
}

}  // namespace

llvm::PreservedAnalyses ReadableBlockNamesPass::run(llvm::Module &module, llvm::ModuleAnalysisManager &)
{
    for (auto &fn : module)
        if (!fn.isDeclaration())
            rename_function_blocks(fn);

    // This pass changes only basic-block labels, not semantics or CFG/dataflow.
    return llvm::PreservedAnalyses::all();
}
