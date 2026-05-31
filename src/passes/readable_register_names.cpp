#include "readable_register_names.h"

#include <llvm/ADT/StringSet.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <algorithm>
#include <string>
#include <vector>

namespace
{

std::string readable_name(size_t index)
{
    std::string name;
    do
    {
        name.push_back(static_cast<char>('a' + (index % 26)));
        index /= 26;
        if (index == 0)
            break;
        --index;
    } while (true);

    std::reverse(name.begin(), name.end());
    return name;
}

void rename_function_registers(llvm::Function &fn)
{
    std::vector<llvm::Instruction *> registers;
    llvm::StringSet<>                reserved;

    // Function arguments intentionally keep their descriptive ABI/source names.
    for (auto &arg : fn.args())
        if (arg.hasName())
            reserved.insert(arg.getName());

    for (auto &bb : fn)
    {
        if (bb.hasName())
            reserved.insert(bb.getName());
        for (auto &inst : bb)
            if (!inst.getType()->isVoidTy())
                registers.push_back(&inst);
    }

    // First move all renamed values out of the way so stale names like %.pre or
    // %a do not force LLVM to uniquify the new readable sequence.
    for (size_t i = 0; i < registers.size(); ++i)
        registers[i]->setName("__readable_tmp_" + std::to_string(i));

    size_t next = 0;
    for (auto *inst : registers)
    {
        std::string name;
        do
        {
            name = readable_name(next++);
        } while (reserved.count(name));

        inst->setName(name);
        reserved.insert(name);
    }
}

}  // namespace

llvm::PreservedAnalyses ReadableRegisterNamesPass::run(llvm::Module &module, llvm::ModuleAnalysisManager &)
{
    for (auto &fn : module)
        if (!fn.isDeclaration())
            rename_function_registers(fn);

    // This pass changes only IR value names, not semantics or CFG/dataflow.
    return llvm::PreservedAnalyses::all();
}
