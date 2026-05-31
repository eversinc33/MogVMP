#pragma once

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>

#include <utility>

namespace helpers
{

struct LlvmPipelineManagers
{
    llvm::LoopAnalysisManager     lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager    cam;
    llvm::ModuleAnalysisManager   mam;
    llvm::PassBuilder             pb;

    LlvmPipelineManagers()
    {
        pb.registerModuleAnalyses(mam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.registerCGSCCAnalyses(cam);
        pb.crossRegisterProxies(lam, fam, cam, mam);
    }
};

inline void run_module_pipeline(llvm::Module &module, llvm::ModulePassManager mpm)
{
    LlvmPipelineManagers managers;
    mpm.run(module, managers.mam);
}

inline void run_function_pipeline(llvm::Function &fn, llvm::FunctionPassManager fpm)
{
    LlvmPipelineManagers managers;
    fpm.run(fn, managers.fam);
}

inline void run_function_pipeline_on_module(llvm::Module &module, llvm::FunctionPassManager fpm)
{
    llvm::ModulePassManager mpm;
    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
    run_module_pipeline(module, std::move(mpm));
}

inline void run_default_o3_pipeline(llvm::Module &module)
{
    LlvmPipelineManagers    managers;
    llvm::ModulePassManager mpm = managers.pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(module, managers.mam);
}

}  // namespace helpers
