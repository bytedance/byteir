#ifndef TORCH_FRONTEND_TRANSFORMS_PASSES_TD
#define TORCH_FRONTEND_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def CanonicalizeExt : Pass<"canonicalize-ext"> {
  let summary = "Perform as an extension of upstream canonicalize pass.";
  let description = [{
    This pass performs as an extension of upstream canonicalize pass, with 
    additional canonicalization pattens, and dce especially for mhlo.custom_call. 
  }];
  let constructor = "mlir::createCanonicalizeExtPass()";
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "mlir::GreedySimplifyRegionLevel",
           /*default=*/"GreedySimplifyRegionLevel::Aggressive",
           "Perform control flow optimizations to the region tree",
             [{::llvm::cl::values(
               clEnumValN(mlir::GreedySimplifyRegionLevel::Disabled, "disabled",
                "Don't run any control-flow simplification."),
               clEnumValN(mlir::GreedySimplifyRegionLevel::Normal, "normal",
                "Perform simple control-flow simplifications (e.g. dead args elimination)."),
               clEnumValN(mlir::GreedySimplifyRegionLevel::Aggressive, "aggressive",
                "Perform aggressive control-flow simplification (e.g. block merging).")
              )}]>,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Max. iterations between applying patterns / simplifying regions">,
  ];
}

def RewriteCustomOp : Pass<"torch-rewrite-custom-op", "func::FuncOp"> {
  let summary = "Rewrite torch custom ops before refine types and shapes";
  let constructor = "mlir::createRewriteCustomOp()";
  let description = [{
    Rewrite custom torch operations to torch.custom_op.
  }];
}

def UnpackPublicFunctionReturn : Pass<"unpack-public-function-return", "func::FuncOp"> {
  let summary = "unpack tuple or list return of public function (means no callsite in current module).";
  let constructor = "mlir::createUnpackPublicFunctionReturnPass()";
}

def EliminateUselessOp : Pass<"eliminate-useless-op", "func::FuncOp"> {
  let summary = "eliminate useless op before entering converion pass, i.e., torch.profiler.";
  let constructor = "mlir::createEliminateUselessOpPass()";
}

//===----------------------------------------------------------------------===//
// FuseOpOnTorch
//===----------------------------------------------------------------------===//

def FuseOpOnTorch : Pass<"fuse-op-on-torch", "func::FuncOp"> {
  let summary = "fuse op on torch dialect";
  let constructor = "mlir::createFuseOpOnTorch()";
}

//===----------------------------------------------------------------------===//
// RewriteEntryFuncName
//===----------------------------------------------------------------------===//

def RewriteEntryFuncName : Pass<"rewrite-entry-func-name", "ModuleOp"> {
  let summary = "rewrite entry function's name";
  let constructor = "mlir::createRewriteEntryFuncNamePass()";
  let options = [
    Option<"targetName", "target-name", "std::string",
           /*default*/"",
           "Rewrite entry function's name with the target name.">
  ];
}

#endif // TORCH_FRONTEND_TRANSFORMS_PASSES_TD
