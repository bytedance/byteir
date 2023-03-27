#ifndef TORCH_FRONTEND_TRANSFORMS_PASSES_TD
#define TORCH_FRONTEND_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// RewriteToByteIREntryPoint
//===----------------------------------------------------------------------===//

def RewriteToByteIREntryPoint : Pass<"rewrite-to-byteir-entry-point", "ModuleOp"> {
  let summary = "rewrite to byteir entry point.";
  let constructor = "mlir::createRewriteToByteIREntryPoint()";
}

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
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Seed the worklist in general top-down order">
  ];
}

def RewriteCustomOp : Pass<"torch-rewrite-custom-op", "func::FuncOp"> {
  let summary = "Rewrite torch custom ops before refine types and shapes";
  let constructor = "mlir::createRewriteCustomOp()";
  let description = [{
    Rewrite custom torch operations to torch.custom_op.
  }];
}

#endif // TORCH_FRONTEND_TRANSFORMS_PASSES_TD
