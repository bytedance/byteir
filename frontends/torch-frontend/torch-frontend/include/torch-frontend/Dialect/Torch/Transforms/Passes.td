#ifndef TORCH_FRONTEND_TORCH_TRANSFORMS_PASSES_TD
#define TORCH_FRONTEND_TORCH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FuseOpOnTorch
//===----------------------------------------------------------------------===//

def FuseOpOnTorch : Pass<"fuse-op-on-torch", "func::FuncOp"> {
  let summary = "fuse op on torch dialect";
  let constructor = "mlir::createFuseOpOnTorch()";
}

//===----------------------------------------------------------------------===//
// DecomposeOnTorch
//===----------------------------------------------------------------------===//

def DecomposeOnTorch : Pass<"decompose-on-torch", "func::FuncOp"> {
    let summary = "decompose op on torch dialect";
    let constructor = "mlir::createDecomposeOnTorch()";
}

#endif // TORCH_FRONTEND_TORCH_TRANSFORMS_PASSES_TD
