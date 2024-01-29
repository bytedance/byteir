#ifndef TORCH_FRONTEND_CONVERSION_PASSES_TD
#define TORCH_FRONTEND_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ConvertTorchToCustomCall
//===----------------------------------------------------------------------===//

def ConvertTorchToCustomCall : Pass<"convert-torch-to-custom-call", "func::FuncOp"> {
  let summary = "convert torch ops to byteir custom call.";
  let constructor = "mlir::createConvertTorchToCustomCall()";
}

//===----------------------------------------------------------------------===//
// ConvertTorchToStablehloExt
//===----------------------------------------------------------------------===//

def ConvertTorchToStablehloExt : Pass<"convert-torch-to-stablehlo-ext", "func::FuncOp"> {
  let summary = "convert torch ops to stablehlo extension.";
  let constructor = "mlir::createConvertTorchToStablehloExt()";
}

#endif // TORCH_FRONTEND_CONVERSION_PASSES_TD