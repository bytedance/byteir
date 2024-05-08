#ifndef TORCH_FRONTEND_CONVERSION_PASSES_TD
#define TORCH_FRONTEND_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ConvertTorchToCustomCall
//===----------------------------------------------------------------------===//

def ConvertTorchToCustomCall : Pass<"convert-torch-to-custom-call", "func::FuncOp"> {
  let summary = "convert torch ops to byteir custom call.";
  let constructor = "mlir::createConvertTorchToCustomCall(/*validCustomCallOps=*/{})";
  let options = [
    ListOption<"validCustomCallOps", "valid-custom-call-ops", "std::string",
               "List of ops to be lowered to byteir custom call">,
  ];
}

//===----------------------------------------------------------------------===//
// ConvertTorchToStablehloExt
//===----------------------------------------------------------------------===//

def ConvertTorchToStablehloExt : Pass<"convert-torch-to-stablehlo-ext", "func::FuncOp"> {
  let summary = "convert torch ops to stablehlo extension.";
  let constructor = "mlir::createConvertTorchToStablehloExt()";
}

//===----------------------------------------------------------------------===//
// ConvertTorchToCcl
//===----------------------------------------------------------------------===//

def ConvertTorchToCcl : Pass<"convert-torch-to-ccl", "func::FuncOp"> {
  let summary = "convert torch communication ops to byteir ccl dialect.";
  let constructor = "mlir::createConvertTorchToCcl()";
}

#endif // TORCH_FRONTEND_CONVERSION_PASSES_TD