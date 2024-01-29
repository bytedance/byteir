add_mlir_library(TorchFrontendConversion
  ConvertTorchToCustomCall.cpp
  ConvertTorchToStablehloExt.cpp

  DEPENDS
  TorchFrontendConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialect
  MLIRTransforms
  ChloOps
  StablehloOps
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRTorchToStablehlo
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchUtils
)
