add_mlir_library(TorchFrontendConversion
  ConvertTorchToCcl.cpp
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
  MLIRCclDialect
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRTorchToStablehlo
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchUtils
)
