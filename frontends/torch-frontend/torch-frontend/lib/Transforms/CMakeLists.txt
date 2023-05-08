add_mlir_library(TorchFrontendTransforms
  RewriteToByteIREntryPoint.cpp
  RewriteCustomOp.cpp
  CanonicalizeExt.cpp
  UnpackPublicFunctionReturn.cpp

  DEPENDS
  TorchFrontendTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialect
  # MLIRTransforms
  # MLIRMhloUtils
  # ChloOps
  # ChloPasses
  # MhloDialect
  # MhloToLinalg
  # MhloTypeConversion
  # TorchMLIRConversionUtils
  # TorchMLIRTorchDialect
  # TorchMLIRTorchPasses
  # TorchMLIRTorchToMhlo
  # TorchMLIRTorchConversionPasses
  # TorchMLIRTorchUtils
)
