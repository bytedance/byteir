add_mlir_library(TorchFrontendTransforms
  RewriteToByteIREntryPoint.cpp
  RewriteCustomOp.cpp
  CanonicalizeExt.cpp
  UnpackPublicFunctionReturn.cpp

  DEPENDS
  TorchFrontendTransformsPassIncGen
  TorchMLIRTorchDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialect
  TorchMLIRTorchDialect
  # MLIRTransforms
  # MLIRMhloUtils
  # ChloOps
  # ChloPasses
  # MhloDialect
  # MhloToLinalg
  # MhloTypeConversion
  # TorchMLIRConversionUtils
  # TorchMLIRTorchPasses
  # TorchMLIRTorchToMhlo
  # TorchMLIRTorchConversionPasses
  # TorchMLIRTorchUtils
)
