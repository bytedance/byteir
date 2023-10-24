add_mlir_library(TorchFrontendTransforms
  RewriteToByteIREntryPoint.cpp
  RewriteCustomOp.cpp
  CanonicalizeExt.cpp
  UnpackPublicFunctionReturn.cpp
  EliminateUselessOp.cpp

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
  TorchFrontendUtils
)
