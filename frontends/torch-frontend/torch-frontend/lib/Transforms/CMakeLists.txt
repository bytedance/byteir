add_mlir_library(TorchFrontendTransforms
  CanonicalizeExt.cpp
  EliminateUselessOp.cpp
  RewriteCustomOp.cpp
  RewriteEntryFuncName.cpp
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
  TorchFrontendUtils
)
