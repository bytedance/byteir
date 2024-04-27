set(LLVM_TARGET_DEFINITIONS FuseOpOnTorchPattern.td)
mlir_tablegen(FuseOpOnTorchPattern.inc -gen-rewriters)
add_public_tablegen_target(FuseOpOnTorchPatternIncGen)

add_mlir_library(TorchFrontendTransforms
  CanonicalizeExt.cpp
  EliminateUselessOp.cpp
  FuseOpOnTorch.cpp
  RewriteCustomOp.cpp
  RewriteEntryFuncName.cpp
  UnpackPublicFunctionReturn.cpp

  DEPENDS
  TorchFrontendTransformsPassIncGen
  TorchMLIRTorchDialect
  FuseOpOnTorchPatternIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialect
  TorchMLIRTorchDialect
  TorchFrontendUtils
)
target_include_directories(TorchFrontendTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
