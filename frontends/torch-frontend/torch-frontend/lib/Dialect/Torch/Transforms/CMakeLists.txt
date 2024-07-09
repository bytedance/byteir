set(LLVM_TARGET_DEFINITIONS FuseOpOnTorchPattern.td)
mlir_tablegen(FuseOpOnTorchPattern.inc -gen-rewriters)
add_public_tablegen_target(FuseOpOnTorchPatternIncGen)

add_mlir_library(TorchFrontendTorchTransforms
  DecomposeOnTorch.cpp
  FuseOpOnTorch.cpp

  DEPENDS
  TorchFrontendTorchTransformsPassIncGen
  FuseOpOnTorchPatternIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialect
  TorchMLIRTorchDialect
  TorchFrontendUtils
)
target_include_directories(TorchFrontendTorchTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
