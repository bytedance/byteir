set(LLVM_TARGET_DEFINITIONS FuseOpOnTorchPattern.td)
mlir_tablegen(FuseOpOnTorchPattern.inc -gen-rewriters)
add_public_tablegen_target(FuseOpOnTorchPatternIncGen)

add_mlir_library(TorchFrontendConversion
  ConvertTorchToCustomCall.cpp
  ConvertTorchToHBMCustomCall.cpp
  ConvertTorchToStablehloExt.cpp
  FuseOpOnTorch.cpp

  DEPENDS
  TorchFrontendConversionPassIncGen
  FuseOpOnTorchPatternIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialect
  MLIRTransforms
  MLIRMhloUtils
  ChloOps
  ChloPasses
  StablehloOps
  MhloDialect
  MhloToLinalg
  MhloTypeConversion
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRTorchToStablehlo
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchUtils
)
target_include_directories(TorchFrontendConversion PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
