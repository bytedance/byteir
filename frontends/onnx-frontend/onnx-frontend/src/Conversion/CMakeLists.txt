set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/OFPasses.td)
mlir_tablegen(OFPasses.inc -gen-pass-decls -name=OFConversion "-I${ONNX_FRONTEND_SRC_ROOT}")
add_public_tablegen_target(OFPassesIncGen)

set(LLVM_TARGET_DEFINITIONS OFRewriteToCustomCall.td)
mlir_tablegen(OFRewriteToCustomCall.inc -gen-rewriters "-I${ONNX_FRONTEND_SRC_ROOT}/third_party/onnx-mlir")
add_public_tablegen_target(OFRewriteToCustomCallIncGen)

set(LLVM_TARGET_DEFINITIONS OFRewriteCustomOnnxOps.td)
mlir_tablegen(OFRewriteCustomOnnxOps.inc -gen-rewriters "-I${ONNX_FRONTEND_SRC_ROOT}/third_party/onnx-mlir")
add_public_tablegen_target(OFRewriteCustomOnnxOpsIncGen)

add_onnx_frontend_library(OFConversion
  OFCanonicalizer.cpp
  OFCheckNonLowered.cpp
  OFModifyEntryPoint.cpp
  OFRewriteCustomOnnxOps.cpp
  OFRewriteToCustomCall.cpp

  DEPENDS
  OFPassesIncGen
  OFRewriteCustomOnnxOpsIncGen
  OFRewriteToCustomCallIncGen

  LINK_LIBS PUBLIC
  OMCompilerUtils  # from onnx-mlir
  OFSupport
  )