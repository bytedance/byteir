add_mlir_library(ByteIRPipelineCommon
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines/Common

  DEPENDS
  ByteIRTransforms
  ByteIRUtils
  MhloDialect
  MLIRBufferTransforms

  LINK_LIBS PUBLIC
  ByteIRLinalgPasses
  ByteIRUtils
  ByteIRSCFPasses
  MLIRIR
  MLIRTransforms
)