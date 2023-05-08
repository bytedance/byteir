
add_mlir_dialect_library(ByteIRTensorPasses
  CanonicalizeExt.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Transforms

  DEPENDS
  ByteIRUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  ByteIRUtils
  )