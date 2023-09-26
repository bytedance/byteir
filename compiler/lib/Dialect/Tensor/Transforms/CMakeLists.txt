
add_mlir_dialect_library(ByteIRTensorPasses
  CanonicalizeExt.cpp
  TensorPadSpecialization.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Transforms

  DEPENDS
  ByteIRUtils
  ByteIRTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSCFDialect
  ByteIRUtils
  )