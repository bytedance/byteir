add_byteir_dialect_library(ByteIRCclPasses
  CclMoveDown.cpp
  CclBufferizeOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Ccl
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Ccl/Transforms

  DEPENDS
  ByteIRCclPassIncGen

  LINK_LIBS PUBLIC
  MLIRLcclDialect
  MLIRIR
  MhloDialect
  MLIRSupport
)
