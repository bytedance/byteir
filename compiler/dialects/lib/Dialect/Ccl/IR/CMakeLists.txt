add_byteir_dialect_library(MLIRCclDialect
  CclOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Ccl

  DEPENDS
  MLIRCclOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
