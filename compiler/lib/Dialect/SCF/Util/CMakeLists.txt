add_mlir_dialect_library(MLIRSCFExtUtils
  Util.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/SCF
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  )