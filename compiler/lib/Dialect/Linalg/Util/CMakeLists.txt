add_mlir_dialect_library(MLIRLinalgExtUtils
  Util.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  )
