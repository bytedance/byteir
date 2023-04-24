add_mlir_dialect_library(MLIRCclTransformOps
  CclTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Ccl/TransformOps

  DEPENDS
  MLIRCclDialect
  MLIRCclTransformOpsIncGen
  

  LINK_LIBS PUBLIC
  MLIRCclDialect
  MLIRIR
  MLIRParser
  MLIRPDLDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  )
