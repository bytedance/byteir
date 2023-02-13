add_mlir_dialect_library(MLIRLinalgExtTransformOps
  LinalgExtTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg/TransformOps

  DEPENDS
  MLIRLinalgExtTransformOpsIncGen
  ByteIRLinalgPasses

  LINK_LIBS PUBLIC
  ByteIRLinalgPasses
  MLIRAffineDialect
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRParser
  MLIRPDLDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRVectorDialect
  )
