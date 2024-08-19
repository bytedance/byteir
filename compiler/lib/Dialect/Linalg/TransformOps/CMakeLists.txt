add_byteir_dialect_library(MLIRLinalgExtTransformOps
  LinalgExtTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg/TransformOps

  DEPENDS
  MLIRCclDialect
  MLIRLinalgExtTransformOpsIncGen
  ByteIRLinalgPasses

  LINK_LIBS PUBLIC
  ByteIRLinalgPasses
  MLIRAffineDialect
  MLIRArithDialect
  MLIRCclDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRParser
  MLIRPDLDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTensorTilingInterfaceImplExt
  MLIRTransformDialect
  MLIRVectorDialect
)
