add_mlir_dialect_library(MLIRVectorExtTransformOps
  VectorExtTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Vector/TransformOps

  DEPENDS
  MLIRVectorExtTransformOpsIncGen
  ByteIRVectorPasses

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
  MLIRGPUDialect
  )
