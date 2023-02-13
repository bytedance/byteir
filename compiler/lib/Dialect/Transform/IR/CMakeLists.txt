add_mlir_dialect_library(MLIRTransformExtDialect
  TransformExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Transform/IR

  DEPENDS
  MLIRTransformExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRPDLDialect
  MLIRTransformDialect
  MLIRLinalgExtTransformOps
)
