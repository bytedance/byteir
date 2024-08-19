add_byteir_dialect_library(ByteIRTransformPasses
  TransformDialectInterpreter.cpp
  TransformInsertion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Transform/Transforms

  DEPENDS
  ByteIRTransformPassIncGen
  MLIRLinalgExtTransformOps
  MLIRTransformExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRPDLDialect
  MLIRTransformDialect
  MLIRLinalgExtTransformOps
)
