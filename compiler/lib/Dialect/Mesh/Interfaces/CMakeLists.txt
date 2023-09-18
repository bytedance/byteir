add_mlir_library(MLIRShardingInterface
  InferDTensorInterface.cpp
  ShardingInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Mesh

  DEPENDS
  ByteIRUtils
  MLIRInferDTensorInterfaceIncGen
  MLIRShardingInterfaceIncGen
  MLIRMeshDialect

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRMeshDialect
  MLIRSupport
)
