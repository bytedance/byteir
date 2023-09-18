add_mlir_library(ByteIRMeshPasses
  ShardingPartition.cpp
  ShardingPropagation.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Mesh
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Mesh/Transforms

  DEPENDS
  ByteIRMeshPassesIncGen
  ByteIRMhloShardingInterfaceImpl
  ByteIRMhloUtils
  ByteIRUtils
  MhloDialect
  MLIRShardingInterface

  LINK_LIBS PUBLIC
  ByteIRMhloShardingInterfaceImpl
  ByteIRMhloUtils
  ByteIRUtils
  MhloDialect
  MLIRIR
  MLIRMeshDialect
  MLIRShardingInterface
  MLIRSupport
  )
