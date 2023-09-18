add_mlir_dialect_library(ByteIRMhloShardingInterfaceImpl
  ShardingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/IR

  DEPENDS
  ByteIRUtils
  MhloDialect
  MLIRShardingInterface
  

  LINK_LIBS PUBLIC
  ByteIRUtils
  MhloDialect
  MLIRDialectUtils
  MLIRIR
  MLIRShardingInterface
  MLIRSupport
  )
