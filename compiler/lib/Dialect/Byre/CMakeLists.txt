add_mlir_dialect_library(MLIRByreDialect
  IR/ByreDialect.cpp
  IR/Common.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Byre

  DEPENDS
  MLIRByreOpsIncGen
  MLIRByreOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRLLVMDialect
  )

add_mlir_dialect_library(MLIRByreSerialization
  IR/Serialization/ByreSerialOps.cpp
  IR/Serialization/Bytecode.cpp
  IR/Serialization/Versioning.cpp
  IR/Serialization.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Byre
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Byre/Serialization

  DEPENDS
  ByteIRByreSerialIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRByreDialect
  )

add_mlir_dialect_library(ByteIRByrePasses
  Transforms/BufferizableOpInterfaceImpl.cpp
  Transforms/Serial.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Byre

  DEPENDS
  ByteIRByrePassIncGen
  MLIRByreDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  MLIRBufferizationDialect
  MLIRByreDialect
  MLIRByreSerialization
  MLIRMemRefDialect
  MLIRTensorDialect
  )