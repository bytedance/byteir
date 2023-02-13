add_mlir_dialect_library(MLIRByreDialect
  IR/ByreDialect.cpp

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

add_mlir_dialect_library(ByteIRByrePasses
  Transforms/Fold.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Byre

  DEPENDS
  ByteIRByrePassIncGen
  MLIRByreDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )
