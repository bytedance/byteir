add_mlir_dialect_library(MLIRAceDialect
  IR/AceDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/mlir/Dialect/Ace

  DEPENDS
  MLIRAceOpsIncGen
  MLIRAceOpsAttrIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  )

add_mlir_dialect_library(ByteIRAcePasses
  Transforms/BufferizableOpInterfaceImpl.cpp
  Transforms/Bufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Ace

  DEPENDS
  ByteIRAcePassIncGen
  MLIRAceDialect
  MLIRLaceDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  MLIRAceDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRLaceDialect
  MLIRMemRefDialect
  )
  