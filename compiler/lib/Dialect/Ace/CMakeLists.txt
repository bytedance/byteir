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
