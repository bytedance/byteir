add_mlir_dialect_library(ByteIRSCFPasses
  InsertTrivialSCFLoop.cpp
  TilingInterfaceToSCFFor.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/SCF
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/SCF/Transforms

  DEPENDS
  ByteIRSCFPassIncGen
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSideEffectInterfaces
  MLIRSupport
  )
