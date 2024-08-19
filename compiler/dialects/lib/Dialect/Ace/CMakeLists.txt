add_byteir_dialect_library(MLIRAceDialect
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
