add_mlir_dialect_library(MLIRLaceDialect
  IR/LaceDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/mlir/Dialect/Lace

  DEPENDS
  MLIRAceOpsIncGen # ace types
  MLIRLaceOpsIncGen
  MLIRLaceOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRViewLikeInterface
  MLIRAceDialect
)