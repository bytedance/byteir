add_mlir_dialect_library(MLIRTensorTilingInterfaceImplExt
  TilingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Tensor

  DEPENDS
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRAffineDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
)
