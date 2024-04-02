add_byteir_conversion_library(ByteIRFuncToByre
  FuncToByre.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/FuncToByre

  DEPENDS
  MLIRByreDialect
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  ByteIRMhloUtils

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRByreDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTransforms
  ByteIRConversionCommon
  ByteIRMhloUtils
)
