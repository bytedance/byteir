add_byteir_conversion_library(ByteIRMemrefToByre
  MemrefToByre.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/MemrefToByre

  DEPENDS
  MLIRByreDialect
  ByteIRConversionPassIncGen
  ByteIRConversionCommon

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRByreDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTransforms
  ByteIRConversionCommon
)
