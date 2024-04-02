add_byteir_conversion_library(ByteIRLcclToByre
  LcclToByre.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/LcclToByre

  DEPENDS
  MLIRByreDialect
  ByteIRConversionPassIncGen
  ByteIRConversionCommon

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRByreDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRTransforms
  ByteIRConversionCommon
)
