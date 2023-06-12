add_byteir_conversion_library(ByteIRToByre
  ToByre.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToByre

  DEPENDS
  MLIRAceDialect
  MLIRByreDialect
  MLIRLaceDialect
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  ByteIRMhloUtils

  LINK_LIBS PUBLIC
  MLIRAceDialect
  MLIRByreDialect
  MLIRIR
  MLIRLaceDialect
  MLIRMemRefDialect
  MLIRTransforms
  ByteIRConversionCommon
  ByteIRMhloUtils
  )
