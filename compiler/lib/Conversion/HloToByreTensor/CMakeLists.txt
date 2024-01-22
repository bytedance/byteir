add_byteir_conversion_library(ByteIRHloToByreTensor
  HloToByreCustom.cpp
  HloToByreTensor.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/HloToByreTensor

  DEPENDS
  MLIRByreDialect
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  ByteIRMhloUtils

  LINK_LIBS PUBLIC
  MhloDialect
  MLIRAceDialect
  MLIRArithDialect
  MLIRByreDialect
  MLIRIR
  MLIRTensorDialect
  MLIRTransforms
  ByteIRConversionCommon
  ByteIRMhloUtils
)
