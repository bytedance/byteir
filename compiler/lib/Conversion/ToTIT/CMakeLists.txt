add_byteir_conversion_library(ByteIRToTIT
  GenTITConfig.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToTIT

  DEPENDS
  ByteIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  ByteIRUtils
  )
