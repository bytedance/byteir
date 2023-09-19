add_byteir_conversion_library(ByteIRToAIT
  GenAITConfig.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToAIT

  DEPENDS
  ByteIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  ByteIRUtils
  )
