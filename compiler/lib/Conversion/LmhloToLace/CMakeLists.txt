add_byteir_conversion_library(ByteIRLmhloToLace
  LmhloToLace.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/LmhloToLace

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRUtils
  LmhloDialect
  MLIRLaceDialect

  LINK_LIBS PUBLIC
  ByteIRUtils
  LmhloDialect
  MLIRIR
  MLIRLaceDialect
)

