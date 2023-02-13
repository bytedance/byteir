add_byteir_conversion_library(ByteIRLmhloToMemref
  LmhloToMemref.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/LmhloToMemref

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRUtils
  LmhloDialect
  MLIRMemRefDialect

  LINK_LIBS PUBLIC
  ByteIRUtils
  LmhloDialect
  MLIRIR
  MLIRMemRefDialect
)

