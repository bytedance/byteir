add_byteir_conversion_library(ByteIRConversionCommon
  FunctionSupport.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/Common

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRTransforms
  )
