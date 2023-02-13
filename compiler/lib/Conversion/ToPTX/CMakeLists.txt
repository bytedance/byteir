add_byteir_conversion_library(ByteIRToPTX
  CollectGPUKernel.cpp
  GenPTXConfig.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToPTX

  DEPENDS
  ByteIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRGPUOps
  MLIRTransforms
  )
