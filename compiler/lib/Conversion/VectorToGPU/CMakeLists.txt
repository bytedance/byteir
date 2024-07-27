add_byteir_conversion_library(ByteIRVectorToGPU
  GPUVectorToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/VectorToGPU


  DEPENDS
  ByteIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRGPUDialect
  MLIRTransforms
  )
