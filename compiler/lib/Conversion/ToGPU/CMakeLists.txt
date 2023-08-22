add_byteir_conversion_library(ByteIRToGPU
  CoalescedForToGPU.cpp
  FuncToGPU.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToGPU

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  ByteIRUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
  ByteIRConversionCommon
  ByteIRUtils
  )
