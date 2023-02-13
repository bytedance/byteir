add_byteir_conversion_library(ByteIRToLLVM
  CollectFuncToLLVM.cpp
  GenLLVMConfig.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToLLVM

  DEPENDS
  ByteIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  ByteIRUtils
  )
