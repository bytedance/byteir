add_byteir_translation_library(ByteIRTargetCUDA
  TranslateRegistration.cpp
  TranslateToCUDA.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Target/CUDA

  DEPENDS
  ByteIRTargetCpp

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRIR
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRMemRefDialect
  MLIRGPUOps
  MLIRSupport
  # MLIRTranslation
  ByteIRTargetCpp
  )
