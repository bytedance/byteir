add_byteir_translation_library(ByteIRTargetLLVM
  TranslateRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Target/LLVM

  LINK_LIBS PUBLIC
  LLVMBitWriter
  MLIRArmNeonToLLVMIRTranslation
  MLIRArmSMEToLLVMIRTranslation
  MLIRArmSVEToLLVMIRTranslation
  MLIRAMXToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRX86VectorToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
)
