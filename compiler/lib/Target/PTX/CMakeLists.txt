
add_byteir_translation_library(ByteIRTargetPTX
  TranslateRegistration.cpp
  TranslateToPTX.cpp
  GPUKernelToPTX.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Target/PTX

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRAMDGPUDialect
  MLIRIndexDialect
  MLIRMLProgramDialect
  MLIRPass
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTransforms
  MLIRTransformUtils
  MLIRTranslateLib
  MLIRGPUTransformOps
  MLIRAffineTransformOps
  MLIRLinalgTransformOps
  MLIRMemRefTransformOps
  MLIRVectorTransformOps
  MLIRSCFTransformOps
  MLIRBufferizationTransformOps
  MLIRShapeOpsTransforms
  )
