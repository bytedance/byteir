get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_byteir_translation_library(ByteIRTargetPTX
  TranslateRegistration.cpp
  TranslateToPTX.cpp
  GPUKernelToPTX.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Target/PTX

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
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
  MLIRIRDL
  MLIRArithValueBoundsOpInterfaceImpl
)
