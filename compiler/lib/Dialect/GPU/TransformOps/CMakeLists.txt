add_mlir_dialect_library(MLIRGPUExtTransformOps
  GPUExtTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/GPU/TransformOps


  DEPENDS
  MLIRGPUExtTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRGPUTransformOps
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRPDLDialect
  MLIRSCFDialect

  # ConversionPatterns
  MLIRNVGPUToNVVM
  MLIRGPUToNVVMTransforms
  )  
