add_byteir_dialect_library(MLIRGPUExtTransformOps
  GPUExtTransformOps.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/GPU/TransformOps


  DEPENDS
  MLIRGPUExtTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRPDLDialect
  MLIRSCFDialect
)  
