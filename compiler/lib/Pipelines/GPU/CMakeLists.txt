add_mlir_library(ByteIRGPUPipelines
  ElementwiseCodegen.cpp
  GemmCodegen.cpp
  GPUOpt.cpp
  LinalgMemrefGPU.cpp
  MappingForall.cpp
  NVVMCodegen.cpp
  ReductionCodegen.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines/GPU

  DEPENDS
  ByteIRPipelineCommon
  ByteIRTransforms
  ByteIRUtils
  MhloDialect
  MLIRBufferTransforms

  LINK_LIBS PUBLIC
  ByteIRGPUPasses
  ByteIRLinalgPasses
  ByteIRPipelineCommon
  ByteIRUtils
  ByteIRSCFPasses
  ByteIRToPTX
  MLIRIR
  MLIRTransforms
  MLIRLinalgExtTransformOps
)