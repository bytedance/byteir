add_mlir_library(ByteIRGPUPipelines
  GPUOpt.cpp
  LinalgMemrefGPU.cpp
  NVVMCodegen.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines/GPU

  DEPENDS
  ByteIRPipelineCommon
  ByteIRTransforms
  ByteIRUtils
  MhloDialect
  MLIRBufferTransforms

  LINK_LIBS PUBLIC
  ByteIRLinalgPasses
  ByteIRPipelineCommon
  ByteIRUtils
  ByteIRSCFPasses
  ByteIRToPTX
  MLIRIR
  MLIRTransforms
)