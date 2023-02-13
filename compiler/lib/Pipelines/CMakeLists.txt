add_subdirectory(Common)
add_subdirectory(GPU)
add_subdirectory(Host)

add_mlir_library(ByteIRPipelines
  AffineOpt.cpp
  AllOpt.cpp
  BufferizeOpt.cpp
  ByreHost.cpp
  ByreOpt.cpp
  HloOpt.cpp
  LinalgTensorOpt.cpp
  SCFOpt.cpp
  ShapeOpt.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines

  DEPENDS
  ByteIRGPUPipelines
  ByteIRHostPipelines
  ByteIRMhloPasses
  ByteIRPipelineCommon
  ByteIRShapePasses
  ByteIRLinalgPasses
  ByteIRMemRefPasses
  ByteIRTransforms
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRGPUPipelines
  ByteIRHostPipelines
  ByteIRPipelineCommon
  ByteIRUtils
  MLIRIR
  MLIRTransforms
  ByteIRAcePasses
  ByteIRByrePasses
  ByteIRHloToLhlo
  ByteIRLmhloToLace
  ByteIRMhloPasses
  ByteIRShapePasses
  ByteIRLinalgPasses
  ByteIRMemRefPasses
  ByteIRToByre
  ByteIRToGPU
  ByteIRHloToLinalg
)