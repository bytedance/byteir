add_subdirectory(Common)
add_subdirectory(GPU)
add_subdirectory(Host)

add_byteir_pipeline_library(ByteIRPipelines
  AffineOpt.cpp
  AllOpt.cpp
  BufferizeOpt.cpp
  ByreHost.cpp
  ByreOpt.cpp
  ByreTensorOpt.cpp
  CatFusionOpt.cpp
  CatPreprocess.cpp
  HloGraphOpt.cpp
  HloFusionOpt.cpp
  LinalgMemrefOpt.cpp
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
  MLIRBufferTransforms

  LINK_LIBS PUBLIC
  ByteIRGPUPipelines
  ByteIRHloToCat
  ByteIRHostPipelines
  ByteIRPipelineCommon
  ByteIRUtils
  MLIRIR
  MLIRTransforms
  ByteIRAcePasses
  ByteIRByrePasses
  ByteIRFuncToByre
  ByteIRHloToByreTensor
  ByteIRMemrefToByre
  ByteIRMhloPasses
  ByteIRShapePasses
  ByteIRLinalgPasses
  ByteIRMemRefPasses
  ByteIRToAIT
  ByteIRToByre
  ByteIRToGPU
  ByteIRToLinalg
  MLIRBufferTransforms
)