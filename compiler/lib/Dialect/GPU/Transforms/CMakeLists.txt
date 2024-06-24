add_mlir_dialect_library(ByteIRGPUPasses
  LegalizeGPULaunch.cpp
  GPUBlockSwizzle.cpp
  GPUDistributeSharedMemoryCopy.cpp
  GPUDistributeToWarp.cpp
  GPUInputSharedMemorySwizzle.cpp
  GPUTensorCoreVectorization.cpp
  GPUPackSharedMemoryAlloc.cpp
  GPUVectorToGPU.cpp
  OptimizeVectorTransfer.cpp
  RemoveTrivialLoops.cpp
  ShmAllocaToWorkgroupArg.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/GPU
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/GPU/Transforms

  DEPENDS
  ByteIRGPUPassIncGen
  ByteIRUtils
  MLIRGPUDialect

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSupport
)
