add_byteir_dialect_library(ByteIRGPUPasses
  GPUBlockSwizzle.cpp
  GPUDistributeSharedMemoryCopy.cpp
  GPUDistributeToWarp.cpp
  GPUTensorCoreVectorization.cpp
  GPUPackSharedMemoryAlloc.cpp
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
