add_mlir_dialect_library(ByteIRGPUPasses
  GPUBlockSwizzle.cpp
  GPUDistributeToWarp.cpp
  GPUTensorCoreVectorization.cpp
  GPUPackSharedMemoryAlloc.cpp
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
