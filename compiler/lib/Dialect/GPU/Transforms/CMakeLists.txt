add_mlir_dialect_library(ByteIRGPUPasses
  GPUTensorCoreVectorization.cpp
  ShmAllocaToWorkgroupArg.cpp
  GPUPackSharedMemoryAlloc.cpp
  GPUDistributeToWarp.cpp
  RemoveTrivialLoops.cpp
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
