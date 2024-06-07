get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_public_c_api_library(ByteIRCAPI
  Dialects.cpp
  Passes.cpp
  Translation.cpp
  PDLValue.cpp

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${extension_libs}
  MLIRCatDialect
  MLIRCclTransformOps

  # conversion passes
  ByteIRGPUToNVVM
  ByteIRHloToCat
  ByteIRToAce
  ByteIRToAIT
  ByteIRToByre
  ByteIRToGPU
  ByteIRToMhlo
  ByteIRToLinalg
  ByteIRToLLVM
  ByteIRToPTX
  ByteIRLcclToByre

  # transofrm passes
  ByteIRTransforms

  # dialect specific passes
  ByteIRAffinePasses
  ByteIRByrePasses
  ByteIRGPUPasses
  ByteIRLinalgPasses
  ByteIRMemRefPasses
  ByteIRMhloPasses
  ByteIRSCFPasses
  ByteIRTransformPasses

  # pipelines
  ByteIRGPUPipelines
  ByteIRHostPipelines
  ByteIRPipelines

  # translate passes
  MLIRTranslateLib
  ByteIRTargetPTX
  ByteIRTargetCUDA
  ByteIRTargetCpp
)
