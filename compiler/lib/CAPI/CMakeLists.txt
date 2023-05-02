add_mlir_public_c_api_library(ByteIRCAPI
  Dialects.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRCatDialect

  # conversion passes
  ByteIRGPUToNVVM
  ByteIRHloToCat
  ByteIRHloToLhlo
  ByteIRLmhloToLace
  ByteIRLmhloToMemref
  ByteIRToAce
  ByteIRToByre
  ByteIRToGPU
  ByteIRToMhlo
  ByteIRToLinalg
  ByteIRToLLVM
  ByteIRToPTX

  # transofrm passes
  ByteIRTransforms

  # dialect specific passes
  ByteIRAffinePasses
  ByteIRByrePasses
  ByteIRLinalgPasses
  ByteIRMemRefPasses
  ByteIRMhloPasses
  ByteIRSCFPasses
  ByteIRTransformPasses

  # pipelines
  ByteIRGPUPipelines
  ByteIRHostPipelines
  ByteIRPipelines
)
