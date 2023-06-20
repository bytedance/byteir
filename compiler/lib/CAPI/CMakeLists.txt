add_mlir_public_c_api_library(ByteIRCAPI
  Dialects.cpp
  Passes.cpp
  Translation.cpp

  LINK_LIBS PUBLIC
  MLIRCatDialect
  MLIRCclTransformOps

  # conversion passes
  ByteIRGPUToNVVM
  ByteIRHloToCat
  ByteIRHloToLhlo
  ByteIRLmhloToLace
  ByteIRLmhloToMemref
  ByteIRToAce
  ByteIRToAIT
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

  # translate passes
  MLIRTranslateLib
  ByteIRTargetPTX
  ByteIRTargetCUDA
  ByteIRTargetCpp
)
