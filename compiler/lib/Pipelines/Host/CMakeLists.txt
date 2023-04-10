add_mlir_library(ByteIRHostPipelines
  Codegen.cpp
  HostOpt.cpp
  ToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines/Host

  DEPENDS
  MLIRTransformExtDialect
  MLIRLinalgExtTransformOps

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  ByteIRToLLVM
  ByteIRPipelineCommon
  ByteIRTransformPasses
  ByteIRVectorPasses
  MLIRTransformExtDialect
  MLIRLinalgExtTransformOps
  MLIRArithTransforms
  MLIRBufferizationTransforms
  MLIRFuncToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRTensorTransforms
  MLIRTransforms
)