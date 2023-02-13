add_mlir_library(ByteIRHostPipelines
  HostOpt.cpp
  ToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines/Host

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  ByteIRToLLVM
  ByteIRPipelineCommon
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