add_mlir_library(ByteIRTransforms
  AnchoredPipeline.cpp
  Bufferize.cpp
  TryCatchModulePipeline.cpp
  CanonicalizeExt.cpp
  CMAE.cpp
  CondCanonicalize.cpp
  CollectFunc.cpp
  FuncTag.cpp
  GenericDeviceConfig.cpp
  GraphClusteringByDevice.cpp
  InsertUniqueId.cpp
  LoopTag.cpp
  LoopUnroll.cpp
  MemoryPlanning.cpp
  RemoveFuncBody.cpp
  RewriteOpToStdCall.cpp
  SetArgShape.cpp
  SetSpace.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Transforms

  DEPENDS
  ByteIRAnalysis
  ByteIRMhloPasses
  ByteIRTensorPasses
  ByteIRTransformsPassIncGen
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRAnalysis
  ByteIRLinalgPasses
  ByteIRMhloPasses
  ByteIRTensorPasses
  ByteIRVectorPasses
  ByteIRUtils
  MLIRIR
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
)