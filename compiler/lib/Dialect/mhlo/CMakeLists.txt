add_mlir_dialect_library(ByteIRMhloUtils
  Util/FusionUtil.cpp
  Util/ShapeInferUtil.cpp
  Util/Util.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Util

  DEPENDS
  MhloDialect
  ByteIRUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  ByteIRUtils
  )


add_mlir_dialect_library(ByteIRMhloDynamicShapeOpRegister
  DynamicShapeOpRegister/Concatenate.cpp
  DynamicShapeOpRegister/Convolution.cpp
  DynamicShapeOpRegister/DotLike.cpp 
  DynamicShapeOpRegister/DynamicBroadcastInDim.cpp
  DynamicShapeOpRegister/DynamicPartition.cpp
  DynamicShapeOpRegister/DynamicStitchLike.cpp
  DynamicShapeOpRegister/Einsum.cpp
  DynamicShapeOpRegister/NonZero.cpp
  DynamicShapeOpRegister/RealDynamicSlice.cpp
  DynamicShapeOpRegister/Reduce.cpp
  DynamicShapeOpRegister/ReshapeLike.cpp
  DynamicShapeOpRegister/Softmax.cpp
  DynamicShapeOpRegister/AddN.cpp
  DynamicShapeOpRegister/TorchIndexSelect.cpp
  DynamicShapeOpRegister/Where.cpp
  DynamicShapeOpRegister/ScatterNd.cpp
  DynamicShapeOpRegister/StridedSlice.cpp
  DynamicShapeOpRegister/BatchMatMul.cpp
  DynamicShapeOpRegister/Repeat.cpp
  DynamicShapeOpRegister/GeLU.cpp
  DynamicShapeOpRegister/LayerNorm.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/DynamicShapeOpRegister

  DEPENDS
  MhloDialect
  ByteIRMhloUtils
  ByteIRUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSupport
  ByteIRUtils
)

add_mlir_dialect_library(ByteIRMhloAnalysis
  Analysis/DimFromBroadcast.cpp
  Analysis/ShapeAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Analysis

  DEPENDS
  ByteIRAnalysis
  MhloDialect

  LINK_LIBS PUBLIC
  ByteIRAnalysis
  MLIRIR
  MLIRSupport
  MhloDialect
)

add_mlir_dialect_library(ByteIRMhloPasses
  Transforms/BoundedShapeInference.cpp
  Transforms/CanonicalizeExt.cpp
  Transforms/ClusterConstraint.cpp
  Transforms/ConvBackwardFusion.cpp
  Transforms/ConvForwardFusion.cpp
  Transforms/ConvertFuncToCustomCall.cpp
  Transforms/ConvertOpToCustomCall.cpp
  Transforms/ConvertInsertion.cpp
  Transforms/FuseTransposeIntoDotGeneral.cpp
  Transforms/DTypeConversion.cpp
  Transforms/DynamicShapeClustering.cpp
  Transforms/FuncArgRearrangement.cpp
  Transforms/FuseBMMDimension.cpp
  Transforms/FusionOutlining.cpp
  Transforms/MatmulLayoutTransform.cpp
  Transforms/GenericFusion.cpp
  Transforms/HloFolder.cpp
  Transforms/HloMoveDown.cpp
  Transforms/HloMoveUp.cpp
  Transforms/HloSimplify.cpp
  Transforms/InsertShapeConstraint.cpp
  Transforms/IOConvertFusion.cpp
  Transforms/LayoutTransformation.cpp
  Transforms/ReduceWindowFusion.cpp
  Transforms/RewriteWithConstraint.cpp
  Transforms/ShapeReification.cpp
  Transforms/StaticShapeInference.cpp
  Transforms/TrivialFusion.cpp
  Transforms/UnfuseBatchNorm.cpp


  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Transforms

  DEPENDS
  ByteIRMhloPassIncGen
  ByteIRAnalysis
  MhloDialect
  ByteIRMhloUtils
  ByteIRUtils
  ByteIRMhloAnalysis
  ByteIRMhloDynamicShapeOpRegister
  MLIRAceDialect
  MLIRCatDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRMhloUtils
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  ByteIRMhloUtils
  ByteIRUtils
  ByteIRMhloAnalysis
  ByteIRMhloDynamicShapeOpRegister
  MLIRAceDialect
  MLIRQuantDialect # FIXME: this should link at MhloDialect
  )
