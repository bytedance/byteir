add_mlir_library(ByteIRAnalysis
  DimFlag.cpp
  Liveness.cpp
  OpDependence.cpp
  ShapeAnalysis.cpp
  SideEffect.cpp
  SymbolicShape.cpp
  UseRange.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Analysis

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRBufferizationTransforms
  MLIRIR
  MLIRShapeDialect
  MLIRTensorDialect
)