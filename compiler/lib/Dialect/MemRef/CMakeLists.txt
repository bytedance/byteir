add_mlir_dialect_library(ByteIRMemRefPasses
  Transforms/ApplyMemRefAffineLayout.cpp
  Transforms/ExtractAddressComputation.cpp
  Transforms/MultiBufferExt.cpp
  Transforms/RemoveCopy.cpp
  Transforms/SimplifyLinearizedIndex.cpp
  Transforms/SimplifyView.cpp
  Utils/Layout.cpp
  Utils/MemEffect.cpp
  Utils/Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/MemRef
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/MemRef/Transforms
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/MemRef/Utils

  DEPENDS
  ByteIRMemRefPassIncGen
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffineDialect
  MLIRAnalysis
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSupport
  )
