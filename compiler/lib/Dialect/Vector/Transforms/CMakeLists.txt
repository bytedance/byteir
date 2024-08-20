add_byteir_dialect_library(ByteIRVectorPasses
  CanonicalizeExt.cpp
  VectorLowerings.cpp
  VectorWarpDistribute.cpp
  MoveForallRegionIntoWarpOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Vector/Transforms

  DEPENDS
  ByteIRVectorPassIncGen
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRSupport

  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
)
