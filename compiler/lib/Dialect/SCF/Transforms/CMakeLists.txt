add_mlir_dialect_library(ByteIRSCFPasses
  ForallCollapsing.cpp
  ForallNormalize.cpp
  ForallTiling.cpp
  FuseNestedForall.cpp
  InsertTrivialSCFLoop.cpp
  TilingInterfaceToSCFFor.cpp
  RemoveSingleIterationLoop.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/SCF
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/SCF/Transforms

  DEPENDS
  ByteIRSCFPassIncGen
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSideEffectInterfaces
  MLIRSupport
)
