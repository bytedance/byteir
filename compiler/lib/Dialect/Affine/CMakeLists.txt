add_byteir_dialect_library(ByteIRAffinePasses
  Transforms/AffineLoopFusionEx.cpp
  Transforms/InsertTrivialAffineLoop.cpp
  Transforms/RewriteAffineToMemref.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Affine
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Affine/Transforms

  DEPENDS
  ByteIRAffinePassIncGen
  ByteIRUtils
  MLIRAffineDialect
  MLIRMemRefDialect

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
)
