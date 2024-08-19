add_byteir_dialect_library(ByteIRLinalgPasses
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  CanonicalizeExt.cpp
  FuseElementwise.cpp
  HoistingExt.cpp
  LinalgCollapseLoops.cpp
  LinalgDataPlace.cpp
  LinalgExtToLoops.cpp
  LinalgGeneralizationExt.cpp
  LinalgPrefetch.cpp
  LinalgPromotion.cpp
  Tiling.cpp
  ScopeTiling.cpp
  Transforms.cpp
  TilingUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg/Transforms

  DEPENDS
  ByteIRLinalgPassIncGen
  ByteIRUtils
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRLinalgExt
  MLIRLinalgExtTransformOpsIncGen

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffineDialect
  MLIRDestinationStyleOpInterface
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRLinalgExt
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRSupport
)
