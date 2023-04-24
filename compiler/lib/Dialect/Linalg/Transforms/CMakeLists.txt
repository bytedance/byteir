add_mlir_dialect_library(ByteIRLinalgPasses
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  CanonicalizeExt.cpp
  FuseElementwise.cpp
  LinalgCollapseLoops.cpp
  LinalgDataPlace.cpp
  LinalgExtToLoops.cpp
  LinalgGeneralizationExt.cpp
  LinalgPrefetch.cpp
  Tiling.cpp
  ScopeTiling.cpp
  Transforms.cpp
  TilingUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg/Transforms

  DEPENDS
  ByteIRLinalgPassIncGen
  MLIRLinalgExtTransformOpsIncGen
  ByteIRUtils
  MLIRDestinationStyleOpInterface
  MLIRLinalgExt
  MLIRLinalgDialect

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
