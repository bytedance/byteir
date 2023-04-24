add_mlir_library(ByteIRUtils
  AffineUtils.cpp
  AttrUtils.cpp
  FuncUtils.cpp
  Hoist.cpp
  IRRewrite.cpp
  LoopUtils.cpp
  MemUtils.cpp
  OpInterfaceUtils.cpp
  PipelineUtils.cpp
  TypeUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Utils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
)