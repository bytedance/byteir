add_mlir_library(ByteIRUtils
  AffineUtils.cpp
  AttrUtils.cpp
  FuncUtils.cpp
  GraphUtils.cpp
  Hoist.cpp
  IRRewrite.cpp
  LoopUtils.cpp
  MemUtils.cpp
  ModuleUtils.cpp
  OpInterfaceUtils.cpp
  OptionUtils.cpp
  PipelineUtils.cpp
  TileUtils.cpp
  TypeUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Utils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRCclDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFExtUtils
)