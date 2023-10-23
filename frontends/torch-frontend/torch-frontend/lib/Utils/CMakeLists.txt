add_mlir_library(TorchFrontendUtils
  ConvertOpFolder.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)