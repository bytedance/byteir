add_onnx_frontend_library(OFCompiler
  OFCompilerOptions.cpp
  OFCompilerPipelines.cpp
  OFCompilerUtils.cpp

  LINK_LIBS PUBLIC
  OFConversion
  )