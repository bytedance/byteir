add_subdirectory(Compiler)
add_subdirectory(Conversion)
add_subdirectory(Support)

add_onnx_frontend_executable(onnx-frontend
  onnx-frontend.cpp

  INSTALL

  LINK_LIBS PRIVATE
  OFCompiler
  OMCompilerOptions
  StablehloPortableApi
  )

add_onnx_frontend_executable(onnx-frontend-opt
  onnx-frontend-opt.cpp

  INSTALL

  LINK_LIBS PRIVATE
  OMCompilerOptions
  OMRegisterPasses
  OFCompiler
  MLIROptLib
  )
