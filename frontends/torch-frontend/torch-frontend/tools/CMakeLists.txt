set(TORCH_FORNTEND_LIBS
  TorchFrontendConversion
  TorchFrontendTransforms
  TorchFrontendPipelines
)

set(STABLEHLO_LIBS
  StablehloOps
  ChloOps
)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${TORCH_FORNTEND_LIBS}
  ${STABLEHLO_LIBS}
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
  ${extension_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRDialect
  MLIRTransforms
  MLIROptLib
)

add_executable(torch-frontend-opt torch-frontend-opt.cpp)
add_dependencies(torch-frontend-opt ${LIBS})

llvm_update_compile_flags(torch-frontend-opt)
target_link_libraries(torch-frontend-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(torch-frontend-opt)

