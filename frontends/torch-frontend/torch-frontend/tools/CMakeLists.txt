set(TORCH_FORNTEND_LIBS
  TorchFrontendConversion
  TorchFrontendTransforms
  TorchFrontendPipelines
)

set(MHLO_LIBS
  MhloDialect
)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${TORCH_FORNTEND_LIBS}
  ${MHLO_LIBS}
  ${mlir_dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRDialect
  MLIRTransforms
  MLIROptLib
)

# FIXME: remove this when upstream fix
target_link_libraries(MhloDialect PUBLIC StablehloTypeInference StablehloAssemblyFormat)

add_executable(torch-frontend-opt torch-frontend-opt.cpp)
add_dependencies(torch-frontend-opt ${LIBS})

llvm_update_compile_flags(torch-frontend-opt)
target_link_libraries(torch-frontend-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(torch-frontend-opt)

