if(BYTEIR_INCLUDE_TESTS)
  set(test_libs
    ByteIRTestAnalysis
    ByteIRTestInterface
    ByteIRTestTransformation
    ByteIRTestUtils
  )
endif()

get_property(byteir_conversion_libs GLOBAL PROPERTY BYTEIR_CONVERSION_LIBS)
get_property(byteir_dialect_libs GLOBAL PROPERTY BYTEIR_DIALECT_LIBS)
get_property(byteir_pipeline_libs GLOBAL PROPERTY BYTEIR_PIPELINE_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
)

set(MHLO_LIBS
  MhloRegisterDialects
  AllMhloPasses
  DeallocationPasses
  MLIRBufferTransforms
  MLIRHLOGPUTransforms
  StablehloRegister
  StablehloOps
  ChloOps
)

set(LIBS
  ${dialect_libs}
  ${extension_libs}
  ${conversion_libs}
  ${byteir_dialect_libs}
  ${byteir_pipeline_libs}
  ${byteir_conversion_libs}
  ByteIRTransforms
  ${MHLO_LIBS}
  ${test_libs}
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
)

link_directories(${BYTEIR_BIN_LIB_DIR})
add_executable(byteir-opt byteir-opt.cpp)
add_dependencies(byteir-opt ${LIBS})

llvm_update_compile_flags(byteir-opt)
target_link_libraries(byteir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(byteir-opt)
