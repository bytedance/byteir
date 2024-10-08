get_property(byteir_dialect_libs GLOBAL PROPERTY BYTEIR_DIALECT_LIBS)
get_property(byteir_stat_libs GLOBAL PROPERTY BYTEIR_STAT_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${byteir_stat_libs}
  ${byteir_dialect_libs}
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRDialect
  MLIRParser
  MLIRSupport
  MLIRIR
)

link_directories(${BYTEIR_BIN_LIB_DIR})
add_executable(byteir-stat byteir-stat.cpp)
add_dependencies(byteir-stat ${LIBS})

llvm_update_compile_flags(byteir-stat)
target_link_libraries(byteir-stat PRIVATE ${LIBS})

mlir_check_all_link_libraries(byteir-stat)
