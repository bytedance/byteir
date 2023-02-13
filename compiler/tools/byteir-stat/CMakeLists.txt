get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(byteir_stat_libs GLOBAL PROPERTY BYTEIR_STAT_LIBS)

set(LIBS
  ${byteir_stat_libs}
  ${dialect_libs}
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
