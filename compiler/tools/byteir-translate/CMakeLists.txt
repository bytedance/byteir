set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(byteir_translation_libs GLOBAL PROPERTY BYTEIR_TRANSLATION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
  ${dialect_libs}
  ${extension_libs}
  ${byteir_translation_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  # MLIRTranslation
  MLIRSupport
  MLIRExecutionEngine
  )

link_directories(${BYTEIR_BIN_LIB_DIR})
add_executable(byteir-translate byteir-translate.cpp)
add_dependencies(byteir-translate ${LIBS})


# Code from mlir-cuda-runner for retrieving LLVM target libs

# Manually expand the target library, since our MLIR libraries
# aren't plugged into the LLVM dependency tracking. If we don't
# do this then we can't insert the CodeGen library after ourselves
llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
# Prepend LLVM in front of every target, this is how the library
# are named with CMake
SET(targets_to_link)
FOREACH(t ${TARGET_LIBS})
  LIST(APPEND targets_to_link "LLVM${t}")
ENDFOREACH(t)

llvm_update_compile_flags(byteir-translate)
target_link_libraries(byteir-translate PRIVATE ${LIBS} ${targets_to_link})

mlir_check_all_link_libraries(byteir-translate)
