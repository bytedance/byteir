#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

# Must unset LLVM_DIR in cache. Otherwise, when MLIR_DIR changes LLVM_DIR
# won't change accordingly.
unset(LLVM_DIR CACHE)
if (NOT DEFINED MLIR_DIR OR "${MLIR_DIR}" STREQUAL "MLIR_DIR-NOTFOUND")
  message(WARNING "MLIR_DIR is not configured so we need build mlir at configure time."
    "Set the cmake option MLIR_DIR for prebuilt MLIR, e.g.,\n"
    "    cmake -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir ..\n"
    )

  # build MLIR at configure time
  function(build_external_mlir)
    set(LLVM_SRC_PATH "${TORCH_FRONTEND_SRC_ROOT}/third_party/torch-mlir/externals/llvm-project")
    set(LLVM_BUILD_PATH "${TORCH_FRONTEND_BIN_ROOT}/llvm_build")
    execute_process(COMMAND ${CMAKE_COMMAND}
      -S "${LLVM_SRC_PATH}/llvm"
      -B "${LLVM_BUILD_PATH}"
      -G "${CMAKE_GENERATOR}"
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_BUILD_TYPE=Release
      -DLLVM_ENABLE_PROJECTS=mlir
      -DLLVM_TARGETS_TO_BUILD=X86
      -DLLVM_ENABLE_ASSERTIONS=ON
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON

      RESULT_VARIABLE result
      WORKING_DIRECTORY ${LLVM_SRC_PATH}
    )

    if(result)
      message(FATAL_ERROR "CMake step for llvm failed: ${result}")
    endif()

    execute_process(
      COMMAND ${CMAKE_COMMAND} --build ${LLVM_BUILD_PATH}
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${LLVM_SRC_PATH}
    )

    if(result)
      message(FATAL_ERROR "Build step for llvm failed: ${result}")
    endif()

    set(MLIR_DIR "${LLVM_BUILD_PATH}/lib/cmake/mlir" CACHE PATH "" FORCE)
  endfunction(build_external_mlir)
  build_external_mlir()
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# TODO: optional python
set(MLIR_ENABLE_BINDINGS_PYTHON ON)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(BUILD_SHARED_LIBS ${LLVM_ENABLE_SHARED_LIBS} CACHE BOOL "" FORCE)
message(STATUS "TORCH-FRONTEND BUILD_SHARED_LIBS : " ${BUILD_SHARED_LIBS})


function(add_torch_frontend_executable name)
  cmake_parse_arguments(ARG
    "INSTALL"
    ""
    "DEPEND;INCLUDE_DIRS;LINK_LIBS;DEFINE"
    ${ARGN}
    )
  add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})

  target_include_directories(${name}
    PUBLIC
    ${TORCH_FRONTEND_SRC_ROOT}
    ${TORCH_FRONTEND_BIN_ROOT}
    )
endfunction(add_torch_frontend_executable)

function(add_byteir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY BYTEIR_DIALECT_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_byteir_dialect_library)
