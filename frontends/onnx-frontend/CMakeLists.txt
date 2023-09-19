# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13.4)

project(onnx-frontend)
set(ONNX_FRONTEND_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(ONNX_FRONTEND_BIN_ROOT ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)

#-------------------------------------------------------------------------------
# Third Parties
#-------------------------------------------------------------------------------

option(ONNX_FRONTEND_ENABLE_WERROR "Enable warnings as errors." OFF)
option(ONNX_FRONTEND_SUPPRESS_THIRD_PARTY_WARNINGS "Suppress warning in third_party code." ON)

# Enable warnings as errors
# Leverage the imported LLVM_ENABLE_WERROR for compiler logic
set(LLVM_ENABLE_WERROR ${ONNX_MLIR_ENABLE_WERROR})

# Suppress warnings in third party code.
if (ONNX_FRONTEND_SUPPRESS_THIRD_PARTY_WARNINGS)
  set(LLVM_ENABLE_WERROR OFF)
  set(CMAKE_C_FLAGS_COPY ${CMAKE_C_FLAGS})
  set(CMAKE_CXX_FLAGS_COPY ${CMAKE_CXX_FLAGS})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
  set(ONNX_MLIR_SUPPRESS_THIRD_PARTY_WARNINGS ON CACHE BOOL "Suppress warning in third_party code." FORCE)
endif()

set(ONNX_MLIR_ENABLE_MHLO ON)
add_subdirectory(third_party/onnx-mlir EXCLUDE_FROM_ALL)
include_directories(${ONNX_FRONTEND_SRC_ROOT}/third_party/onnx-mlir/third_party/mlir-hlo)
include_directories(${ONNX_FRONTEND_BIN_ROOT}/third_party/onnx-mlir/third_party/mlir-hlo)

# Ensure warnings are reported for onnx-frontend code.
if (ONNX_FRONTEND_SUPPRESS_THIRD_PARTY_WARNINGS)
  set(LLVM_ENABLE_WERROR ${ONNX_MLIR_ENABLE_WERROR})
  set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS_COPY})
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_COPY})
endif()

#-------------------------------------------------------------------------------
# Compile ONNX Frontend
#-------------------------------------------------------------------------------

include(MLIR.cmake)

if (ONNX_MLIR_ENABLE_MHLO)
  llvm_update_compile_flags(stablehlo-opt)
  llvm_update_compile_flags(stablehlo-translate)
  llvm_update_compile_flags(mlir-hlo-opt)
endif()

if (ONNX_MLIR_ENABLE_MHLO)
  add_compile_definitions(ONNX_MLIR_ENABLE_MHLO)
endif()

add_compile_options(-Wno-comment -Wno-suggest-override)
add_subdirectory(onnx-frontend)
