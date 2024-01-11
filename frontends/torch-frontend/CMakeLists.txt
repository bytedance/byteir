cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(torch-frontend)
set(TORCH_FRONTEND_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(TORCH_FRONTEND_BIN_ROOT ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)

include(MLIR.cmake)

#-------------------------------------------------------------------------------
# Third Parties
#-------------------------------------------------------------------------------

set(TORCH_MLIR_BUILD_EMBEDDED ON)
set(TORCH_MLIR_ENABLE_STABLEHLO ON)
add_definitions(-DTORCH_MLIR_ENABLE_STABLEHLO)
set(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER ON)
set(TORCH_MLIR_ENABLE_LTC OFF)
add_subdirectory(third_party/torch-mlir ${TORCH_FRONTEND_BIN_ROOT}/torch-mlir EXCLUDE_FROM_ALL)
# add torch-mlir header files
include_directories(${TORCH_FRONTEND_SRC_ROOT}/third_party/torch-mlir/include)
include_directories(${TORCH_FRONTEND_BIN_ROOT}/torch-mlir/include)
# add stablehlo header files
include_directories(${TORCH_FRONTEND_SRC_ROOT}/third_party/torch-mlir/externals/stablehlo)
include_directories(${TORCH_FRONTEND_BIN_ROOT}/torch-mlir/stablehlo)

#-------------------------------------------------------------------------------
# Compile Torch Frontend
#-------------------------------------------------------------------------------

set(TORCH_FRONTEND_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/torch-frontend/include)
include_directories(${CMAKE_BINARY_DIR}/torch-frontend/include)

add_compile_options(-Wno-comment -Wno-suggest-override)
add_subdirectory(torch-frontend)


add_custom_target(${PROJECT_NAME} ALL)
add_dependencies(${PROJECT_NAME} check-torch-frontend-opt torch_frontend_python_pack)

# for minimal test
add_custom_target(torch-frontend-minimal-test DEPENDS TorchFrontendPythonModules check-torch-frontend-opt)
