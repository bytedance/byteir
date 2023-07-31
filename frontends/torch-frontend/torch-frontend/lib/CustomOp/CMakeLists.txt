cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_op)

# copied from torch-mlir
function(TorchFrontendProbeForPyTorchInstall)
  if(Torch_DIR)
    message(STATUS "Using cached Torch_DIR = ${Torch_DIR}")
  else()
    message(STATUS "Checking for PyTorch using ${Python3_EXECUTABLE} ...")
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
      -c "import os;import torch;print(torch.utils.cmake_prefix_path, end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE PYTORCH_STATUS
      OUTPUT_VARIABLE PYTORCH_PACKAGE_DIR)
    if(NOT PYTORCH_STATUS EQUAL "0")
      message(STATUS "Unable to 'import torch' with ${Python3_EXECUTABLE} (fallback to explicit config)")
      return()
    endif()
    message(STATUS "Found PyTorch installation at ${PYTORCH_PACKAGE_DIR}")

    set(Torch_DIR "${PYTORCH_PACKAGE_DIR}/Torch" CACHE STRING
        "Torch configure directory" FORCE)
  endif()
endfunction()

TorchFrontendProbeForPyTorchInstall()
find_package(Torch REQUIRED)
add_library(custom_op SHARED 
    dynamic_partition.cpp
    dynamic_stitch.cpp
    dynamic_mask_stitch.cpp
)

# add_library(custom_torch IMPORTED INTERFACE GLOBAL)
# set_property(
#   TARGET custom_torch
#   APPEND
#   PROPERTY INTERFACE_LINK_LIBRARIES "${Torch_DIR}/../../../lib/libtorch.so")
# set_property(
#   TARGET custom_torch
#   APPEND
#   PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Torch_DIR}/../../../include")

# add_library(custom_torch SHARED IMPORTED GLOBAL)
# set_property(
#   TARGET custom_torch
#   PROPERTY IMPORTED_LOCATION
#   "${Torch_DIR}/../../../lib/libtorch.so"
# )
# get_property(custom_torch_include_dir TARGET torch PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
# target_include_directories(custom_op PRIVATE ${custom_torch_include_dir})

target_compile_features(custom_op PRIVATE cxx_std_17)
# target_link_libraries(custom_op PRIVATE custom_torch)
target_link_libraries(custom_op PRIVATE "${TORCH_LIBRARIES}")
