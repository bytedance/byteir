## Copyright (c) ByteDance Inc. All rights reserved.
## Licensed under the Apache License, Version 2.0

# Minimum CMake required
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)

project(brt-libs LANGUAGES CXX CUDA)


set(REPO_ROOT ${PROJECT_SOURCE_DIR})
message("REPO_ROOT = ${REPO_ROOT}")
set(BYTEIR_ROOT ${REPO_ROOT}/../..)
set(CUTLASS_ROOT ${BYTEIR_ROOT}/external/cutlass)
message("CUTLASS_ROOT = ${CUTLASS_ROOT}")

#set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)

add_subdirectory(flash_attn)
