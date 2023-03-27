cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_op)

find_package(Torch REQUIRED)
add_library(custom_op SHARED 
    dynamic_partition.cpp
    dynamic_stitch.cpp
)
target_compile_features(custom_op PRIVATE cxx_std_14)
target_link_libraries(custom_op PRIVATE "${TORCH_LIBRARIES}")
