
cmake_minimum_required(VERSION 3.18)

project(brt_external_project C CXX)

option(BRT_INSTALL_PATH "The path to the installed BRT library")
get_filename_component(BRT_INSTALL_PATH ${BRT_INSTALL_PATH} ABSOLUTE)

list(APPEND CMAKE_MODULE_PATH "${BRT_INSTALL_PATH}/lib/cmake/brt")
include(brt-targets)

add_executable(main main.cpp)
target_link_libraries(main brt)
