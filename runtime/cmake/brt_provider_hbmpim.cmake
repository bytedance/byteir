

set(brt_all_includes brt_common brt_framework)

# include_directories("${UPMEM_ROOT}/include")

# add_
file(GLOB_RECURSE brt_hbm_provider_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/pim/samsung/providers/default/*.h"
  "${LIB_ROOT}/backends/pim/samsung/providers/default/*.h"
  "${LIB_ROOT}/backends/pim/samsung/providers/default/*.cc"
  "${LIB_ROOT}/backends/pim/samsung/providers/default/*.c"
  "${LIB_ROOT}/backends/pim/samsung/providers/default/*.cpp"
)

# file(GLOB_RECURSE pimlibsrcs CONFIGURE_DEPENDS
# "${BRT_ROOT}/external/pimlib/src/*.h"
# "${BRT_ROOT}/external/pimlib/src/*.cpp"

# )

# set(CMAKE_C_COMPILER dpu-clang)
list(APPEND brt_all_providers_srcs ${brt_hbm_provider_srcs})
list(APPEND brt_all_includes brt_device_hbmpim)

# add_library(pimsim SHARED ${pimlibsrcs} )
source_group(TREE ${REPO_ROOT} FILES ${brt_hbm_provider_srcs})

brt_add_object_library(brt_provider_hbmpim ${brt_hbm_provider_srcs})

target_include_directories(brt_provider_hbmpim PUBLIC "${REPO_ROOT}/external/pimlib/src")
target_include_directories(brt_provider_hbmpim PUBLIC "${REPO_ROOT}/external/pimlib/lib")

# target_link_directories(brt_provider_hbmpim PUBLIC "${REPO_ROOT}/external/pimlib/lib")÷
file(GLOB_RECURSE pimlibsrcs "${REPO_ROOT}/external/pimlib/src/*.cpp"
  "${REPO_ROOT}/external/pimlib/src/tests/*.cpp"
  "${REPO_ROOT}/external/pimlib/src/tests/*.h"
  "${REPO_ROOT}/external/pimlib/src/*.h"
)

# message(STATUS "pimlibsrcs: ${pimlibsrcs}")

# add_executable(brt pimlibsrcs)
# add_library(pimlib SHARED ${pimlibsrcs})
# include("${REPO_ROOT}/external/pimlib/lib/half.h")
include_directories("${REPO_ROOT}/external/pimlib/lib" "${REPO_ROOT}/external/pimlib/src" "${REPO_ROOT}/external/pimlib/src/tests")
add_library(pimlib ${pimlibsrcs})

# target_link_libraries(brt_provider_hbmpim  )
# target_sources(brt_provider_hbmpim PUBLIC ${pimlibsrcs})
target_include_directories(brt_provider_hbmpim PUBLIC "{BRT_INCLUDE_DIR}/brt/backends/pim/samsung/device")

# find(glog)
# find_package(googletest REQUIRED)
message("Use gtest from submodule")

# gtest and gmock
# set_msvc_c_cpp_compiler_warning_level(4)
add_subdirectory(${REPO_ROOT}/../external/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)

# set_msvc_c_cpp_compiler_warning_level(3)
set_target_properties(gmock PROPERTIES FOLDER "External/GTest")

set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_link_libraries(brt_provider_hbmpim brt_framework brt_common brt_ir brt_device_hbmpim pimlib)

# brt_add_shared_library(brt_provider_hbmpim "/root/upmem_sdk/lib/libdpu.so")
brt_add_include_to_target(brt_provider_hbmpim ${brt_all_includes})
set_target_properties(brt_provider_hbmpim PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_provider_hbmpim PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/pim/samsung/providers"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/pim/samsung")
