message("build unittests")

set(TEST_SRC_DIR ${REPO_ROOT}/test)
set(disabled_warnings)

## general program entrypoint for C++ unit tests
set(brt_unittest_main_src "${TEST_SRC_DIR}/unittest_main/test_main.cc")

## test framework
set(brt_test_common_src_patterns
  "${TEST_SRC_DIR}/common/*.cc"
  "${TEST_SRC_DIR}/include/brt/test/common/*.h"
)

if(brt_USE_CUDA)
  list(APPEND brt_test_common_src_patterns "${TEST_SRC_DIR}/include/brt/test/common/cuda/*.h")
endif()

file(GLOB brt_test_common_src CONFIGURE_DEPENDS
  ${brt_test_common_src_patterns}
)

## test framework
set(brt_test_framework_src_patterns
  "${TEST_SRC_DIR}/framework/*.cc"
  "${TEST_SRC_DIR}/framework/*.h"
  "${TEST_SRC_DIR}/context/*.cc"
  "${TEST_SRC_DIR}/context/*.h"
  "${TEST_SRC_DIR}/platform/*.cc"
)

file(GLOB brt_test_framework_src CONFIGURE_DEPENDS
  ${brt_test_framework_src_patterns}
)

## test ir 
set(brt_test_ir_src_patterns
  "${TEST_SRC_DIR}/ir/*.cc"
  "${TEST_SRC_DIR}/ir/*.h"
)

file(GLOB brt_test_ir_src CONFIGURE_DEPENDS
  ${brt_test_ir_src_patterns}
)

## test session
set(brt_test_session_src_patterns
  "${TEST_SRC_DIR}/session/*.cc"
  "${TEST_SRC_DIR}/session/*.h"
)

file(GLOB brt_test_session_src CONFIGURE_DEPENDS
  ${brt_test_session_src_patterns}
)

## test providers
set(brt_test_providers_src "")

### test cpu providers
set(brt_test_cpu_provider_src_patterns
  "${TEST_SRC_DIR}/backends/cpu/providers/default/kernel/*.cc"
  "${TEST_SRC_DIR}/backends/cpu/providers/default/kernel/*.h"
  "${TEST_SRC_DIR}/backends/cpu/providers/default/e2e/*.cc"
  "${TEST_SRC_DIR}/backends/cpu/providers/default/e2e/*.h"
)
file(GLOB brt_test_cpu_provider_src CONFIGURE_DEPENDS
  ${brt_test_cpu_provider_src_patterns}
)
list(APPEND brt_test_providers_src ${brt_test_cpu_provider_src})

### test cuda providers
if(brt_USE_CUDA)
  set(brt_test_cuda_provider_src_patterns
  "${TEST_SRC_DIR}/backends/cuda/providers/default/*.cc"
  "${TEST_SRC_DIR}/backends/cuda/providers/default/*.h"
  "${TEST_SRC_DIR}/backends/cuda/providers/default/kernel/*.cc"
  "${TEST_SRC_DIR}/backends/cuda/providers/default/kernel/*.h"
  "${TEST_SRC_DIR}/backends/cuda/providers/default/e2e/*.cc"
  "${TEST_SRC_DIR}/backends/cuda/providers/default/e2e/*.h"
  )

  file(GLOB brt_test_cuda_provider_src CONFIGURE_DEPENDS
    ${brt_test_cuda_provider_src_patterns}
  )
  
  list(APPEND brt_test_providers_src ${brt_test_cuda_provider_src})
endif()

## test devices
set(brt_test_devices_src "")

### test cpu device
set(brt_test_cpu_device_src_patterns
  "${TEST_SRC_DIR}/backends/cpu/device/*.cc"
  "${TEST_SRC_DIR}/backends/cpu/device/*.h"
)

file(GLOB brt_test_cpu_device_src CONFIGURE_DEPENDS
  ${brt_test_cpu_device_src_patterns}
)

list(APPEND brt_test_devices_src ${brt_test_cpu_device_src})

### test cuda device
if(brt_USE_CUDA)
  ## CUDA device
  set(brt_test_cuda_device_src_patterns
    "${TEST_SRC_DIR}/backends/cuda/device/*.cu"
    "${TEST_SRC_DIR}/backends/cuda/device/*.cuh"
    "${TEST_SRC_DIR}/backends/cuda/device/*.cc"
    "${TEST_SRC_DIR}/backends/cuda/device/*.h"
  )
  
  file(GLOB brt_test_cuda_device_src CONFIGURE_DEPENDS
    ${brt_test_cuda_device_src_patterns}
  )

 list(APPEND brt_test_devices_src ${brt_test_cuda_device_src})
endif()

## include all src's  
set(all_test 
  ${brt_test_common_src}
  ${brt_test_devices_src}
  ${brt_test_framework_src}
  ${brt_test_ir_src}
  ${brt_test_session_src}
  ${brt_test_providers_src}
  ${brt_unittest_main_src}
)

set(all_link_libs
  gtest
  gtest_main
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND all_link_libs tinfo)
endif()

if(NOT WIN32)
  list(APPEND all_link_libs dl)
endif()

brt_add_executable(brt_test_all ${all_test})
target_link_libraries(brt_test_all brt.objs)

set(DYNAMIC_LIST ${TEST_SRC_DIR}/exported.ld)
target_include_directories(brt_test_all PUBLIC "${REPO_ROOT}/test/include")
target_link_libraries(brt_test_all ${all_link_libs} -Wl,--dynamic-list=${DYNAMIC_LIST})
set_target_properties(brt_test_all PROPERTIES LINK_DEPENDS ${DYNAMIC_LIST})

## compile external kernels
set(brt_external_kernels_patterns
  "${TEST_SRC_DIR}/external_kernels/cpu/*.cc"
  "${TEST_SRC_DIR}/external_kernels/cpu/*.h"
)

if (brt_USE_CUDA)
  list(APPEND brt_external_kernels_patterns
    "${TEST_SRC_DIR}/external_kernels/cuda/*.cc"
    "${TEST_SRC_DIR}/external_kernels/cuda/*.cu"
    "${TEST_SRC_DIR}/external_kernels/cuda/*.h")
endif()

file(GLOB brt_external_kernels_src CONFIGURE_DEPENDS
  ${brt_external_kernels_patterns}
)

brt_add_shared_library(external_kernels ${brt_external_kernels_src})
target_include_directories(external_kernels PRIVATE SYSTEM ${REPO_ROOT}/../external/half/include)
# we know that all brt::* symbols are defined in and exported from brt_test_all executable
target_link_libraries(external_kernels PRIVATE -Wl,--unresolved-symbols=ignore-in-object-files)
add_dependencies(brt_test_all external_kernels)

## copy test files
file(COPY ${TEST_SRC_DIR}/test_files DESTINATION ${CMAKE_BINARY_DIR}/test/)

