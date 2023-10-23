

set(brt_all_includes brt_common brt_framework)
include_directories("${UPMEM_ROOT}/include")

# add_
file(GLOB_RECURSE brt_upmem_provider_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/pim/upmem/providers/default/*.h"
  "${LIB_ROOT}/backends/pim/upmem/providers/default/*.h"
  "${LIB_ROOT}/backends/pim/upmem/providers/default/*.cc"
  "${LIB_ROOT}/backends/pim/upmem/providers/default/*.c"
  "${LIB_ROOT}/backends/pim/upmem/providers/default/*.cpp"
)


list(FILTER brt_upmem_provider_srcs EXCLUDE REGEX
  ".*backends/pim/upmem/providers/default/gemv/gemv_dpu.c"
)
list(FILTER brt_upmem_provider_srcs EXCLUDE REGEX
  ".*backends/pim/upmem/providers/default/softmax/softmax_dpu.c"
)

list(FILTER brt_upmem_provider_srcs EXCLUDE REGEX
  ".*backends/pim/upmem/providers/default/math/*_dpu.c"
)



# set(CMAKE_C_COMPILER dpu-clang)
list(APPEND brt_all_providers_srcs ${brt_upmem_provider_srcs})
list(APPEND brt_all_includes brt_device_upmem)


source_group(TREE ${REPO_ROOT} FILES ${brt_upmem_provider_srcs})

brt_add_object_library(brt_provider_upmem ${brt_upmem_provider_srcs})

# target_include_directories(brt_provider_upmem PUBLIC "/root/upmem_sdk/include/dpu/")
#  -std=c11 -O3 `dpu-pkg-config --cflags --libs dpu`
set(dpuoption -I/usr/include/dpu -ldpu)
target_compile_options(brt_provider_upmem PUBLIC ${dpuoption})
target_link_options(brt_provider_upmem PUBLIC ${dpuoption})
add_compile_options("-I/usr/include/dpu -ldpu")
target_include_directories(brt_provider_upmem PUBLIC "{BRT_INCLUDE_DIR}/brt/backends/pim/upmem/device")
# target_link_directories(brt_provider_upmem PUBLIC "/usr/include/dpu")
# target_link_libraries(brt_provider_upmem PUBLIC dpu)
target_link_libraries(brt_provider_upmem brt_framework brt_common brt_ir brt_device_upmem dpu ) 


# brt_add_shared_library(brt_provider_upmem "/root/upmem_sdk/lib/libdpu.so")
brt_add_include_to_target(brt_provider_upmem ${brt_all_includes})
set_target_properties(brt_provider_upmem PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_provider_upmem PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/pim/upmem/providers"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/pim/upmem")
