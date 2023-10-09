file(GLOB_RECURSE brt_upmem_provider_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/pim/upmem/providers/*.h"
  "${LIB_ROOT}/backends/pim/upmem/providers/*.h"
  "${LIB_ROOT}/backends/pim/upmem/providers/*.cc"
)
list(FILTER brt_upmem_provider_srcs EXCLUDE REGEX
  ".*backends/pim/upmem/providers/gemv/dpu/gemv_dpu.cc"
)
set(brt_all_includes brt_common brt_framework)

source_group(TREE ${REPO_ROOT} FILES ${brt_upmem_provider_srcs})

brt_add_object_library(brt_provider_upmem ${brt_upmem_provider_srcs})
target_link_libraries(brt_provider_upmem brt_framework brt_common brt_ir brt_device_upmem "/root/upmem_sdk/lib/libdpu.so") 
target_include_directories(brt_provider_upmem PUBLIC "/root/upmem_sdk/include")
target_link_directories(brt_provider_upmem PUBLIC "/root/upmem_sdk/lib")
target_link_options(brt_provider_upmem PUBLIC "-dpu-pkg-config --cflags --libs dpu")


brt_add_include_to_target(brt_provider_upmem ${brt_all_includes})
set_target_properties(brt_provider_upmem PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_provider_upmem PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/pim/upmem/providers"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/pim/upmem")
