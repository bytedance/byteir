

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




# set(CMAKE_C_COMPILER dpu-clang)
list(APPEND brt_all_providers_srcs ${brt_hbm_provider_srcs})
list(APPEND brt_all_includes brt_device_hbm)


source_group(TREE ${REPO_ROOT} FILES ${brt_hbm_provider_srcs})

brt_add_object_library(brt_provider_hbm ${brt_hbm_provider_srcs})

target_include_directories(brt_provider_hbm PUBLIC "/home/csgrad/amirnass/PIMLibrary/external_libs/include/dramsim2")
#  -std=c11 -O3 `dpu-pkg-config --cflags --libs dpu`
# set(dpuoption -I/usr/include/dpu -ldpu)
# target_compile_options(brt_provider_hbm PUBLIC ${dpuoption})
# target_link_options(brt_provider_hbm PUBLIC ${dpuoption})
# add_compile_options("-I/usr/include/dpu -ldpu")
target_include_directories(brt_provider_hbm PUBLIC "{BRT_INCLUDE_DIR}/brt/backends/pim/samsung/device")
# target_link_directories(brt_provider_hbm PUBLIC "/usr/include/dpu")
# target_link_libraries(brt_provider_hbm PUBLIC dpu)
target_link_libraries(brt_provider_hbm brt_framework brt_common brt_ir brt_device_hbm dramsim2 ) 


# brt_add_shared_library(brt_provider_hbm "/root/upmem_sdk/lib/libdpu.so")
brt_add_include_to_target(brt_provider_hbm ${brt_all_includes})
set_target_properties(brt_provider_hbm PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_provider_hbm PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/pim/samsung/providers"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/pim/upmem")
