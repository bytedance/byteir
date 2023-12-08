set(brt_all_includes brt_common brt_framework brt_provider_cuda)

file(GLOB_RECURSE brt_nccl_provider_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/nccl/providers/*.h"
  "${LIB_ROOT}/backends/nccl/providers/*.h"
  "${LIB_ROOT}/backends/nccl/providers/*.cc"
)


list(APPEND brt_all_providers_srcs ${brt_nccl_provider_srcs})
list(APPEND brt_all_includes brt_device_nccl)

source_group(TREE ${REPO_ROOT} FILES ${brt_nccl_provider_srcs})

brt_add_object_library(brt_provider_nccl ${brt_nccl_provider_srcs})

target_link_libraries(brt_provider_nccl brt_provider_cuda)
target_link_libraries(brt_provider_nccl brt_device_nccl)
target_link_libraries(brt_provider_nccl ${NCCL_LIBRARIES})
brt_add_include_to_target(brt_provider_nccl ${brt_all_includes})
set_target_properties(brt_provider_nccl PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_provider_nccl PROPERTIES FOLDER "Brt")


install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/nccl/providers"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/nccl")
