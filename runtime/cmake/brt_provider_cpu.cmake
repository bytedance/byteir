file(GLOB_RECURSE brt_cpu_provider_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/cpu/providers/default/*.h"
  "${LIB_ROOT}/backends/cpu/providers/default/*.h"
  "${LIB_ROOT}/backends/cpu/providers/default/*.cc"
)

set(brt_all_includes brt_common brt_framework)

source_group(TREE ${REPO_ROOT} FILES ${brt_cpu_provider_srcs})

brt_add_object_library(brt_provider_cpu ${brt_cpu_provider_srcs})
target_link_libraries(brt_provider_cpu brt_framework brt_common brt_ir brt_device_cpu)
brt_add_include_to_target(brt_provider_cpu ${brt_all_includes})
set_target_properties(brt_provider_cpu PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_provider_cpu PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/cpu/providers"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/cpu")
