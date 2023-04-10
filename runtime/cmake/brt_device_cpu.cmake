file(GLOB_RECURSE brt_device_cpu_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/cpu/device/*.h"
  "${LIB_ROOT}/backends/cpu/device/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${brt_device_cpu_srcs})

brt_add_object_library(brt_device_cpu ${brt_device_cpu_srcs})
target_link_libraries(brt_device_cpu LLVMOrcJIT LLVMX86CodeGen LLVMX86AsmParser)
brt_add_include_to_target(brt_device_cpu brt_framework brt_common)
set_target_properties(brt_device_cpu PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/cpu/device"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/cpu")
