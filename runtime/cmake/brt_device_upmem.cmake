file(GLOB_RECURSE brt_device_upmem_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/pim/upmem/device/*.h"
  "${LIB_ROOT}/backends/pim/upmem/device/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${brt_device_upmem_srcs})

brt_add_object_library(brt_device_upmem ${brt_device_upmem_srcs})
target_link_libraries(brt_device_upmem LLVMOrcJIT LLVMX86CodeGen LLVMX86AsmParser)
target_include_directories(brt_device_upmem PUBLIC "/root/upmem_sdk/include")
target_link_directories(brt_device_upmem PUBLIC "/root/upmem_sdk/lib")

brt_add_include_to_target(brt_device_upmem brt_framework brt_common)
set_target_properties(brt_device_upmem PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/pim/upmem/device"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/pim/upmem")
