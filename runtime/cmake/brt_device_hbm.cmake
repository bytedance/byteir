file(GLOB_RECURSE brt_device_hbm_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/pim/samsung/device/*.h"
  "${LIB_ROOT}/backends/pim/samsung/device/*.cc"
  "${LIB_ROOT}/backends/pim/samsung/device/*.cpp"
)

source_group(TREE ${REPO_ROOT} FILES ${brt_device_hbm_srcs})

brt_add_object_library(brt_device_hbm ${brt_device_hbm_srcs})
target_link_libraries(brt_device_hbm LLVMOrcJIT LLVMX86CodeGen LLVMX86AsmParser dramsim2)


target_include_directories(brt_device_hbm PUBLIC "/home/csgrad/amirnass/PIMLibrary/external_libs/include/dramsim2")
brt_add_include_to_target(brt_device_hbm brt_framework brt_common )
set_target_properties(brt_device_hbm PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/pim/hbm/device"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/pim/hbm")
