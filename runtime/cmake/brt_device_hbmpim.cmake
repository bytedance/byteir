file(GLOB_RECURSE brt_device_hbmpim_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/pim/samsung/device/*.h"
  "${LIB_ROOT}/backends/pim/samsung/device/*.cc"
  "${LIB_ROOT}/backends/pim/samsung/device/*.cpp"
  "${BRT_ROOT}/external/pimlib/src/*.h"
  "${BRT_ROOT}/external/pimlib/src/*.c"
)

source_group(TREE ${REPO_ROOT} FILES ${brt_device_hbmpim_srcs})

brt_add_object_library(brt_device_hbmpim ${brt_device_hbmpim_srcs})
target_link_libraries(brt_device_hbmpim LLVMOrcJIT LLVMX86CodeGen LLVMX86AsmParser )
target_include_directories(brt_device_hbmpim PUBLIC "${REPO_ROOT}/external/pimlib/src")
target_include_directories(brt_device_hbmpim PUBLIC "${REPO_ROOT}/external/pimlib/lib")
brt_add_include_to_target(brt_device_hbmpim brt_framework brt_common )
set_target_properties(brt_device_hbmpim PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/pim/samsung/device"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/samsung/hbmpim")
