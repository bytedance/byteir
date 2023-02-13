include_directories("${CUTLASS_ROOT}/include")
file(GLOB_RECURSE brt_device_cuda_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/cuda/device/*.h"
  "${LIB_ROOT}/backends/cuda/device/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${brt_device_cuda_srcs})

brt_add_object_library(brt_device_cuda ${brt_device_cuda_srcs})
target_link_libraries(brt_device_cuda cuda cudart nvrtc)
brt_add_include_to_target(brt_device_cuda brt_framework brt_common)
set_target_properties(brt_device_cuda PROPERTIES FOLDER "Brt")


#add_dependencies(brt_framework ${brt_EXTERNAL_DEPENDENCIES})

# In order to find the shared provider libraries we need to add the origin to the rpath for all executables we build
# For the shared brt library, this is set in brt.cmake through CMAKE_SHARED_LINKER_FLAGS
# But our test files don't use the shared library so this must be set for them.
# For Win32 it generates an absolute path for shared providers based on the location of the executable/brt.dll
if (UNIX AND NOT APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
endif()

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/cuda/device"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/cuda")
