file(GLOB_RECURSE brt_framework_srcs CONFIGURE_DEPENDS
    "${BRT_INCLUDE_DIR}/brt/core/context/*.h"
    "${BRT_INCLUDE_DIR}/brt/core/distributed/*.h"
    "${BRT_INCLUDE_DIR}/brt/core/framework/*.h"
    "${BRT_INCLUDE_DIR}/brt/core/session/*.h"
    "${LIB_ROOT}/core/context/*.h"
    "${LIB_ROOT}/core/context/*.cc"
    "${LIB_ROOT}/core/distributed/*.h"
    "${LIB_ROOT}/core/distributed/*.cc"
    "${LIB_ROOT}/core/framework/*.h"
    "${LIB_ROOT}/core/framework/*.cc"
    "${LIB_ROOT}/core/session/*.h"
    "${LIB_ROOT}/core/session/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${brt_framework_srcs})
brt_add_object_library(brt_framework ${brt_framework_srcs})

brt_add_include_to_target(brt_framework brt_common brt_ir)
set_target_properties(brt_framework PROPERTIES FOLDER "Brt")
target_link_libraries(brt_framework brt_ir brt_common)

# In order to find the shared provider libraries we need to add the origin to the rpath for all executables we build
# For the shared brt library, this is set in brt.cmake through CMAKE_SHARED_LINKER_FLAGS
# But our test files don't use the shared library so this must be set for them.
# For Win32 it generates an absolute path for shared providers based on the location of the executable/brt.dll
if (UNIX AND NOT APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
endif()

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/core/framework"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/core")
install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/core/context"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/core")
install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/core/session"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/core")

# install backends common headers
# TODO: move to core/framework?
file(GLOB backend_common_headers
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/*.h")
install(
  FILES ${backend_common_headers}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends"
)