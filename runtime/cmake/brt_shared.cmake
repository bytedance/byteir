brt_add_shared_library(brt)
set(VERSION_SCRIPT ${REPO_ROOT}/version.ld)
target_link_libraries(brt
  PUBLIC $<BUILD_INTERFACE:brt.objs>
  PRIVATE $<INSTALL_INTERFACE:brt.objs>
  PRIVATE -Wl,--no-undefined -Wl,--version-script=${VERSION_SCRIPT}
)
set_target_properties(brt PROPERTIES LINK_DEPENDS ${VERSION_SCRIPT})

install(
  TARGETS brt
  EXPORT brt-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(
  EXPORT brt-targets
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/brt")
