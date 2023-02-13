if (MSVC)
  # On platforms where CMAKE_CFG_INTDIR is not equal to '.' (e.g. Windows),
  # configure_lit_site_cfg will replace LLVM_BUILD_MODE in LLVM_TOOLS_DIR
  # with a pattern which will be replaced by --param=build_mode=XXX.
  # So, let's pass build_mode to llvm-lit, otherwise we would end up
  # with incomplete config.llvm_tools_dir
  set(LLVM_LIT_ARGS "-sv --param=build_mode=${CMAKE_BUILD_TYPE}")
else()
  set(LLVM_LIT_ARGS "-sv")
endif()

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(BYTEIR_MLIR_NUMERICAL_DEPENDS
    byteir-opt)

add_lit_testsuite(check-byteir-numerical
                  "Running the ByteIR Nuermical regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS
                  ${BYTEIR_MLIR_NUMERICAL_DEPENDS})
set_target_properties(check-byteir-numerical PROPERTIES FOLDER "Numerical")
