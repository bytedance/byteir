set(ONNX_FRONTEND_TOOLS_DIR ${ONNX_FRONTEND_BIN_ROOT}/onnx-frontend/src)

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(ONNX_FRONTEND_TEST_DEPENDS
  onnx-frontend-opt
  FileCheck
  count
  not
  split-file
  )

add_lit_testsuite(check-of-lit
                  "Running the ONNX-FRONTEND lit tests"
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS
                  ${ONNX_FRONTEND_TEST_DEPENDS})
set_target_properties(check-of-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(ONNX_FRONTEND
                   ${CMAKE_CURRENT_SOURCE_DIR}
                   DEPENDS
                   ${ONNX_FRONTEND_TEST_DEPENDS})
