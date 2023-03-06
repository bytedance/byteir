configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TORCH_FRONTEND_TEST_DEPENDS
        FileCheck count not
        torch-frontend-opt
        )

add_lit_testsuite(check-torch-frontend-opt "Running the torch-frontend-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TORCH_FRONTEND_TEST_DEPENDS}
        )
set_target_properties(check-torch-frontend-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(TORCH_FRONTEND_TEST ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TORCH_FRONTEND_TEST_DEPENDS})