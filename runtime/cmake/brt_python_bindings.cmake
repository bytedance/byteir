set(PYTHON_BINDINGS_ROOT ${REPO_ROOT}/python)
set(PYTHON_BINDINGS_BIN_ROOT ${PROJECT_BINARY_DIR}/python)
file(GLOB_RECURSE SRCS ${PYTHON_BINDINGS_ROOT}/src/*.h ${PYTHON_BINDINGS_ROOT}/src/*.cc)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(${REPO_ROOT}/../external/pybind11 ${PROJECT_BINARY_DIR}/external/pybind11)

pybind11_add_module(_brt ${SRCS})
set_target_properties(_brt PROPERTIES 
    SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}
    LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BINDINGS_BIN_ROOT})

brt_set_compile_flags(_brt)

# TODO: linker scripts
target_link_libraries(_brt PRIVATE brt)

file(COPY ${PYTHON_BINDINGS_ROOT}/examples DESTINATION "${PYTHON_BINDINGS_BIN_ROOT}")
file(COPY ${REPO_ROOT}/test/test_files DESTINATION "${PYTHON_BINDINGS_BIN_ROOT}/examples" FILES_MATCHING PATTERN "*.mlir")
