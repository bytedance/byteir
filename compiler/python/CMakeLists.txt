include(AddMLIRPython)

declare_mlir_python_sources(ByteIRPythonSources)
declare_mlir_python_sources(ByteIRPythonExtensions)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=byteir.")

set(BYTEIR_PYTHON_PACKAGES_DIR ${PROJECT_BINARY_DIR}/python_packages)

################################################################################
# Python sources
################################################################################

declare_mlir_python_sources(ByteIRPythonSources.Common
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/byteir"
  ADD_TO_PARENT ByteIRPythonSources
  SOURCES
    __init__.py
    _byteir_registry.py
    utils.py
    compile.py
    tools/compiler.py

    # involve byteir init hooks
    _mlir_libs/_site_initialize_0.py
)

declare_mlir_python_sources(ByteIRPythonSources.Dialects
  ADD_TO_PARENT ByteIRPythonSources
)

################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ByteIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/byteir"
  TD_FILE dialects/CatOps.td
  SOURCES_GLOB dialects/cat/*.py
  DIALECT_NAME cat
)

################################################################################
# Python extensions
################################################################################

declare_mlir_python_extension(ByteIRPythonExtensions.Main
  MODULE_NAME _byteir
  ADD_TO_PARENT ByteIRPythonExtensions
  SOURCES
    ByteIRModules.cpp
  EMBED_CAPI_LINK_LIBS
    ByteIRCAPI
    StablehloCAPI
    ChloCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# CAPI aggregation library
################################################################################

add_mlir_python_common_capi_library(ByteIRCAPIAggregation
  INSTALL_COMPONENT ByteIRPythonModules
  INSTALL_DESTINATION python_packages/byteir/byteir/_mlir_libs
  OUTPUT_DIRECTORY "${BYTEIR_PYTHON_PACKAGES_DIR}/byteir/byteir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRHLOPythonSources
    MLIRHLOPythonExtensions
    ByteIRPythonSources
    ByteIRPythonExtensions
)

target_link_options(ByteIRCAPIAggregation
  PRIVATE $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
)

################################################################################
# Python module
################################################################################

add_mlir_python_modules(ByteIRPythonModules
  ROOT_PREFIX "${BYTEIR_PYTHON_PACKAGES_DIR}/byteir/byteir"
  INSTALL_PREFIX "python_packages/byteir/byteir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRHLOPythonSources
    MLIRHLOPythonExtensions
    ByteIRPythonSources
    ByteIRPythonExtensions
  COMMON_CAPI_LINK_LIBS
    ByteIRCAPIAggregation
)

add_subdirectory(test)

################################################################################
# Build Python Wheel
################################################################################

add_custom_target(
  byteir-python-pack
  COMMAND "python3" "${BYTEIR_SRC_DIR}/python/setup.py" "bdist_wheel"
  DEPENDS ByteIRPythonModules
)
