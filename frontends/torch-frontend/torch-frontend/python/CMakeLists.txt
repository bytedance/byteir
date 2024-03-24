include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=torch_frontend.")

################################################################################
# Python sources
################################################################################

declare_mlir_python_sources(TorchFrontendPythonSources)
declare_mlir_python_sources(TorchFrontendPythonSources.TopLevel
  ADD_TO_PARENT TorchFrontendPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_frontend"
  SOURCES
    _mlir_libs/_site_initialize_0.py
    __init__.py
    ts_utils.py
    flash_attn_op.py
    fx_rewrite.py
    fx_tracer.py
    fx_utils.py
    compile.py
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_sources(TorchFrontendMLIRPythonExtensions)

declare_mlir_python_extension(TorchFrontendMLIRPythonExtensions.Main
  MODULE_NAME _torchFrontend
  ADD_TO_PARENT TorchFrontendMLIRPythonExtensions
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  SOURCES
    TorchFrontendModule.cpp
  EMBED_CAPI_LINK_LIBS
    TorchFrontendCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate aggregate CAPI shared library and packages
################################################################################

set(_source_components
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
  TorchMLIRPythonExtensions
  StablehloPythonExtensions

  TorchFrontendPythonSources
  TorchFrontendMLIRPythonExtensions
)

add_mlir_python_common_capi_library(TorchFrontendMLIRAggregateCAPI
  INSTALL_COMPONENT TorchFrontendPythonModules
  INSTALL_DESTINATION python_packages/torch_frontend/_mlir_libs
  OUTPUT_DIRECTORY "${TORCH_FRONTEND_BINARY_DIR}/python_packages/torch_frontend/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES ${_source_components}
)

target_link_options(TorchFrontendMLIRAggregateCAPI PRIVATE $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>)

add_mlir_python_modules(TorchFrontendPythonModules
  ROOT_PREFIX "${TORCH_FRONTEND_BINARY_DIR}/python_packages/torch_frontend"
  INSTALL_PREFIX "python_packages/torch_frontend"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS TorchFrontendMLIRAggregateCAPI
)

################################################################################
# Build Python Wheel
################################################################################

add_custom_target(
  torch_frontend_python_pack
  COMMAND ${Python3_EXECUTABLE} "${TORCH_FRONTEND_SRC_ROOT}/torch-frontend/python/setup.py" "bdist_wheel"
  DEPENDS TorchFrontendPythonModules
)
