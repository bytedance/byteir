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
    compile.py
    extra_shape_fn.py
    flash_attn_op.py
    fx_rewrite.py
    fx_tracer.py
    fx_utils.py
    ts_utils.py

    byteir_backend/__init__.py
    byteir_backend/compilation_cache.py
    byteir_backend/compiled_function.py
    byteir_backend/compiler.py
    byteir_backend/config.py
    byteir_backend/debug.py
    byteir_backend/inner_compile.py
    byteir_backend/utils.py
    byteir_backend/byteir_fusible_pattern.py
    byteir_backend/fx_match_utils.py
    byteir_backend/fx_utils.py
    byteir_backend/partitioners.py

    tools/compiler.py
    tools/gen_extra_library.py
    tools/extra_fn.mlir

    utils/__init__.py
    utils/jit_transforms.py
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_sources(TorchFrontendMLIRPythonExtensions)

declare_mlir_python_extension(TorchFrontendMLIRPythonExtensions.Main
  MODULE_NAME _torchFrontend
  ADD_TO_PARENT TorchFrontendMLIRPythonExtensions
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
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

# only link soversion TorchFrontendMLIRAggregateCAPI
get_target_property(LLVM_VERSION TorchFrontendMLIRAggregateCAPI VERSION)
set_target_properties(TorchFrontendMLIRAggregateCAPI PROPERTIES SOVERSION
                                                           ${LLVM_VERSION})

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
  COMMAND TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER=${TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER} ${Python3_EXECUTABLE} "${TORCH_FRONTEND_SRC_ROOT}/torch-frontend/python/setup.py" "bdist_wheel"
  DEPENDS TorchFrontendPythonModules
)
