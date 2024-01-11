include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=torch_mlir.")

################################################################################
# Python sources
################################################################################

declare_mlir_python_sources(TorchFrontendPythonSources)
declare_mlir_python_sources(TorchFrontendPythonSources.TopLevel
  ADD_TO_PARENT TorchFrontendPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_frontend"
  SOURCES
    __init__.py
    _torch_frontend_registry.py
    ts_utils.py
    flash_attn_op.py
    fx_rewrite.py
    fx_tracer.py
    fx_utils.py
    convert_to_mhlo.py
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
  TorchMLIRPythonSources
  TorchMLIRPythonExtensions
  TorchFrontendMLIRPythonExtensions

  # Sources related to optional Torch extension dependent features. Typically
  # empty unless if project features are enabled.
  TorchMLIRPythonTorchExtensionsSources
)

add_mlir_python_common_capi_library(TorchFrontendMLIRAggregateCAPI
  INSTALL_COMPONENT TorchFrontendMLIRPythonModules
  INSTALL_DESTINATION python_packages/torch_frontend/torch_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${TORCH_FRONTEND_BINARY_DIR}/python_packages/torch_frontend/torch_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

target_link_options(TorchFrontendMLIRAggregateCAPI PRIVATE $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>)

add_mlir_python_modules(TorchFrontendMLIRPythonModules
  ROOT_PREFIX "${TORCH_FRONTEND_BINARY_DIR}/python_packages/torch_frontend/torch_mlir"
  INSTALL_PREFIX "python_packages/torch_frontend/torch_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS TorchFrontendMLIRAggregateCAPI
)

# change output dir of _jit_ir_importer
set_target_properties(TorchMLIRJITIRImporterPybind PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_FRONTEND_BINARY_DIR}/python_packages/torch_frontend/torch_mlir/_mlir_libs"
)

# manually link against TorchFrontendMLIRAggregateCAPI instead of TorchMLIRAggregateCAPI
function(patch_linked_library)
  foreach(target ${ARGN})
    get_target_property(LIBS ${target} LINK_LIBRARIES)
    if (TorchMLIRAggregateCAPI IN_LIST LIBS)
      list(REMOVE_ITEM LIBS TorchMLIRAggregateCAPI)
      list(APPEND LIBS TorchFrontendMLIRAggregateCAPI)
      set_target_properties(${target} PROPERTIES LINK_LIBRARIES "${LIBS}")
    endif()

    get_target_property(LIBS ${target} INTERFACE_LINK_LIBRARIES)
    if (TorchMLIRAggregateCAPI IN_LIST LIBS)
      list(REMOVE_ITEM LIBS TorchMLIRAggregateCAPI)
      list(APPEND LIBS TorchFrontendMLIRAggregateCAPI)
      set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES "${LIBS}")
    endif()
  endforeach()
endfunction()
patch_linked_library(TorchMLIRJITIRImporter)

add_mlir_python_modules(TorchFrontendPythonModules
  ROOT_PREFIX "${TORCH_FRONTEND_BINARY_DIR}/python_packages/torch_frontend"
  INSTALL_PREFIX "python_packages/torch_frontend"
  DECLARED_SOURCES TorchFrontendPythonSources
)

add_dependencies(TorchFrontendPythonModules TorchFrontendMLIRPythonModules)
add_dependencies(TorchFrontendPythonModules TorchMLIRJITIRImporterPybind)

################################################################################
# Build Python Wheel
################################################################################

add_custom_target(
  torch_frontend_python_pack
  COMMAND ${Python3_EXECUTABLE} "${TORCH_FRONTEND_SRC_ROOT}/torch-frontend/python/setup.py" "bdist_wheel"
  DEPENDS TorchFrontendPythonModules
)
