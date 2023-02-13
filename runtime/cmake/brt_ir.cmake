file(GLOB_RECURSE brt_ir_srcs CONFIGURE_DEPENDS
    "${BRT_INCLUDE_DIR}/brt/core/ir/*.h"
    "${LIB_ROOT}/core/ir/*.h"
    "${LIB_ROOT}/core/ir/*.cc"
)

set(LLVM_REQUIRE_LIBS 
  #LLVMBinaryFormat
  #LLVMBitstreamReader
  #LLVMCore
  #LLVMRemarks
  LLVMSupport
  LLVMDemangle
)

set(MLIR_REQUIRE_LIBS 
  MLIRArithDialect
  MLIRArithUtils
  MLIRDialect
  MLIRDialectUtils
  MLIRParser
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRControlFlowDialect
  MLIRCopyOpInterface
  MLIRSupport
  MLIRIR
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRViewLikeInterface
  MLIRMemRefUtils
  MLIRSideEffectInterfaces
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  #MLIRDataLayoutInterfaces
  MLIRInferTypeOpInterface
  MLIRInferIntRangeInterface
  MLIRShapedOpInterfaces
)

set(BYRE_ALL_EXTERNAL_LIBS
  -Wl,--start-group
  ${LLVM_REQUIRE_LIBS}
  ${MLIR_REQUIRE_LIBS}
  MLIRAceDialect # TODO: remove it when we have byre.string
  MLIRByreDialect
  -Wl,--end-group
)

# TODO(?): If we use imported LLVM/MLIR cmake targets directly instead of
# static libraries(e.g. cmake target MLIRIR rather than libMLIRIR.a)
# via exported LLVMConfig.cmake, we don't need following code which would
# handle linking dependencies manually anymore
if ("${LLVM_SRC_PATH}" STREQUAL "")
  set(BYRE_ENABLE_ZLIB ON)
  set(BYRE_ENABLE_TERMINFO ON)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_INSTALL_PATH}/lib/cmake/llvm")
  include(LLVMConfig)
  include(HandleLLVMOptions)
  set(BYRE_ENABLE_ZLIB ${LLVM_ENABLE_ZLIB})
  set(BYRE_ENABLE_TERMINFO ${LLVM_ENABLE_TERMINFO})
else()
  # FIXME(?): cannot read LLVMConfig due to target conflict
  # message(WARNING "cannot read LLVMConfig, linking zlib and terminfo by default.")
  set(BYRE_ENABLE_TERMINFO ON)
endif()

if(BYRE_ENABLE_ZLIB)
  list(APPEND BYRE_ALL_EXTERNAL_LIBS z)
endif()

if(BYRE_ENABLE_TERMINFO)
  list(APPEND BYRE_ALL_EXTERNAL_LIBS tinfo)
endif()

message("BYRE_BIN_LIB_DIR = ${BYRE_BIN_LIB_DIR}")
message("LLVM_BIN_LIB_DIR = ${LLVM_BIN_LIB_DIR}")
message("BYRE_INCLUDE_DIRS = ${BYRE_INCLUDE_DIRS}")
message("LLVM_INCLUDE_DIRS = ${LLVM_INCLUDE_DIRS}")

# suppres warnings from mlir headers
include_directories(
  SYSTEM
  ${BYRE_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

source_group(TREE ${REPO_ROOT} FILES ${brt_ir_srcs})
brt_add_object_library(brt_ir ${brt_ir_srcs})
target_link_libraries(brt_ir ${BYRE_ALL_EXTERNAL_LIBS} brt_common)
target_link_directories(brt_ir PUBLIC
  $<BUILD_INTERFACE:${BYRE_BIN_LIB_DIR}>
  $<BUILD_INTERFACE:${LLVM_BIN_LIB_DIR}>)

brt_add_include_to_target(brt_ir brt_common)
set_target_properties(brt_ir PROPERTIES FOLDER "Brt")


#add_dependencies(brt_framework ${brt_EXTERNAL_DEPENDENCIES})

# In order to find the shared provider libraries we need to add the origin to the rpath for all executables we build
# For the shared brt library, this is set in brt.cmake through CMAKE_SHARED_LINKER_FLAGS
# But our test files don't use the shared library so this must be set for them.
# For Win32 it generates an absolute path for shared providers based on the location of the executable/brt.dll
if (UNIX AND NOT APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
endif()

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/core/ir"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/core")
