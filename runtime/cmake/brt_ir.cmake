set(MLIR_DIR "${LLVM_INSTALL_PATH}/lib/cmake/mlir")
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

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
  MLIRSideEffectInterfaces
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  #MLIRDataLayoutInterfaces
  MLIRInferTypeOpInterface
  MLIRInferIntRangeInterface
  MLIRShapedOpInterfaces
)

# involve byre
set(BYRE_SRC_DIR ${REPO_ROOT}/../compiler)
set(BYRE_BUILD_DIR ${CMAKE_BINARY_DIR}/byteir)
set(BYTEIR_INCLUDE_TESTS OFF)
set(BYTEIR_BUILD_EMBEDDED ON)
add_subdirectory(${BYRE_SRC_DIR}/cmake ${BYRE_BUILD_DIR} EXCLUDE_FROM_ALL)
include_directories(${BYRE_SRC_DIR}/include)
include_directories(${BYRE_BUILD_DIR}/include)

set(BYRE_ALL_EXTERNAL_LIBS
  -Wl,--start-group
  ${LLVM_REQUIRE_LIBS}
  ${MLIR_REQUIRE_LIBS}
  MLIRAceDialect # TODO: remove it when we have byre.string
  MLIRByreDialect
  -Wl,--end-group
)

set(BYRE_ENABLE_ZLIB ${LLVM_ENABLE_ZLIB})
set(BYRE_ENABLE_TERMINFO ${LLVM_ENABLE_TERMINFO})

if(BYRE_ENABLE_ZLIB)
  list(APPEND BYRE_ALL_EXTERNAL_LIBS z)
endif()

if(BYRE_ENABLE_TERMINFO)
  list(APPEND BYRE_ALL_EXTERNAL_LIBS tinfo)
endif()

file(GLOB_RECURSE brt_ir_srcs CONFIGURE_DEPENDS
    "${BRT_INCLUDE_DIR}/brt/core/ir/*.h"
    "${LIB_ROOT}/core/ir/*.h"
    "${LIB_ROOT}/core/ir/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${brt_ir_srcs})
brt_add_object_library(brt_ir ${brt_ir_srcs})
target_link_libraries(brt_ir ${BYRE_ALL_EXTERNAL_LIBS} brt_common)

brt_add_include_to_target(brt_ir brt_common)
set_target_properties(brt_ir PROPERTIES FOLDER "Brt")

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
