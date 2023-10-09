## Copyright (c) ByteDance Inc. All rights reserved.
## Licensed under the Apache License, Version 2.0

# Minimum CMake required
cmake_minimum_required(VERSION 3.18)

# Project
project(brt C CXX)
# Needed for Java
set (CMAKE_C_STANDARD 99)

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(CMakeDependentOption)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# NOTE: POSITION INDEPENDENT CODE hurts performance, and it only makes sense on POSIX systems
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Options
option(brt_BUILD_SHARED_LIB "Build using shared libraries" ON)
option(brt_USE_CUDA "Build with CUDA support" OFF)
option(brt_BUILD_UNIT_TESTS "Build Runtime unit tests" ON)
option(brt_CROSS_COMPILING "Cross compiling for another platform" OFF)

option(brt_ENABLE_FLASH_ATTENSION "Build with flash attension" OFF)
if (brt_ENABLE_FLASH_ATTENSION AND NOT brt_USE_CUDA)
  message(FATAL_ERROR "config brt_ENABLE_FLASH_ATTENSION=ON must with brt_USE_CUDA=ON")
endif()

# Parameters for LLVM
set(LLVM_INSTALL_PATH "" CACHE STRING "The path to the installed LLVM library")

# Optimizations Related
option(brt_ENABLE_LTO "Enable LTO" OFF)

#A special option just for debugging and sanitize check. Please do not enable in option in retail builds.
#The option has no effect on Windows.
option(brt_USE_VALGRIND "Build with valgrind" OFF)

option(brt_ENABLE_ASAN "Enable address sanitizer" OFF)

option(brt_ENABLE_PYTHON_BINDINGS "Enable python bindings" OFF)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  option(brt_LLJIT_DEBUG "Enable LLJIT debug" ON)
else()
  option(brt_LLJIT_DEBUG "Enable LLJIT debug" OFF)
endif()

if (brt_LLJIT_DEBUG)
  add_definitions(-DBRT_LLJIT_DEBUG)
endif()

## configure files for tests
macro(configure_files srcDir destDir)
    message(STATUS "Configuring directory ${destDir}")
    make_directory(${destDir})

    file(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*)
    foreach(templateFile ${templateFiles})
        set(srcTemplatePath ${srcDir}/${templateFile})
        if(NOT IS_DIRECTORY ${srcTemplatePath})
            message(STATUS "Configuring file ${templateFile}")
            configure_file(
                    ${srcTemplatePath}
                    ${destDir}/${templateFile}
                    @ONLY)
        endif(NOT IS_DIRECTORY ${srcTemplatePath})
    endforeach(templateFile)
endmacro(configure_files)


# msvc support
function(set_msvc_c_cpp_compiler_warning_level warning_level)
  if (NOT "${warning_level}" MATCHES "^[0-4]$")
    message(FATAL_ERROR "Expected warning_level value of 0-4, got '${warning_level}'.")
  endif()

  if(MSVC)
    set(warning_flag "/W${warning_level}")
    get_property(opts DIRECTORY PROPERTY COMPILE_OPTIONS)
    # only match the generator expression added by this function
    list(FILTER opts
         EXCLUDE REGEX "^\\$<\\$<OR:\\$<COMPILE_LANGUAGE:C>,\\$<COMPILE_LANGUAGE:CXX>>:/W[0-4]>$")
    list(APPEND opts "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${warning_flag}>")
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "${opts}")
  endif()
endfunction()

# set default MSVC warning level to 3 for external dependencies
set_msvc_c_cpp_compiler_warning_level(3)

# Valgrind
if(brt_USE_VALGRIND AND NOT WIN32)
  add_definitions(-DRE2_ON_VALGRIND=1)
endif()

if(brt_ENABLE_ASAN)
  if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "ASAN should be with Debug build.")
  endif()
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

# Enable space optimization for gcc/clang
# Cannot use "-ffunction-sections -fdata-sections" if we enable bitcode (iOS)
if (NOT MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -ffunction-sections -fdata-sections")
  string(APPEND CMAKE_C_FLAGS " -ffunction-sections -fdata-sections")
endif()

if(brt_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_enabled OUTPUT ipo_output)
    if(NOT ipo_enabled)
      message(WARNING "IPO is not supported by this compiler")
      set(brt_ENABLE_LTO OFF)
    else()
      set (CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()

if(brt_DISABLE_RTTI)
  if(MSVC)
    # Disable RTTI and turn usage of dynamic_cast and typeid into errors
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GR->" "$<$<COMPILE_LANGUAGE:CXX>:/we4541>")
  else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>")
  endif()
else()
  #MSVC RTTI flag /GR is not added to CMAKE_CXX_FLAGS by default. But, anyway VC++2019 treats "/GR" default on.
  #So we don't need the following three lines. But it's better to make it more explicit.
  if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GR>")
  endif()
endif()


set(REPO_ROOT ${PROJECT_SOURCE_DIR}/..)
message("REPO_ROOT = ${REPO_ROOT}")
set(LIB_ROOT ${REPO_ROOT}/lib)
message("LIB_ROOT = ${LIB_ROOT}")
set(CUTLASS_ROOT ${REPO_ROOT}/../external/cutlass)
message("CUTLASS_ROOT = ${CUTLASS_ROOT}")

file (STRINGS "${REPO_ROOT}/VERSION_NUMBER" BRT_VERSION)

# TODO: check if x86
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mf16c")

if(brt_CROSS_COMPILING)
  set(CMAKE_CROSSCOMPILING ON)
  check_cxx_compiler_flag(-Wno-error HAS_NOERROR)
  if(HAS_NOERROR)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=attributes")
    string(APPEND CMAKE_C_FLAGS " -Wno-error=attributes")
  endif()
endif()

# Mark symbols to be invisible, for macOS/iOS target only
# Due to many dependencies have different symbol visibility settings, set global compile flags here.
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS")
  foreach(flags CMAKE_CXX_FLAGS CMAKE_OBJC_FLAGS CMAKE_OBJCXX_FLAGS)
    string(APPEND ${flags} " -fvisibility=hidden -fvisibility-inlines-hidden")
  endforeach()
endif()

macro(check_nvcc_compiler_flag _FLAG _RESULT)
    execute_process(COMMAND ${CUDA_HOME}/bin/nvcc "${_FLAG}" RESULT_VARIABLE NVCC_OUT ERROR_VARIABLE NVCC_ERROR)
    message("NVCC_ERROR = ${NVCC_ERROR}")
    message("NVCC_OUT = ${NVCC_OUT}")
    if("${NVCC_OUT}" MATCHES "0")
        set(${_RESULT} 1)
    else()
        set(${_RESULT} 0)
    endif()
endmacro()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    #For Mac compliance
    message("Adding flags for Mac builds")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "iOSCross")
    #For ios compliance
    message("Adding flags for ios builds")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target arm64-apple-darwin-macho")
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "arm")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target armv7a-apple-darwin-macho")
    endif()
endif()

if(CMAKE_CROSSCOMPILING)
  message("Doing crosscompiling")
endif()

get_filename_component(BRT_ROOT "${BRT_ROOT}" ABSOLUTE)
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)
set(BRT_INCLUDE_DIR ${REPO_ROOT}/include)

# linking 
set(brt_LINK_DIRS )

# link CUDADNN
if(brt_USE_CUDA)
  message(STATUS "brt_USE_CUDA On")
  if (brt_ENABLE_FLASH_ATTENSION)
    message(STATUS "brt_ENABLE_FLASH_ATTENSION On")
  endif()
  find_package(CUDAToolkit REQUIRED)
	include_directories("${CUDAToolkit_INCLUDE_DIRS}")
  message("CUDAToolkit Include Dirs = ${CUDAToolkit_INCLUDE_DIRS}")
    if (WIN32)
      list(APPEND brt_LINK_DIRS ${CUDA_HOME}/lib/x64 ${CUDA_HOME}/x64/lib64)
    else()
      list(APPEND brt_LINK_DIRS ${CUDA_HOME}/lib64)
    endif()
endif()

FILE(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}  BRT_BINARY_DIR)
FILE(TO_NATIVE_PATH ${PROJECT_SOURCE_DIR}  BRT_SOURCE_DIR)


#names in this var must match the directory names under brt/core/providers
#BRT_PROVIDER_NAMES is the list of providers that needs to export additional symbols in the global namespace. 
#For example CUDA EP exports "OrtSessionOptionsAppendExecutionProvider_CUDA", whƒich is a global function.
#However, all these things are legacy and deprecated and should be replaced with functions in brt_c_api.h.
set(BRT_PROVIDER_NAMES cpu)
list(APPEND BRT_PROVIDER_NAMES upmem)
set(BRT_PROVIDER_FLAGS)
set(BRT_PROVIDER_CMAKE_FLAGS)

if (brt_USE_CUDA)
    list(APPEND BRT_PROVIDER_FLAGS -DBRT_USE_CUDA=1)
    list(APPEND BRT_PROVIDER_CMAKE_FLAGS -Dbrt_USE_CUDA=1)
    list(APPEND BRT_PROVIDER_NAMES cuda)
    if (brt_ENABLE_FLASH_ATTENSION)
      list(APPEND BRT_PROVIDER_FLAGS -DBRT_ENABLE_FLASH_ATTENSION=1)
      list(APPEND BRT_PROVIDER_CMAKE_FLAGS -Dbrt_ENABLE_FLASH_ATTENSION=1)
    endif()
endif()

# utility functions 
function(brt_set_compile_flags target_name)
    if (MSVC)
      target_compile_definitions(${target_name} PUBLIC -DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
      if(brt_ENABLE_MEMLEAK_CHECKER)
        target_compile_definitions(${target_name} PUBLIC -DBRT_ENABLE_MEMLEAK_CHECK)
      endif()
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /sdl>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/sdl>")
      set_target_properties(${target_name}
                      PROPERTIES VS_GLOBAL_CAExcludePath "${BRT_BINARY_DIR};${BRT_SOURCE_DIR}")
      if (brt_ENABLE_STATIC_ANALYSIS)
        target_compile_options(${target_name} PRIVATE  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/analyze:stacksize 131072>")
        target_compile_options(${target_name} PRIVATE  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/analyze:external->")
      endif()
    else()
      # Enable warning
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wall>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall>")
      target_compile_options(${target_name} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wextra>")
      if(brt_DEV_MODE)
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Werror>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror>")
      endif()

      target_compile_definitions(${target_name} PUBLIC -DNSYNC_ATOMIC_CPP11)
      if(APPLE AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "iOSCross" AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "iOS")
        target_compile_definitions(${target_name} PUBLIC -Doptional_CONFIG_SELECT_OPTIONAL=optional_OPTIONAL_NONSTD)
      endif()
      target_include_directories(${target_name} PRIVATE "${REPO_DIR}/../external/nsync/public")
    endif()
    foreach(BRT_FLAG ${BRT_PROVIDER_FLAGS})
      target_compile_definitions(${target_name} PRIVATE ${BRT_FLAG})
    endforeach()
    if(HAS_DEPRECATED_COPY)
      #too many such errors in eigen
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-deprecated-copy>" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>")
    endif()
    if(brt_USE_CUDA)
      if((NVCC_HAS_STRICT_ALIASING AND "${target_name}" MATCHES "cuda") OR (HAS_STRICT_ALIASING AND NOT "${target_name}" MATCHES "cuda"))
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-strict-aliasing>")
      endif()
    endif()
    foreach(BRT_FLAG ${BRT_WARNING_FLAGS})
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options ${BRT_FLAG}>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${BRT_FLAG}>")
    endforeach()
endfunction()

function(brt_set_source_file_properties target_name)
  get_target_property(srcs ${target_name} SOURCES)

  # enable ARC for Objective-C/C++
  set(objective_c_cc_srcs ${srcs})
  list(FILTER objective_c_cc_srcs INCLUDE REGEX "\\.mm?$")
  set_property(SOURCE ${objective_c_cc_srcs} APPEND PROPERTY COMPILE_OPTIONS "-fobjc-arc")
endfunction()

function(brt_configure_target target_name)
  target_link_directories(${target_name} PRIVATE ${brt_LINK_DIRS})
  brt_set_compile_flags(${target_name})
  brt_set_source_file_properties(${target_name})
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${BRT_ROOT})
  if(brt_ENABLE_LTO)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
  endif()
endfunction()

function(brt_add_shared_library target_name)
  add_library(${target_name} SHARED ${ARGN})
  brt_configure_target(${target_name})
endfunction()

function(brt_add_static_library target_name)
  add_library(${target_name} ${ARGN})
  brt_configure_target(${target_name})
endfunction()

function(brt_add_object_library target_name)
  add_library(${target_name} OBJECT ${ARGN})
  brt_configure_target(${target_name})
endfunction()

#For plugins that are not linked into other targets but may be loaded dynamically at runtime using dlopen-like functionality.
function(brt_add_shared_library_module target_name)
  if ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin") OR (${CMAKE_SYSTEM_NAME} MATCHES "iOSCross") OR (${CMAKE_SYSTEM_NAME} MATCHES "iOS"))
    add_library(${target_name} SHARED ${ARGN})
  else()
    #On Windows, this target shouldn't generate an import lib, but I don't know how to disable it.
    add_library(${target_name} MODULE ${ARGN})
  endif()

  brt_configure_target(${target_name})
endfunction()

function(brt_add_executable target_name)
  if(${CMAKE_SYSTEM_NAME} MATCHES "iOSCross")
    message(FATAL_ERROR "iOS doesn't support commmand line tool")
  endif()
  add_executable(${target_name} ${ARGN})
  brt_configure_target(${target_name})
endfunction()

function(brt_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()
endfunction()

#Adjust warning flags for CUDA
if(brt_USE_CUDA)
  set_msvc_c_cpp_compiler_warning_level(3)
else()
  set_msvc_c_cpp_compiler_warning_level(4)
endif()

set(brt_DELAYLOAD_FLAGS "")

include_directories(
  ${BRT_INCLUDE_DIR}
  ${REPO_ROOT}/include/core/session
)

configure_file(brt_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/brt_config.h)
#if(WIN32)
#  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/Debug/requirements.txt)
#  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/Release/requirements.txt)
#  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo/requirements.txt)
#  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/MinSizeRel/requirements.txt)
#else()
#  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt)
#endif()

if (brt_USE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 70)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  enable_language(CUDA)
  message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  if (WIN32)
    set(CMAKE_CUDA_STANDARD 17)
    foreach(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      string(APPEND CMAKE_CXX_FLAGS " /external:I\"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY}\"")
    endforeach()
  else()
    #CUDA 10.2 on Linux doesn't support C++17
    set(CMAKE_CUDA_STANDARD 17)
  endif()
  
  #comment out copying cudnn
  #file(TO_CMAKE_PATH ${brt_CUDNN_HOME} brt_CUDNN_HOME)
  
  set(BRT_CUDA_LIBRARIES CUDA::cudart)


  if (brt_ENABLE_NVTX_PROFILE)
    list(APPEND BRT_CUDA_LIBRARIES CUDA::cublas cudnn CUDA::curand CUDA::cufft CUDA::nvToolsExt)
  else()
    list(APPEND BRT_CUDA_LIBRARIES CUDA::cublas cudnn CUDA::curand CUDA::cufft)
  endif()

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL "aarch64-linux-gnu")
      # Support for Jetson/Tegra ARM devices
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_53,code=sm_53") # TX1, Nano
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_62,code=sm_62") # TX2
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_72,code=sm_72") # AGX Xavier, NX Xavier
    else()
      # the following compute capabilities are removed in CUDA 11 Toolkit
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80") # A series
      endif()
    endif()
  endif()
  
  message(STATUS "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror default-stream-launch")
  endif()

  if (NOT WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")
  endif()
  
  # Options passed to cudafe
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=bad_friend_decl\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=expr_has_no_effect\"")

  if (brt_ENABLE_CUDA_LINE_NUMBER_INFO)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>")
  endif()
endif()

message(STATUS "CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}")

#names in this var must match the directory names under brt/core/providers
#build all existing targets
set(BRT_TARGETS
    brt_common
    brt_ir
    brt_framework
    brt_device_cpu
    brt_provider_cpu
    brt_device_upmem
    brt_provider_upmem)

if(brt_USE_CUDA)
  list(APPEND BRT_TARGETS brt_device_cuda)
  list(APPEND BRT_TARGETS brt_provider_cuda)
endif()

if(brt_ENABLE_PYTHON_BINDINGS)
  list(APPEND BRT_TARGETS brt_python_bindings)
endif()

#if(brt_ENABLE_EAGER_MODE)
#  list(APPEND BRT_TARGETS brt_eager)
#endif()
foreach(target_name ${BRT_TARGETS})
  include(${target_name}.cmake)
endforeach()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  list(APPEND brt_EXTERNAL_LIBRARIES log)
endif()

if(WIN32)
  list(APPEND brt_EXTERNAL_LIBRARIES ${SYS_PATH_LIB} Shlwapi)
  list(APPEND brt_EXTERNAL_LIBRARIES debug Dbghelp)
else()
  list(APPEND brt_EXTERNAL_LIBRARIES nsync_cpp)
  list(APPEND brt_EXTERNAL_LIBRARIES ${CMAKE_DL_LIBS} Threads::Threads)
endif()

if (WIN32)
  if (WINDOWS_STORE)
    # Setting WINAPI_FAMILY, WINVER and _WIN32_WINNT restrict the APIs exposed in Windows headers to those available
    # in store or desktop, and that support at least the version of Windows specified
    # add_compile_definitions(WINAPI_FAMILY=2)
    add_compile_definitions(WINVER=0x0A00 _WIN32_WINNT=0x0A00)  # Support Windows 10
    # /ZW adds /FUplatform.winmd and /FUwindows.winmd. windows.winmd, in turn, overrides the include path for
    # the cppwinrt headers, which we set to use the WinML built ones.
    # Instead, we use /ZW:nostdlib and force include platform.winml only.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:SHELL:/ZW:nostdlib /FUplatform.winmd>")
    get_filename_component(msvc_path "${CMAKE_C_COMPILER}/../../../.." ABSOLUTE)
    link_directories("${msvc_path}/lib/${brt_target_platform}/store")
    add_link_options(/APPCONTAINER)
  else()
    add_compile_definitions(WINVER=0x0601)  # Support Windows 7 and newer
    # add_compile_definitions(WINAPI_FAMILY=3)
    # FIXME adding _WIN32_WINNT=0x0601 and WINAPI_FAMILY is desirable. However, it hides some symbols that are used in WRL
    # headers and breaks the build of WinML. We should have separate WINVER/_WIN32_WINNT definitions for WinML
  endif()
endif()

# collect all brt components to brt.obj
add_library(brt.objs INTERFACE)
target_sources(brt.objs INTERFACE
  $<TARGET_OBJECTS:brt_common>
  $<TARGET_OBJECTS:brt_framework>
  $<TARGET_OBJECTS:brt_ir>
  $<TARGET_OBJECTS:brt_device_cpu>
  $<TARGET_OBJECTS:brt_provider_cpu>
  $<TARGET_OBJECTS:brt_device_upmem>
  $<TARGET_OBJECTS:brt_provider_upmem>
)
target_link_libraries(brt.objs INTERFACE
  brt_common
  brt_framework
  brt_ir
  brt_device_cpu
  brt_provider_cpu
  brt_provider_upmem
  brt_device_upmem
)
if(brt_USE_CUDA)
  target_sources(brt.objs INTERFACE
    $<TARGET_OBJECTS:brt_device_cuda>
    $<TARGET_OBJECTS:brt_provider_cuda>)
  target_link_libraries(brt.objs INTERFACE
    brt_device_cuda
    brt_provider_cuda)
endif()

#The following files may use the 'brt_libs' and 'brt_EXTERNAL_LIBRARIES' vars
if (brt_BUILD_SHARED_LIB OR brt_BUILD_APPLE_FRAMEWORK OR brt_ENABLE_PYTHON_BINDINGS)
  if (brt_BUILD_APPLE_FRAMEWORK AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS")
    message(FATAL_ERROR "brt_BUILD_APPLE_FRAMEWORK can only be enabled for macOS or iOS.")
  endif()
  include(brt_shared.cmake)
endif()

# The following build UNIT_TEST
if (brt_BUILD_UNIT_TESTS)
  if(NOT TARGET GTest::gtest)
    message("Use gtest from submodule")
    # gtest and gmock
    set_msvc_c_cpp_compiler_warning_level(4)
    add_subdirectory(${REPO_ROOT}/../external/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
    set_msvc_c_cpp_compiler_warning_level(3)
    set_target_properties(gmock PROPERTIES FOLDER "External/GTest")
    if(NOT MSVC)
      # disable treating all warnings as errors for gmock
      target_compile_options(gmock PRIVATE "-w")
      target_compile_options(gtest PRIVATE "-w")
    endif()
    set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
    set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
    set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")
  endif()
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  include(brt_unittests.cmake)
endif()
