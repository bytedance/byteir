
set(brt_common_src_patterns
    "${BRT_INCLUDE_DIR}/brt/core/common/*.h"
    "${BRT_INCLUDE_DIR}/brt/core/common/logging/*.h"
    "${BRT_INCLUDE_DIR}/brt/core/common/utils/*.h"
    "${BRT_INCLUDE_DIR}/brt/core/platform/*.h"
    "${LIB_ROOT}/core/common/*.h"
    "${LIB_ROOT}/core/common/*.cc"
    "${LIB_ROOT}/core/common/logging/*.h"
    "${LIB_ROOT}/core/common/logging/*.cc"
    "${LIB_ROOT}/core/common/logging/sinks/*.h"
    "${LIB_ROOT}/core/common/logging/sinks/*.cc"
    "${LIB_ROOT}/core/common/utils/*.cc"
)

if(WIN32)
    list(APPEND brt_common_src_patterns
         "${LIB_ROOT}/core/platform/windows/*.h"
         "${LIB_ROOT}/core/platform/windows/*.cc"
         "${LIB_ROOT}/core/platform/windows/logging/*.h"
         "${LIB_ROOT}/core/platform/windows/logging/*.cc"
    )
    # Windows platform adapter code uses advapi32, which isn't linked in by default in desktop ARM
    if (NOT WINDOWS_STORE)
        list(APPEND brt_EXTERNAL_LIBRARIES advapi32)
    endif()
else()
    list(APPEND brt_common_src_patterns
         "${LIB_ROOT}/core/platform/posix/*.h"
         "${LIB_ROOT}/core/platform/posix/*.cc"
    )

    if (brt_USE_SYSLOG)
        list(APPEND brt_common_src_patterns
            "${LIB_ROOT}/core/platform/posix/logging/*.h"
            "${LIB_ROOT}/core/platform/posix/logging/*.cc"
        )
    endif()

    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
        list(APPEND brt_common_src_patterns
            "${LIB_ROOT}/core/platform/android/logging/*.h"
            "${LIB_ROOT}/core/platform/android/logging/*.cc"
        )
    endif()

    if (APPLE)
        list(APPEND brt_common_src_patterns
            "${LIB_ROOT}/core/platform/apple/logging/*.h"
            "${LIB_ROOT}/core/platform/apple/logging/*.mm"
            )
    endif()
endif()

if(CMAKE_GENERATOR_PLATFORM)
    # Multi-platform generator
    set(brt_target_platform ${CMAKE_GENERATOR_PLATFORM})
else()
    set(brt_target_platform ${CMAKE_SYSTEM_PROCESSOR})
endif()
if(brt_target_platform STREQUAL "ARM64")
    set(brt_target_platform "ARM64")
elseif(brt_target_platform STREQUAL "ARM64EC")
    set(brt_target_platform "ARM64EC")
elseif(brt_target_platform STREQUAL "ARM" OR CMAKE_GENERATOR MATCHES "ARM")
    set(brt_target_platform "ARM")
elseif(brt_target_platform STREQUAL "x64" OR brt_target_platform STREQUAL "x86_64" OR brt_target_platform STREQUAL "AMD64" OR CMAKE_GENERATOR MATCHES "Win64")
    set(brt_target_platform "x64")
elseif(brt_target_platform STREQUAL "Win32" OR brt_target_platform STREQUAL "x86" OR brt_target_platform STREQUAL "i386" OR brt_target_platform STREQUAL "i686")
    set(brt_target_platform "x86")
endif()

if(brt_target_platform STREQUAL "ARM64EC")
    if (MSVC)
        link_directories("$ENV{VCINSTALLDIR}/Tools/MSVC/$ENV{VCToolsVersion}/lib/ARM64EC")
        link_directories("$ENV{VCINSTALLDIR}/Tools/MSVC/$ENV{VCToolsVersion}/ATLMFC/lib/ARM64EC")
        link_libraries(softintrin.lib)
        add_compile_options("/bigobj")
    endif()
endif()

file(GLOB brt_common_src CONFIGURE_DEPENDS
    ${brt_common_src_patterns}
    )

source_group(TREE ${REPO_ROOT} FILES ${brt_common_src})

brt_add_object_library(brt_common ${brt_common_src})

# include nsync
#if(NOT WIN32)
#  target_include_directories(brt_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
#endif()

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/core/common"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/core")
set_target_properties(brt_common PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_common PROPERTIES FOLDER "Brt")

if(WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(brt_common PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
endif()

# check if we need to link against librt on Linux
include(CheckLibraryExists)
include(CheckFunctionExists)
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)

  if (NOT HAVE_CLOCK_GETTIME)
    set(CMAKE_EXTRA_INCLUDE_FILES time.h)
    check_function_exists(clock_gettime HAVE_CLOCK_GETTIME)
    set(CMAKE_EXTRA_INCLUDE_FILES)
  else()
    target_link_libraries(brt_common rt)
  endif()
endif()

# check if we need to link against libatomic due to std::atomic usage by the threadpool code
# e.g. Raspberry Pi requires this
if (brt_LINK_LIBATOMIC)
  list(APPEND brt_EXTERNAL_LIBRARIES atomic)
endif()

# include dependent path
target_include_directories(brt_common PRIVATE "${REPO_ROOT}/../external/date/include")  # date

# involve half float
target_include_directories(brt_common PUBLIC SYSTEM
  $<BUILD_INTERFACE:${REPO_ROOT}/../external/half/include>)
install(
  DIRECTORY "${REPO_ROOT}/../external/half/include/half"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/")

if(APPLE)
  target_link_libraries(brt_common "-framework Foundation")
endif()
