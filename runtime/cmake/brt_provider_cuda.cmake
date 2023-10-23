set(brt_all_includes brt_common brt_framework)

# CUDA provide with default CUDA_ARCHITECTURES (70)
file(GLOB_RECURSE brt_cuda_provider_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/cuda/providers/default/*.h"
  "${LIB_ROOT}/backends/cuda/providers/default/*.h"
  "${LIB_ROOT}/backends/cuda/providers/default/*.cc"
  "${LIB_ROOT}/backends/cuda/providers/default/*.cu"
)

# exclude flash attension code from brt_cuda_provider_srcs
list(FILTER brt_cuda_provider_srcs EXCLUDE REGEX
  ".*/backends/cuda/providers/default/flash_attn/.*"
)

if (brt_BUILD_FLASH_ATTN)
  file(GLOB_RECURSE brt_cuda_provider_sm_80_cuda_srcs CONFIGURE_DEPENDS
    "${LIB_ROOT}/backends/cuda/providers/default/flash_attn/kernels/*.h"
    "${LIB_ROOT}/backends/cuda/providers/default/flash_attn/kernels/*.cu"
  )
  set_source_files_properties(${brt_cuda_provider_sm_80_cuda_srcs}
    PROPERTIES COMPILE_FLAGS "-gencode=arch=compute_80,code=sm_80")

  file(GLOB_RECURSE brt_cuda_provider_sm_80_cpp_srcs CONFIGURE_DEPENDS
    "${BRT_INCLUDE_DIR}/brt/backends/cuda/providers/default/flash_attn/*.h"
    "${LIB_ROOT}/backends/cuda/providers/default/flash_attn/*.cc"
    "${LIB_ROOT}/backends/cuda/providers/default/flash_attn/kernels/*.cc"
  )
  list(APPEND brt_cuda_provider_sm_80_cuda_srcs ${brt_cuda_provider_sm_80_cpp_srcs})
  brt_add_shared_library(brt_flash_attn_cuda ${brt_cuda_provider_sm_80_cuda_srcs})
  target_include_directories(brt_flash_attn_cuda PRIVATE SYSTEM ${REPO_ROOT}/../external/half/include)
  target_include_directories(brt_flash_attn_cuda PRIVATE SYSTEM "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include")
  # we know that all brt::* symbols are defined/linked in brt_provider_cuda library
  target_link_libraries(brt_flash_attn_cuda PRIVATE -Wl,--unresolved-symbols=ignore-in-object-files)

  set_target_properties(brt_flash_attn_cuda PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(brt_flash_attn_cuda PROPERTIES FOLDER "Brt")

  install(
    TARGETS brt_flash_attn_cuda
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()


list(APPEND brt_all_providers_srcs ${brt_cuda_provider_srcs})
list(APPEND brt_all_includes brt_device_cuda)

source_group(TREE ${REPO_ROOT} FILES ${brt_cuda_provider_srcs})

brt_add_object_library(brt_provider_cuda ${brt_cuda_provider_srcs})
# cutlass
target_include_directories(brt_provider_cuda PUBLIC "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include")

# add flash attention dependencies if any
if(FLASH_ATTN_INSTALL_PATH)
  target_link_libraries(brt_provider_cuda ${FLASH_ATTN_INSTALL_PATH})
  install(
    FILES "${FLASH_ATTN_INSTALL_PATH}"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

if (brt_BUILD_FLASH_ATTN)
  target_link_libraries(brt_provider_cuda brt_flash_attn_cuda)
endif()

target_link_libraries(brt_provider_cuda brt_framework brt_common brt_ir)
target_link_libraries(brt_provider_cuda ${BRT_CUDA_LIBRARIES})
brt_add_include_to_target(brt_provider_cuda ${brt_all_includes})
set_target_properties(brt_provider_cuda PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_provider_cuda PROPERTIES FOLDER "Brt")


install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/cuda/providers"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/cuda")
