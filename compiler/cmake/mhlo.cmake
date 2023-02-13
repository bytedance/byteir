set(MHLO_BUILD_EMBEDDED ON)

add_subdirectory(${REPO_ROOT_DIR}/../external/mlir-hlo ${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo EXCLUDE_FROM_ALL)

# FIXME: remove this when upstream fix
target_link_libraries(MhloDialect PUBLIC StablehloTypeInference StablehloAssemblyFormat)
target_link_libraries(GmlStTilingInterface PUBLIC GmlStDialect)
target_link_libraries(GmlStPasses PUBLIC MLIRGmlStUtils)
target_link_libraries(THLODialect PUBLIC GmlStTilingInterface)

include_directories(${REPO_ROOT_DIR}/../external/mlir-hlo)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo)
include_directories(${REPO_ROOT_DIR}/../external/mlir-hlo/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo/include)
include_directories(${REPO_ROOT_DIR}/../external/mlir-hlo/stablehlo)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo/stablehlo)

install(DIRECTORY ${REPO_ROOT_DIR}/../external/mlir-hlo ${REPO_ROOT_DIR}/../external/mlir-hlo/include/mlir-hlo
  DESTINATION external/include
  COMPONENT byteir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  )

install(DIRECTORY ${CMAKE_BINARY_DIR}/mlir-hlo ${CMAKE_BINARY_DIR}/mlir-hlo/include/mlir-hlo
  DESTINATION external/include
  COMPONENT byteir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )
