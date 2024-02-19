add_subdirectory(${BYTEIR_SRC_DIR}/../external/mlir-hlo ${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo EXCLUDE_FROM_ALL)

include_directories(${BYTEIR_SRC_DIR}/../external/mlir-hlo)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo)
include_directories(${BYTEIR_SRC_DIR}/../external/mlir-hlo/stablehlo)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo/stablehlo)

install(DIRECTORY ${BYTEIR_SRC_DIR}/../external/mlir-hlo ${BYTEIR_SRC_DIR}/../external/mlir-hlo/include/mlir-hlo
  DESTINATION external/include
  COMPONENT byteir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  )

install(DIRECTORY ${BYTEIR_BINARY_DIR}/mlir-hlo ${BYTEIR_BINARY_DIR}/mlir-hlo/include/mlir-hlo
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
