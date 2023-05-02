set(LLVM_TARGET_DEFINITIONS FuseHloToCatPattern.td)
mlir_tablegen(FuseHloToCatPattern.inc -gen-rewriters)
add_public_tablegen_target(FuseHloToCatPatternIncGen)

add_byteir_conversion_library(ByteIRHloToCat
  HloToCat.cpp
  FuseHloToCat.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/HloToCat

  LINK_LIBS PUBLIC
  MhloDialect
  MLIRCatDialect
  MLIRIR
  MLIRTransforms
  MLIRPass
  MLIRRewrite

  DEPENDS
  MLIRCatDialect
  MLIRTransforms
  MhloPasses
  FuseHloToCatPatternIncGen
  ByteIRConversionPassIncGen
)

target_include_directories(ByteIRHloToCat PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
