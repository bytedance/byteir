add_byteir_conversion_library(ByteIRHloToLinalg
  HloFusionToLinalg.cpp
  UnrealizedCastToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/HloToLinalg

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRConversionCommon

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MhloToLinalg
  MLIRRewrite
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
  ByteIRConversionCommon
  )
