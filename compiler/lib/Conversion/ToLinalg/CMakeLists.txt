add_byteir_conversion_library(ByteIRToLinalg
  HloFusionToLinalg.cpp
  TensorToLinalg.cpp
  UnrealizedCastToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/HloToLinalg

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  HloToLinalgUtils

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
  HloToLinalgUtils
  )
