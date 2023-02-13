# Exclude tests from libMLIR.so
add_mlir_library(ByteIRTestTransformation
  TestMhloCanonicalizeExt.cpp
  TestConvertFuncToCustomCall.cpp
  TestConvertInsertion.cpp
  TestDTypeConversion.cpp
  TestFuncArgRearrangement.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ByteIRMhloPasses
  ByteIRUtils
  MhloDialect
  )