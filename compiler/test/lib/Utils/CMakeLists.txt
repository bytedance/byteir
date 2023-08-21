# Exclude tests from libMLIR.so
add_mlir_library(ByteIRTestUtils
  TestBroadcastDenseElementsAttr.cpp
  TestMergeTwoModules.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ByteIRMhloUtils
  ByteIRUtils
  MhloDialect
  )