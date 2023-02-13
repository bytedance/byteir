# Exclude tests from libMLIR.so
add_mlir_library(ByteIRTestUtils
  TestBroadcastDenseElementsAttr.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ByteIRMhloUtils
  MhloDialect
  )