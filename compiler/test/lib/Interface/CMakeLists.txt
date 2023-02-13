# Exclude tests from libMLIR.so
add_mlir_library(ByteIRTestInterface
  TestByreOpInterface.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRByreDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRByreDialect
  )