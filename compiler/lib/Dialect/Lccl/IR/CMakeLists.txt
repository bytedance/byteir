add_mlir_dialect_library(MLIRLcclDialect
  LcclOps.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  MLIRLcclOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
