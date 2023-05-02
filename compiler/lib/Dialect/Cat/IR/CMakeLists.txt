add_mlir_dialect_library(MLIRCatDialect
  CatDialect.cpp

  DEPENDS
  MLIRCatOpsIncGen
  MLIRCatOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

