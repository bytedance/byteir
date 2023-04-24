add_mlir_dialect_library(MLIRLinalgExt
  LinalgExtInterfaces.cpp
  LinalgExtOps.cpp

  DEPENDS
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRLinalgUtils
  MLIRSupport
  MLIRLinalgExtUtils
)
