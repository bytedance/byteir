add_mlir_dialect(LinalgExtOps linalg_ext)
add_mlir_doc(LinalgExtOps LinalgExtOps Dialects/ -gen-op-doc)


set(LLVM_TARGET_DEFINITIONS LinalgExtInterfaces.td)
mlir_tablegen(LinalgExtOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LinalgExtOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRLinalgExtInterfacesIncGen)
add_dependencies(MLIRLinalgExtOpsIncGen MLIRLinalgExtInterfacesIncGen)
