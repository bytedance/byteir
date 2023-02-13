add_mlir_dialect(LaceOps lace)
add_mlir_doc(LaceOps LaceOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS LaceBase.td)
mlir_tablegen(LaceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LaceOpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(MLIRLaceOpInterfacesIncGen)
