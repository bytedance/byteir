add_mlir_dialect(CatOps cat)
add_mlir_doc(CatOps CatOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS CatBase.td)
mlir_tablegen(CatOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CatOpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(MLIRCatOpInterfacesIncGen)