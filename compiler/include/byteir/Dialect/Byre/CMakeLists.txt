add_mlir_dialect(ByreOps byre)
add_mlir_doc(ByreOps ByreOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS ByreBase.td)
mlir_tablegen(ByreOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ByreOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(ByreEnums.h.inc -gen-enum-decls)
mlir_tablegen(ByreEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(MLIRByreOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRByre)
add_public_tablegen_target(ByteIRByrePassIncGen)

add_subdirectory(Serialization)
