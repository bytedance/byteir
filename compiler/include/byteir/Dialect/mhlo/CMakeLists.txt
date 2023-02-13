set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRMhlo)
add_public_tablegen_target(ByteIRMhloPassIncGen)
