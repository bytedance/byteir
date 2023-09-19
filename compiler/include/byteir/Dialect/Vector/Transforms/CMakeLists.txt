set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRVector)
add_public_tablegen_target(ByteIRVectorPassIncGen)
