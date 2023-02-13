add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRTransform)
add_public_tablegen_target(ByteIRTransformPassIncGen)
