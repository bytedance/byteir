add_subdirectory(IR)
add_subdirectory(TransformOps)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRLinalg)
add_public_tablegen_target(ByteIRLinalgPassIncGen)
