set(LLVM_TARGET_DEFINITIONS CclTransformOps.td)
mlir_tablegen(CclTransformOps.h.inc -gen-op-decls)
mlir_tablegen(CclTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRCclTransformOpsIncGen)

add_mlir_doc(CclTransformOps CclTransformOps Dialects/ -gen-op-doc)