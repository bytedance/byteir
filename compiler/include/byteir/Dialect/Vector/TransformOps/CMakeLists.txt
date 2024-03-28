set(LLVM_TARGET_DEFINITIONS VectorExtTransformOps.td)
mlir_tablegen(VectorExtTransformOps.h.inc -gen-op-decls)
mlir_tablegen(VectorExtTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRVectorExtTransformOpsIncGen)

add_mlir_doc(VectorExtTransformOps VectorExtTransformOps Dialects/ -gen-op-doc)