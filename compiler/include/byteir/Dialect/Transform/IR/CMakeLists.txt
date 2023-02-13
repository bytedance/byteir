set(LLVM_TARGET_DEFINITIONS TransformExtOps.td)
mlir_tablegen(TransformExtOps.h.inc -gen-op-decls)
mlir_tablegen(TransformExtOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTransformExtOpsIncGen)

add_mlir_doc(TransformExtOps TransformExtOps Dialects/ -gen-op-doc)