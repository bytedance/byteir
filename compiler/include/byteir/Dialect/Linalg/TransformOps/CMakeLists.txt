set(LLVM_TARGET_DEFINITIONS LinalgExtTransformOps.td)
mlir_tablegen(LinalgExtTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgExtTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLinalgExtTransformOpsIncGen)

add_mlir_doc(LinalgExtTransformOps LinalgExtTransformOps Dialects/ -gen-op-doc)