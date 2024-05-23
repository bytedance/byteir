set(LLVM_TARGET_DEFINITIONS GPUExtTransformOps.td)
mlir_tablegen(GPUExtTransformOps.h.inc -gen-op-decls)
mlir_tablegen(GPUExtTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRGPUExtTransformOpsIncGen)

add_mlir_doc(GPUExtTransformOps GPUExtTransformOps Dialects/ -gen-op-doc)