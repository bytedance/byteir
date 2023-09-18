set(LLVM_TARGET_DEFINITIONS ShardingInterface.td)
mlir_tablegen(ShardingInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShardingInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRShardingInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS InferDTensorInterface.td)
mlir_tablegen(InferDTensorInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InferDTensorInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRInferDTensorInterfaceIncGen)
