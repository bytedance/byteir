module attributes {byre.container_module, gpu.container_module} {
  func.func @forward(%arg0: memref<3x2xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<3x8xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %alloc = memref.alloc() : memref<512xi8, "cuda">
    %alloc_0 = memref.alloc() : memref<16x8xf32, "cuda">
    byre.compute @FillOp(%alloc_0) {device = "cuda", memory_effects = [2 : i32], value = dense<2.000000e+00> : tensor<16x8xf32>} : memref<16x8xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<8x16xf32, "cuda">
    byre.compute @FillOp(%alloc_1) {device = "cuda", memory_effects = [2 : i32], value = dense<4.000000e+00> : tensor<8x16xf32>} : memref<8x16xf32, "cuda">
    %0 = "byre.alias"(%alloc) <{offset = 0 : i64}> {device = "cuda"} : (memref<512xi8, "cuda">) -> memref<2x3xf32, "cuda">
    byre.compute @TransposeOp_f32_f32(%arg0, %0) {device = "cuda", memory_effects = [1 : i32, 2 : i32], permutation = dense<[1, 0]> : tensor<2xi64>} : memref<3x2xf32, "cuda">, memref<2x3xf32, "cuda">
    %1 = "byre.alias"(%alloc) <{offset = 256 : i64}> {device = "cuda"} : (memref<512xi8, "cuda">) -> memref<8x3xf32, "cuda">
    byre.compute @nccl.AllGather(%0, %1) {axis = 0 : i64, device = "cuda", memory_effects = [1 : i32, 2 : i32], replica_group = [0, 1, 2, 3], synchronous = true} : memref<2x3xf32, "cuda">, memref<8x3xf32, "cuda">
    %2 = "byre.alias"(%alloc) <{offset = 0 : i64}> {device = "cuda"} : (memref<512xi8, "cuda">) -> memref<3x16xf32, "cuda">
    byre.compute @MatmulOp_f32f32_f32(%1, %alloc_1, %2) {device = "cuda", lhs_contracting_dimension = 0 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<8x3xf32, "cuda">, memref<8x16xf32, "cuda">, memref<3x16xf32, "cuda">
    %3 = "byre.alias"(%alloc) <{offset = 256 : i64}> {device = "cuda"} : (memref<512xi8, "cuda">) -> memref<3x16xf32, "cuda">
    byre.compute @PTXOp(%2, %3) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 2 : i32], call_convention = "bare_ptr", device = "cuda", device_file_name = "ccl.ptx", kernel_name = "Unknown0", memory_effects = [1 : i32, 2 : i32]} : memref<3x16xf32, "cuda">, memref<3x16xf32, "cuda">
    %4 = "byre.alias"(%alloc) <{offset = 0 : i64}> {device = "cuda"} : (memref<512xi8, "cuda">) -> memref<3x8xf32, "cuda">
    byre.compute @MatmulOp_f32f32_f32(%3, %alloc_0, %4) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<3x16xf32, "cuda">, memref<16x8xf32, "cuda">, memref<3x8xf32, "cuda">
    %5 = "byre.alias"(%alloc) <{offset = 256 : i64}> {device = "cuda"} : (memref<512xi8, "cuda">) -> memref<3x8xf32, "cuda">
    byre.compute @PTXOp(%4, %5) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 2 : i32], call_convention = "bare_ptr", device = "cuda", device_file_name = "ccl.ptx", kernel_name = "Unknown1", memory_effects = [1 : i32, 2 : i32]} : memref<3x8xf32, "cuda">, memref<3x8xf32, "cuda">
    byre.compute @nccl.AllReduce(%5, %arg1) {device = "cuda", memory_effects = [1 : i32, 2 : i32], reduction = "sum", replica_group = [0, 1, 2, 3], synchronous = true} : memref<3x8xf32, "cuda">, memref<3x8xf32, "cuda">
    return
  }
}
