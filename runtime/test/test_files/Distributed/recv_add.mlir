module attributes {byre.container_module} {
  func.func @test_recv_add(%arg0 : memref<4xf32, "cuda"> {byre.argname = "in", byre.argtype = 1: i32}, 
                           %arg1 : memref<4xf32, "cuda"> {byre.argname = "out0", byre.argtype = 2: i32}, 
                           %arg2 : memref<4xf32, "cuda"> {byre.argname = "out1", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @NCCLRecv_f32(%arg1) {rank = 0 : i64} : memref<4xf32, "cuda">
    byre.compute @AddOp_f32f32_f32(%arg0, %arg1, %arg2) : memref<4xf32, "cuda">, memref<4xf32, "cuda">, memref<4xf32, "cuda">         
    return
  }
}