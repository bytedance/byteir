module attributes {byre.container_module} {
  func.func @test_broadcast(%arg0 : memref<8xf32, "cuda"> {byre.argname = "in0", byre.argtype = 1: i32},
                            %arg1 : memref<4xf32, "cuda"> {byre.argname = "out", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @nccl.Broadcast(%arg0, %arg1) { len = 4 : i64, root = 0 : i64 } : memref<8xf32, "cuda">, memref<4xf32, "cuda">
    return
  }
}
