module attributes {byre.container_module} {
  func.func @test_all_gather(%arg0 : memref<4xf32, "cuda"> {byre.argname = "in0", byre.argtype = 1: i32},
                             %arg1 : memref<8xf32, "cuda"> {byre.argname = "out", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @nccl.AllGather(%arg0, %arg1) : memref<4xf32, "cuda">, memref<8xf32, "cuda">
    return
  }
}
