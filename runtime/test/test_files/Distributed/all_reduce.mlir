module attributes {byre.container_module} {
  func.func @test_all_reduce(%arg0 : memref<4xf32, "cuda"> {byre.argname = "in0", byre.argtype = 1: i32},
                             %arg1 : memref<4xf32, "cuda"> {byre.argname = "out", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @nccl.AllReduce(%arg0, %arg1) { reduction = "sum" , replica_group = [1 ,2, 3]} : memref<4xf32, "cuda">, memref<4xf32, "cuda">
    return
  }
}
