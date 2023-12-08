module attributes {byre.container_module} {
  func.func @test_recv(%arg0 : memref<4xf32, "cuda"> {byre.argname = "src", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @NCCLRecv_f32(%arg0) {rank = 0 : i64} : memref<4xf32, "cuda">
    return
  }
}