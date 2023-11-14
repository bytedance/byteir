module attributes {byre.container_module} {
  func.func @test_send(%arg0 : memref<4xf32, "cuda"> {byre.argname = "src", byre.argtype = 1: i32}) attributes {byre.entry_point} {
    byre.compute @NCCLSend_f32(%arg0) {rank = 1 : i64} : memref<4xf32, "cuda">
    return
  }
}