module attributes {byre.container_module} {
  func.func @test_broadcast(%arg0 : memref<8xf32, "cuda"> {byre.argname = "in0", byre.argtype = 1: i32}) attributes {byre.entry_point} {
    byre.compute @nccl.Broadcast(%arg0) {replica_group = [1, 0, 2]} : memref<8xf32, "cuda">
    return
  }
}
