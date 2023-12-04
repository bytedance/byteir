module attributes {byre.container_module} {
  func.func @test_rng_state(%arg0 : memref<i64, "cuda"> {byre.argname = "seed", byre.argtype = 2: i32},
                 %arg1 : memref<i64, "cuda"> {byre.argname = "offset0", byre.argtype = 2: i32},
                 %arg2 : memref<i64, "cuda"> {byre.argname = "offset1", byre.argtype = 2: i32},
                 %arg3 : memref<i64, "cuda"> {byre.argname = "offset2", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @GetSeed(%arg0) {device = "cuda", memory_effects = [2 : i32]} : memref<i64, "cuda">
    byre.compute @NextOffset(%arg1) {device = "cuda", memory_effects = [2 : i32]} : memref<i64, "cuda">
    byre.compute @NextOffset(%arg2) {device = "cuda", memory_effects = [2 : i32]} : memref<i64, "cuda">
    byre.compute @NextOffset(%arg3) {device = "cuda", memory_effects = [2 : i32]} : memref<i64, "cuda">
    return
  }
}