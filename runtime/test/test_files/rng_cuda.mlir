module attributes {byre.container_module} {
  func.func @test_rng(%arg0 : memref<256x1024xf32, "cuda"> {byre.argname = "RngUniform0", byre.argtype = 2: i32},
                 %arg1 : memref<256x1024xf32, "cuda"> {byre.argname = "RngUniform1", byre.argtype = 2: i32},
                 %arg2 : memref<256x1024xf32, "cuda"> {byre.argname = "RngNormal", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @RngUniform(%arg0) {low = -1.000000e+00 : f32, high = 2.000000e+00 : f32} : memref<256x1024xf32, "cuda">
    byre.compute @RngUniform(%arg1) {low = -1.000000e+00 : f32, high = 2.000000e+00 : f32} : memref<256x1024xf32, "cuda">
    byre.compute @RngNormal(%arg2) {mean = 3.000000e+00 : f32, stddev = 2.330000e+00 : f32} : memref<256x1024xf32, "cuda">
    return
  }
}
