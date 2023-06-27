module attributes {byre.container_module} {
  func.func @test_rng(%arg0 : memref<256x1024xf32, "cuda"> {byre.argname = "RngUniformf320", byre.argtype = 2: i32},
                 %arg1 : memref<256x1024xf32, "cuda"> {byre.argname = "RngUniformf321", byre.argtype = 2: i32},
                 %arg2 : memref<256x1024xf32, "cuda"> {byre.argname = "RngNormal", byre.argtype = 2: i32},
                 %arg3 : memref<256x1024xf64, "cuda"> {byre.argname = "RngUniformf640", byre.argtype = 2: i32},
                 %arg4 : memref<256x1024xf64, "cuda"> {byre.argname = "RngUniformf641", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @RngUniform_f32f32_f32(%arg0) {low = -1.000000e+00 : f32, high = 2.000000e+00 : f32} : memref<256x1024xf32, "cuda">
    byre.compute @RngUniform_f32f32_f32(%arg1) {low = -1.000000e+00 : f32, high = 2.000000e+00 : f32} : memref<256x1024xf32, "cuda">
    byre.compute @RngNormal(%arg2) {mean = 3.000000e+00 : f32, stddev = 2.330000e+00 : f32} : memref<256x1024xf32, "cuda">
    byre.compute @RngUniform_f64f64_f64(%arg3) {low = -1.000000e+00 : f64, high = 2.000000e+00 : f64} : memref<256x1024xf64, "cuda">
    byre.compute @RngUniform_f64f64_f64(%arg4) {low = -1.000000e+00 : f64, high = 2.000000e+00 : f64} : memref<256x1024xf64, "cuda">
    return
  }
}
