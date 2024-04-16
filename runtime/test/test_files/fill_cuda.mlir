module attributes {byre.container_module} {
  func.func @test_fill(%arg0 : memref<512x128xf32, "cuda"> {byre.argname = "Fill0", byre.argtype = 2: i32},
                 %arg1 : memref<512x128xf32, "cuda"> {byre.argname = "Fill1", byre.argtype = 2: i32},
                 %arg2 : memref<512x128xf16, "cuda"> {byre.argname = "Fill1FP16", byre.argtype = 2: i32},
                 %arg3 : memref<3xf16, "cuda"> {byre.argname = "FillNonSplat", byre.argtype = 2: i32},
                 %arg4 : memref<3xi1, "cuda"> {byre.argname = "Fill2i1", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @FillOp(%arg0) {value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32, "cuda">
    byre.compute @FillOp(%arg1) {value = dense<1.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32, "cuda">
    byre.compute @FillOp(%arg2) {value = dense<1.000000e+00> : tensor<512x128xf16>} : memref<512x128xf16, "cuda">
    byre.compute @FillOp(%arg3) {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>} : memref<3xf16, "cuda">
    byre.compute @FillOp(%arg4) {value = dense<true> : tensor<3xi1>} : memref<3xi1, "cuda">
    return
  }
}