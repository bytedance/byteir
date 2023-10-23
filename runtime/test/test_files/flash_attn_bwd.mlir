module attributes {byre.container_module} {
  func.func @test_flash_attn_fwd_and_bwd(%arg0 : memref<1x128x3x32xf16, "cuda"> {byre.argname = "d_Out", byre.argtype = 2: i32},
                 %arg1 : memref<1x128x3x32xf16, "cuda"> {byre.argname = "Q", byre.argtype = 2: i32},
                 %arg2 : memref<1x128x3x32xf16, "cuda"> {byre.argname = "K", byre.argtype = 2: i32},
                 %arg3 : memref<1x128x3x32xf16, "cuda"> {byre.argname = "V", byre.argtype = 2: i32},
                 %arg4 : memref<1x128x3x32xf16, "cuda"> {byre.argname = "Out", byre.argtype = 2: i32},
                 %arg5 : memref<1x3x128xf32, "cuda"> {byre.argname = "SoftmaxLse", byre.argtype = 2: i32},
                 %arg6 : memref<1x128x3x32xf16, "cuda"> {byre.argname = "d_Q", byre.argtype = 2: i32},
                 %arg7 : memref<1x128x3x32xf16, "cuda"> {byre.argname = "d_K", byre.argtype = 2: i32},
                 %arg8 : memref<1x128x3x32xf16, "cuda"> {byre.argname = "d_V", byre.argtype = 2: i32},
                 %arg9 : memref<2xi64, "cuda"> {byre.argname = "RngState", byre.argtype = 2: i32},
                 %arg10 : memref<1x3x128xf32, "cuda"> {byre.argname = "d_SoftmaxLse", byre.argtype = 2: i32},
                 %arg11 : memref<1x3x128x32xf32, "cuda"> {byre.argname = "d_Q_accum", byre.argtype = 2: i32},
                 %arg12 : memref<1x3x128x128xf32, "cuda"> {byre.argname = "SoftmaxPtr", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    byre.compute @byteir.flash_attn_fwd(%arg1, %arg2, %arg3, %arg9, %arg4, %arg5, %arg12) {causal = true, dropout_p = 0.000000e+00 : f32, return_softmax = false, softmax_scale = 0.500000e+00 : f32} : memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<2xi64, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x3x128xf32, "cuda">, memref<1x3x128x128xf32, "cuda">
    byre.compute @byteir.flash_attn_bwd(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg9, %arg6, %arg7, %arg8, %arg10, %arg11) {causal = true, dropout_p = 0.000000e+00 : f32, softmax_scale = 0.500000e+00 : f32} : memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x3x128xf32, "cuda">, memref<2xi64, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x3x128xf32, "cuda">, memref<1x3x128x32xf32, "cuda">
    return
  }
}