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
    "byre.custom"(%arg1, %arg2, %arg3, %arg9, %arg4, %arg5, %arg12) {callee = "custom", lib_path = "test/test_files/external_libs/libflash_attn.so", api_name = "run_flash_attn_fwd", version = "2.5.3", extra_args = [12288 : i64, 12288 : i64, 12288 : i64, 12288 : i64, 96 : i64, 96 : i64, 96 : i64, 96 : i64, 32 : i64, 32 : i64, 32 : i64, 32 : i64, 1 : i64, 3 : i64, 3 : i64, 32 : i64, 32 : i64, 0.5 : f32, 128 : i64, 128 : i64, 128 : i64, 128 : i64, 0.0 : f32, -1 : i64, 0 : i64]} : (memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<2xi64, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x3x128xf32, "cuda">, memref<1x3x128x128xf32, "cuda">) -> ()
    "byre.custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg9, %arg6, %arg7, %arg8, %arg10, %arg11) {callee = "custom", lib_path = "test/test_files/external_libs/libflash_attn.so", api_name = "run_flash_attn_bwd", version = "2.5.3", extra_args = [12288 : i64, 12288 : i64, 12288 : i64, 12288 : i64, 96 : i64, 96 : i64, 96 : i64, 96 : i64, 32 : i64, 32 : i64, 32 : i64, 32 : i64, 1 : i64, 3 : i64, 3 : i64, 32 : i64, 32 : i64, 0.5 : f32, 128 : i64, 128 : i64, 128 : i64, 128 : i64, 0.0 : f32, -1 : i64, 0 : i64]} : (memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x3x128xf32, "cuda">, memref<2xi64, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x128x3x32xf16, "cuda">, memref<1x3x128xf32, "cuda">, memref<1x3x128x32xf32, "cuda">) -> () 
    return
  }
}
