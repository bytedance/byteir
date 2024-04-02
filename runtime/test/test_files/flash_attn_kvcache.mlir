module attributes {byre.container_module} {
  func.func @test_flash_attn_kvcache(%arg0 : memref<2x1x3x32xf16, "cuda"> {byre.argname = "Q", byre.argtype = 2: i32},
                 %arg1 : memref<2x128x3x32xf16, "cuda"> {byre.argname = "KCache", byre.argtype = 2: i32},
                 %arg2 : memref<2x128x3x32xf16, "cuda"> {byre.argname = "VCache", byre.argtype = 2: i32},
                 %arg3 : memref<2x1x3x32xf16, "cuda"> {byre.argname = "K", byre.argtype = 2: i32},
                 %arg4 : memref<2x1x3x32xf16, "cuda"> {byre.argname = "V", byre.argtype = 2: i32},
                 %arg5: memref<2xi32, "cuda"> {byre.argname = "SeqLenK", byre.argtype = 2: i32},
                 %arg6: memref<2x1x3x32xf16, "cuda"> {byre.argname = "Output", byre.argtype = 2: i32},
                 %arg7 : memref<2x3x1xf32, "cuda"> {byre.argname = "SoftmaxLse", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    "byre.custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) {callee = "custom", lib_path = "test/test_files/external_libs/libflash_attn.so", api_name = "run_flash_attn_kvcache", version = "2.5.3", extra_args = [96 : i64, 12288 : i64, 12288 : i64, 96 : i64, 96 : i64, 96 : i64, 96 : i64, 96 : i64, 96 : i64, 96 : i64, 96 : i64, 96 : i64, 32 : i64, 32 : i64, 32 : i64, 32 : i64, 32 : i64, 32 : i64, 2 : i64, 3 : i64, 3 : i64, 32 : i64, 32 : i64, 1 : i64, 0.5 : f32, 1 : i64, 128 : i64, 128 : i64, 128 : i64, -1 : i64, -1 : i64]} : (memref<2x1x3x32xf16, "cuda">, memref<2x128x3x32xf16, "cuda">, memref<2x128x3x32xf16, "cuda">, memref<2x1x3x32xf16, "cuda">, memref<2x1x3x32xf16, "cuda">, memref<2xi32, "cuda">, memref<2x1x3x32xf16, "cuda">, memref<2x3x1xf32, "cuda">) -> ()
    return
  }
}
