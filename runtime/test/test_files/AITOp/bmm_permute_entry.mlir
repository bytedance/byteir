module attributes {byre.container_module} {
  func.func @main(%arg0 : memref<384x256x256xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32},
                  %arg1 : memref<384x256x64xf32, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32},
                  %arg2 : memref<64x256x6x64xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    byre.compute @AITOp(%arg0, %arg1, %arg2) {kernel_name = "bmm_permute", ait_lib_file = "bmm_permute_a100.so"} : memref<384x256x256xf32, "cuda">, memref<384x256x64xf32, "cuda">, memref<64x256x6x64xf32, "cuda">
    return
  }
}
