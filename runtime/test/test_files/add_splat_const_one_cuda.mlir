module attributes {byre.container_module} {
    func.func @mhlo_add_splat_const(%arg0: memref<100x32xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32},
                               %arg1: memref<100x32xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
        %0 = memref.alloc() : memref<100x32xf32, "cuda">
        byre.compute @FillOp(%0) {value = dense<1.000000e+00> : tensor<100x32xf32>} : memref<100x32xf32, "cuda">
        byre.compute @AddOpf32f32f32(%arg0, %0, %arg1) : memref<100x32xf32, "cuda">, memref<100x32xf32, "cuda">, memref<100x32xf32, "cuda">
        return
    }
}