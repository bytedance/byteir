
module attributes {byre.container_module} {
  func.func @test1(%arg0 : memref<100x32xf32, "cpu"> {byre.argname = "A", byre.argtype = 1: i32},
              %arg1 : memref<100x32xf32, "cpu"> {byre.argname = "B", byre.argtype = 1: i32},
              %arg2 : memref<100x32xf32, "cpu"> {byre.argname = "C", byre.argtype = 2: i32},
              %arg3 : memref<100x32xf32, "cpu"> {byre.argname = "D", byre.argtype = 2: i32}) attributes {byre.entry_point} {

    byre.compute @AddOp_f32f32_f32(%arg0, %arg1, %arg2) : memref<100x32xf32, "cpu">, memref<100x32xf32, "cpu">, memref<100x32xf32, "cpu">
    byre.compute @AddOp_f32f32_f32(%arg1, %arg2, %arg3) : memref<100x32xf32, "cpu">, memref<100x32xf32, "cpu">, memref<100x32xf32, "cpu">
    return
  }
}
