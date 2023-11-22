
module attributes {byre.container_module} {
  func.func @test1(%arg0 : memref<100x32xf32, "HBMPIM"> {byre.argname = "A", byre.argtype = 1: i32},
              %arg1 : memref<100x32xf32, "HBMPIM"> {byre.argname = "B", byre.argtype = 1: i32},
              %arg2 : memref<100x32xf32, "HBMPIM"> {byre.argname = "C", byre.argtype = 2: i32},
              %arg3 : memref<100x32xf32, "HBMPIM"> {byre.argname = "D", byre.argtype = 2: i32}) attributes {byre.entry_point} {

    byre.compute @pytorch.gemv_hbm.fp32(%arg0, %arg1, %arg2){backend_config = "", byteir_attrs = {device = "HBMPIM"}, device = "HBMPIM"}  : memref<100x32xf32, "HBMPIM">, memref<100x32xf32, "HBMPIM">, memref<100x32xf32, "HBMPIM">
    byre.compute @pytorch.gemv_hbm.fp32(%arg1, %arg2, %arg3){backend_config = "", byteir_attrs = {device = "HBMPIM"}, device = "HBMPIM"}  : memref<100x32xf32, "HBMPIM">, memref<100x32xf32, "HBMPIM">, memref<100x32xf32, "HBMPIM">
    return
  }
}
