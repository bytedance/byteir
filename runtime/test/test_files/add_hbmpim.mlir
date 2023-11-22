
module attributes {byre.container_module} {
  func.func @test1(%arg0 : memref<100x32xf32, "hbmpim"> {byre.argname = "A", byre.argtype = 1: i32},
              %arg1 : memref<100x32xf32, "hbmpim"> {byre.argname = "B", byre.argtype = 1: i32},
              %arg2 : memref<100x32xf32, "hbmpim"> {byre.argname = "C", byre.argtype = 2: i32},
              %arg3 : memref<100x32xf32, "hbmpim"> {byre.argname = "D", byre.argtype = 2: i32}) attributes {byre.entry_point} {

    byre.compute @pytorch.add_hbm.fp32(%arg0, %arg1, %arg2){backend_config = "", byteir_attrs = {device = "hbmpim"}, device = "hbmpim"}  : memref<100x32xf32, "hbmpim">, memref<100x32xf32, "hbmpim">, memref<100x32xf32, "hbmpim">
    byre.compute @pytorch.add_hbm.fp32(%arg1, %arg2, %arg3){backend_config = "", byteir_attrs = {device = "hbmpim"}, device = "hbmpim"}  : memref<100x32xf32, "hbmpim">, memref<100x32xf32, "hbmpim">, memref<100x32xf32, "hbmpim">
    return
  }
}
