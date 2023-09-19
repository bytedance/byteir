module attributes {byre.container_module} {
  func.func @test1(%arg0 : memref<100x32xf32, "cuda"> {byre.argname = "A", byre.argtype = 1: i32},
              %arg1 : memref<100x32xf32, "cuda"> {byre.argname = "B", byre.argtype = 1: i32},
              %arg2 : memref<100x32xf32, "cuda"> {byre.arg_alias_index = 0 : i64, byre.argname = "C", byre.argtype = 2: i32},
              %arg3 : memref<100x32xf32, "cuda"> {byre.arg_alias_index = 1 : i64, byre.argname = "D", byre.argtype = 2: i32}) attributes {byre.entry_point} {

    byre.copy(%arg0, %arg2) {callee = "cuda2cuda", device = "cuda"} : memref<100x32xf32, "cuda">, memref<100x32xf32, "cuda">
    byre.copy(%arg1, %arg3) {callee = "cuda2cuda", device = "cuda"} : memref<100x32xf32, "cuda">, memref<100x32xf32, "cuda">
    return
  }
}
