module attributes {byre.container_module} {
  func.func @main(%arg0 : memref<100x32xf32, "cpu"> {byre.argname = "A", byre.argtype = 1: i32},
              %arg1 : memref<100x32xf32, "cpu"> {byre.argname = "B", byre.argtype = 1: i32},
              %arg2 : memref<100x32xf32, "cpu"> {byre.argname = "C", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    %alloc = memref.alloc() : memref<100x32xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<100x32xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<100x32xf32, "cuda">
    byre.compute @cpu2cuda(%arg0, %alloc) {memory_effects = [1 : i32, 2 : i32]} : memref<100x32xf32, "cpu">, memref<100x32xf32, "cuda">
    byre.compute @cpu2cuda(%arg1, %alloc_1) {memory_effects = [1 : i32, 2 : i32]} : memref<100x32xf32, "cpu">, memref<100x32xf32, "cuda">
    byre.compute @AddOp_f32f32_f32(%alloc, %alloc_1, %alloc_2) {memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<100x32xf32, "cuda">, memref<100x32xf32, "cuda">, memref<100x32xf32, "cuda">
    byre.compute @cuda2cpu(%alloc_2, %arg2) {memory_effects = [1 : i32, 2 : i32]} : memref<100x32xf32, "cuda">, memref<100x32xf32, "cpu">
    return
  }
}
