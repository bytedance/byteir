module attributes {byre.container_module} {
  func.func @main(%arg0: memref<?x?x3xf32, "cpu"> {byre.argname = "Input0", byre.argtype = 1 : i32},
                  %arg1: memref<?x?x3xf32, "cpu"> {byre.argname = "Input1", byre.argtype = 1 : i32},
                  %arg2: memref<?x?x3xf32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "kernel"} {
    %0:2 = "byre.compute_shape"(%arg0) {device = "cpu", shape_fn = "shape_fn", llvm_file_name = "shape_fn.ll"} : (memref<?x?x3xf32, "cpu">) -> (index, index)
    %1 = memref.alloc(%0#0, %0#1) : memref<?x?x3xf32, "cpu">
    byre.compute @AddOpf32f32f32(%arg0, %arg1, %1) {device = "cpu", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<?x?x3xf32, "cpu">, memref<?x?x3xf32, "cpu">, memref<?x?x3xf32, "cpu">
    byre.compute @AddOpf32f32f32(%1, %1, %arg2) {device = "cpu", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<?x?x3xf32, "cpu">, memref<?x?x3xf32, "cpu">, memref<?x?x3xf32, "cpu">
    return
  }
}
