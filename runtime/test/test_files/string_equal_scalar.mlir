module attributes {byre.container_module} {
  func.func @main(%arg0 : memref<1x!ace.string, "cpu"> {byre.argname = "Input", byre.argtype = 1: i32},
                  %arg1 : memref<4xi1, "cpu"> {byre.argname = "Output", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<4x!ace.string, "cpu">
    byre.compute @FillOp(%0) {memory_effects = [2 : i32], value = dense<"aaa"> : tensor<4x!ace.string, "cpu">} : memref<4x!ace.string, "cpu">
    byre.compute @tf.Equal(%arg0, %0, %arg1) {memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x!ace.string, "cpu">, memref<4x!ace.string, "cpu">, memref<4xi1, "cpu">
    return
  }
}
