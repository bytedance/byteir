// RUN: byteir-opt %s -o %t
// RUN: byteir-opt --load-byre %s.bc -o %t0 && diff %t %t0
// RUN: byteir-opt --dump-byre="file-name=%t1.bc version=1.0.0" %s &>/dev/null && diff %t1.bc %s.bc
// RUN: byteir-opt --load-byre %t1.bc -o %t1 && diff %t %t1
// RUN: byteir-opt --load-byre --dump-byre="file-name=%t2.bc version=1.0.0" %s.bc &>/dev/null && diff %t2.bc %s.bc
// RUN: byteir-opt --load-byre %t2.bc -o %t2 && diff %t %t2

module attributes {byre.container_module} {
  func.func @test_compute(%arg0: memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B", byre.arg_alias_index = 0 : i64}) attributes {byre.entry_point, byteir.entry_point, tf.original_input_names} {
    byre.compute @some_kernel(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
    return
  }

  func.func @test_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.copy(%arg0, %arg1) {callee = "cuda2cuda"} : memref<100x?xf32>, memref<100x?xf32>
    return
  }

  func.func @test_group_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}, %arg2 : memref<200x?xf32> {byre.argtype = 1: i32, byre.argname = "C"}, %arg3 : memref<200x?xf32> {byre.argtype = 2: i32, byre.argname = "D"}) attributes {byre.entry_point} {
    "byre.group_copy"(%arg0, %arg2, %arg1, %arg3) {callee = "h2d_array"} : (memref<100x?xf32>, memref<200x?xf32>, memref<100x?xf32>, memref<200x?xf32>) -> ()
    return
  }

  func.func @test_alias(%arg0 : memref<100x32xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x32xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    %0 = "byre.alias"(%arg0) {offset = 0: i64} : (memref<100x32xf32>) -> memref<100x32xf32>
    byre.compute @some_kernel(%0, %arg1) : memref<100x32xf32>, memref<100x32xf32>
    return
  }

  func.func @test_scalar_attr(%arg0: memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.compute @some_kernel(%arg0, %arg1) {memory_effects = [1 : i32, 1 : i32], bool_attr = true, float_attr = 1.0 : f32, integer_attr = 1 : i32, string_attr = "string", ui8_attr = 1: ui8} : memref<100x?xf32>, memref<100x?xf32>
    return
  }

  func.func @test_dense_attr(%arg0: memref<100x100xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.compute @some_kernel(%arg0, %arg1) {dense_array_attr = array<i32: 10, 42>, weight0 = dense<1.0> : tensor<100x100xf32>, weight1 = dense<[-1, 1]> : tensor<2xi32>, value0 = dense<"-1"> : tensor<100x!ace.string>, value1 = dense<["test", "string"]> : tensor<2x!ace.string>} : memref<100x100xf32>, memref<100x?xf32>
    return
  }

  func.func @test_sparse_attr(%arg0: memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.compute @some_kernel(%arg0, %arg1) {memory_effects = [1 : i32, 1 : i32], unit_attr, type_attr = !ace.string, array_attr = ["a", "b", 1], dict_attr = {a = "a", b = 1 : ui32}} : memref<100x?xf32>, memref<100x?xf32>
    return
  }
}
