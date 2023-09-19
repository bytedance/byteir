// RUN: byteir-opt %s | FileCheck %s

module attributes {byre.container_module} {
  func.func @test_compute(%arg0: memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.compute @some_kernel(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
    return
  }
// CHECK-LABEL: func.func @test_compute
// CHECK: %arg0: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 1 : i32
//  CHECK-DAG: byre.argname = "A"
// CHECK: %arg1: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 2 : i32
//  CHECK-DAG: byre.argname = "B"
// CHECK: attributes {byre.entry_point} {
// CHECK:   byre.compute @some_kernel(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
// CHECK:   return
// CHECK: }


  func.func @test_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.copy(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
    return
  }
// CHECK-LABEL: func.func @test_copy
// CHECK: %arg0: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 1 : i32
//  CHECK-DAG: byre.argname = "A"
// CHECK: %arg1: memref<100x?xf32> {
//  CHECK-DAG: byre.argtype = 2 : i32
//  CHECK-DAG: byre.argname = "B"
// CHECK: attributes {byre.entry_point} {
// CHECK:   byre.copy(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
// CHECK:   return
// CHECK: }

  func.func @test_group_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}, %arg2 : memref<200x?xf32> {byre.argtype = 1: i32, byre.argname = "C"}, %arg3 : memref<200x?xf32> {byre.argtype = 2: i32, byre.argname = "D"}) attributes {byre.entry_point} {
    "byre.group_copy"(%arg0, %arg2, %arg1, %arg3) {callee = "h2d_array"} : (memref<100x?xf32>, memref<200x?xf32>, memref<100x?xf32>, memref<200x?xf32>) -> ()
    return
  }
// CHECK-LABEL: func.func @test_group_copy
// CHECK: "byre.group_copy"

  func.func @test_alias(%arg0 : memref<100x32xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x32xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    %0 = "byre.alias"(%arg0) {offset = 0: i64} : (memref<100x32xf32>) -> memref<100x32xf32>
    byre.compute @some_kernel(%0, %arg1) : memref<100x32xf32>, memref<100x32xf32>
    return
  }
// CHECK-LABEL: func.func @test_alias
// CHECK: "byre.alias"

  func.func @test_compute_shape(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    %0 = "byre.compute_shape"(%arg0) {shape_fn = "shape_fn"} : (memref<100x?xf32>) -> index
    return
  }
// CHECK-LABEL: func.func @test_compute_shape
// CHECK: "byre.compute_shape"
}

module attributes {byre.container_module, byre.memory_space = [1, "CPU", 12, "CUDA"]} {
// CHECK: module attributes {byre.container_module, byre.memory_space = [1, "CPU", 12, "CUDA"]}
  func.func @dummy() {
    return 
  }
}
