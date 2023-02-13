// RUN: byteir-opt --canonicalize %s | FileCheck %s

module attributes {byre.container_module} {
  func.func @collapse_alias(%arg0: memref<512xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<512xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @collapse_alias
    %0 = "byre.alias"(%arg0) {offset = 128 : i64} : (memref<512xf32>) -> memref<256xf32>
    %1 = "byre.alias"(%0) {offset = 32 : i64} : (memref<256xf32>) -> memref<128xf32>
// CHECK: byre.alias
//   CHECK-SAME: offset = 160
    %2 = memref.alloc() : memref<512xf32>
    %3 = "byre.alias"(%2) {offset = 128 : i64} : (memref<512xf32>) -> memref<256xf32>
    %4 = "byre.alias"(%3) {offset = 32 : i64} : (memref<256xf32>) -> memref<128xf32>
// CHECK: byre.alias
//   CHECK-SAME: offset = 160
    byre.compute @SomeOp(%1, %4, %arg1) : memref<128xf32>, memref<128xf32>, memref<512xf32>
    return
  }

  func.func @remove_identity_alias(%arg0: memref<256xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<256xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @remove_identity_alias
    %0 = "byre.alias"(%arg0) {offset = 0 : i64} : (memref<256xf32>) -> memref<256xf32>
    byre.compute @SomeOp(%0, %arg1) : memref<256xf32>, memref<256xf32>
// CHECK-NEXT: byre.compute @SomeOp(%arg0, %arg1)
    return
  }

  func.func @test_group_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}, %arg2 : memref<200x?xf32> {byre.argtype = 1: i32, byre.argname = "C"}, %arg3 : memref<200x?xf32> {byre.argtype = 2: i32, byre.argname = "D"}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @test_group_copy
    "byre.group_copy"(%arg0, %arg2, %arg1, %arg2) {callee = "d2h_array"} : (memref<100x?xf32>, memref<200x?xf32>, memref<100x?xf32>, memref<200x?xf32>) -> ()
// CHECK-NEXT: byre.copy(%arg0, %arg1) {callee = "d2h"}
    return
  }
}
