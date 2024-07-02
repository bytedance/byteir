// RUN: byteir-opt -memref-to-byre --split-input-file %s | FileCheck %s

module attributes {byre.container_module} {
// CHECK: module attributes {byre.container_module}  {
  func.func @copy(%arg0: memref<4xf32, "cpu"> {byre.argname = "A", byre.argtype = 1 : i32}, %arg1: memref<4xf32, "cpu"> {byre.argname = "B", byre.argtype = 2 : i32}) attributes { byre.entry_point } {
    %alloc = memref.alloc() : memref<4xf32, "gpu">
    memref.copy %arg0, %alloc : memref<4xf32, "cpu"> to memref<4xf32, "gpu">
  // CHECK: byre.copy(%arg0, %alloc) {callee = "cpu2gpu"} : memref<4xf32, "cpu">, memref<4xf32, "gpu">
    return
  }
}

// -----

module attributes {byre.container_module} {
// CHECK: module attributes {byre.container_module}  {
  func.func @forward(%arg0: memref<i64, "cuda"> {byre.argname = "A", byre.argtype = 1 : i32}, %arg1: memref<2xi64, "cuda"> {byre.argname = "Out", byre.argtype = 2 : i32}) attributes { byre.entry_point } {
    %expand_shape = memref.expand_shape %arg0 [] output_shape [1] : memref<i64, "cuda"> into memref<1xi64, "cuda">
    // CHECK: byre.alias
    %alloc = memref.alloc() : memref<2xi64, "cuda">
    %subview = memref.subview %alloc[0] [1] [1] : memref<2xi64, "cuda"> to memref<1xi64, strided<[1]>, "cuda">
    // CHECK: byre.alias
    memref.copy %expand_shape, %subview : memref<1xi64, "cuda"> to memref<1xi64, strided<[1]>, "cuda">
    // CHECK: byre.copy
    memref.copy %alloc, %arg1 : memref<2xi64, "cuda"> to memref<2xi64, "cuda">
    // CHECK: byre.copy
    return
  }
}
