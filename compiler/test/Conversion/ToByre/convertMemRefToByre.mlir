// RUN: byteir-opt -convert-lmhlo-to-byre %s | FileCheck %s

module attributes {byre.container_module} {
// CHECK: module attributes {byre.container_module}  {
  func.func @copy(%arg0: memref<4xf32, "cpu"> {byre.argname = "A", byre.argtype = 1 : i32}, %arg1: memref<4xf32, "cpu"> {byre.argname = "B", byre.argtype = 2 : i32}) attributes { byre.entry_point } {
    %alloc = memref.alloc() : memref<4xf32, "gpu">
    memref.copy %arg0, %alloc : memref<4xf32, "cpu"> to memref<4xf32, "gpu">
  // CHECK: byre.copy(%arg0, %alloc) {callee = "cpu2gpu"} : memref<4xf32, "cpu">, memref<4xf32, "gpu">
    return
  }
}