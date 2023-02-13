// RUN: byteir-opt -byre-fold %s | FileCheck %s

module attributes {byre.container_module} {
  func.func @fold_alias(%arg0: memref<512xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<512xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @fold_alias
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<128xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<256xf32>
    %6 = memref.alloc() : memref<128xf32>
    %7 = memref.alloc() : memref<512xf32>
    byre.compute @AliasOp(%arg0, %0) {arg_alias, offset = 128 : i32} : memref<512xf32>, memref<256xf32>
    byre.compute @AliasOp(%0, %1) {offset = 32 : i32} : memref<256xf32>, memref<128xf32>
// CHECK: byre.compute @AliasOp(%arg0, %alloc_0) {arg_alias, offset = 160 : i32}
    byre.compute @AliasOp(%arg1, %2) {arg_alias, offset = 128 : i32} : memref<512xf32>, memref<256xf32>
    byre.compute @AliasOp(%2, %3) {offset = 32 : i32} : memref<256xf32>, memref<128xf32>
// CHECK: byre.compute @AliasOp(%arg1, %alloc_2) {arg_alias, offset = 160 : i32}
    byre.compute @AliasOp(%4, %5) {offset = 128 : i32} : memref<512xf32>, memref<256xf32>
    byre.compute @AliasOp(%5, %6) {offset = 32 : i32} : memref<256xf32>, memref<128xf32>
// CHECK: byre.compute @AliasOp(%alloc_3, %alloc_5) {offset = 160 : i32}
    byre.compute @SomeOp(%1, %3, %6) : memref<128xf32>, memref<128xf32>, memref<128xf32>
    return
  }

  func.func @fold_identity_alias(%arg0: memref<256xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<256xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
// CHECK-LBAEL: func.func @fold_identity_alias
    %0 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%arg0, %0) {arg_alias, offset = 0 : i32} : memref<256xf32>, memref<256xf32>
    byre.compute @SomeOp(%0, %arg1) : memref<256xf32>, memref<256xf32>
// CHECK: byre.compute @SomeOp(%arg0, %arg1)
    return
  }

  func.func @remove_unused_alias(%arg0: memref<256xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<256xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @remove_unused_alias
    %0 = memref.alloc() : memref<128xf32>
    byre.compute @AliasOp(%arg0, %0) {arg_alias, offset = 128 : i32} : memref<256xf32>, memref<128xf32>
    byre.compute @SomeOp(%arg0, %arg1) : memref<256xf32>, memref<256xf32>
// CHECK-NOT: byre.compute @AliasOp
// CHECK: byre.compute @SomeOp(%arg0, %arg1)
    return
  }
}
