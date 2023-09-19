// RUN: byteir-opt -memrefcopy-to-linalg="attach-attr=test" --split-input-file %s | FileCheck %s

func.func @memref_copy(%arg0: memref<2x4xf32>, %arg1: memref<2x2xf32>) {
  %0 = memref.subview %arg0[0, 0][2, 2][1, 2] : memref<2x4xf32> to memref<2x2xf32, strided<[4, 2]>>
  memref.copy %0, %arg1 : memref<2x2xf32, strided<[4, 2]>> to memref<2x2xf32>
  return
}
// CHECK-LABEL: func.func private @memref_copy_kernel
//   CHECK-SAME: attributes {test}
//   CHECK-NEXT: memref.subview
//   CHECK-NEXT: linalg.generic
// CHECK-LABEL: func.func @memref_copy
//   CHECK-NEXT: call @memref_copy_kernel
