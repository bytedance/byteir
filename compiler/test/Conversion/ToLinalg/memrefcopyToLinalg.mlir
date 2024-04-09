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

// -----

func.func @dynamic_memref_copy(%arg0: memref<?x3xf16>, %arg1: memref<?x3xf16>) -> memref<?x15xf16> {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x3xf16>
  %alloc = memref.alloc(%dim) : memref<?x15xf16>
  %subview = memref.subview %alloc[0, 0] [%dim, 3] [1, 1] : memref<?x15xf16> to memref<?x3xf16, strided<[15, 1]>>
  memref.copy %arg0, %subview : memref<?x3xf16> to memref<?x3xf16, strided<[15, 1]>>
  %subview_0 = memref.subview %alloc[0, 3] [%dim, 3] [1, 1] : memref<?x15xf16> to memref<?x3xf16, strided<[15, 1], offset: 3>>
  memref.copy %arg1, %subview_0 : memref<?x3xf16> to memref<?x3xf16, strided<[15, 1], offset: 3>>
  return %alloc : memref<?x15xf16>
}
// CHECK-LABEL: func.func private @memref_copy_kernel
//   CHECK-SAME: attributes {test}
//   CHECK: memref.dim
//   CHECK-NEXT: memref.subview
//   CHECK-NEXT: linalg.generic
// CHECK-LABEL: func.func private @memref_copy_kernel_0
//   CHECK-SAME: attributes {test}
//   CHECK: memref.dim
//   CHECK-NEXT: memref.subview
//   CHECK-NEXT: linalg.generic
// CHECK-LABEL: func.func @dynamic_memref_copy
//   CHECK: call @memref_copy_kernel_0
//   CHECK-NEXT: call @memref_copy_kernel
