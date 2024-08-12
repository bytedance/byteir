// RUN: byteir-opt -memref-to-byre --canonicalize --split-input-file %s | FileCheck %s

func.func @test_copy(%arg0: memref<4xf32, "cpu">, %arg1: memref<4xf32, "gpu">) attributes {__placeholder__byre.entry_point} {
  memref.copy %arg0, %arg1 : memref<4xf32, "cpu"> to memref<4xf32, "gpu">
  return
}
// CHECK: byre.copy
//   CHECK-SAME {callee = "cpu2gpu"} : memref<4xf32, "cpu">, memref<4xf32, "gpu">

// -----

func.func @test_copy_stride$0(%arg0: memref<100x2xf32, strided<[2, 1], offset: 10>, "cpu">, %arg1: memref<100x2xf32, strided<[2, 1], offset: 20>, "gpu">) attributes {__placeholder__byre.entry_point} {
  memref.copy %arg0, %arg1 : memref<100x2xf32, strided<[2, 1], offset: 10>, "cpu"> to memref<100x2xf32, strided<[2, 1], offset: 20>, "gpu">
  return
}
// CHECK-LABEL: func.func @test_copy_stride$0
// CHECK-NEXT:  %0 = "byre.alias"(%arg0) <{offset = 10 : i64}> : (memref<100x2xf32, strided<[2, 1], offset: 10>, "cpu">) -> memref<100x2xf32, "cpu">
// CHECK-NEXT:  %1 = "byre.alias"(%arg1) <{offset = 20 : i64}> : (memref<100x2xf32, strided<[2, 1], offset: 20>, "gpu">) -> memref<100x2xf32, "gpu">
// CHECK-NEXT:  byre.copy(%0, %1) {callee = "cpu2gpu"} : memref<100x2xf32, "cpu">, memref<100x2xf32, "gpu">
// CHECK-NEXT:  return

// -----

func.func @test_copy_stride$1(%arg0: memref<1xf16, strided<[1]>, "cpu">, %arg1: memref<1xf16, "cpu">) attributes {__placeholder__byre.entry_point} {
  memref.copy %arg0, %arg1 : memref<1xf16, strided<[1]>, "cpu"> to memref<1xf16, "cpu">
  return
}
// CHECK-LABEL: func.func @test_copy_stride$1
// CHECK-NEXT:  %0 = "byre.alias"(%arg0) <{offset = 0 : i64}> : (memref<1xf16, strided<[1]>, "cpu">) -> memref<1xf16, "cpu">
// CHECK-NEXT:  byre.copy(%0, %arg1) {callee = "cpu2cpu"} : memref<1xf16, "cpu">, memref<1xf16, "cpu">
// CHECK-NEXT:  return

// -----

func.func @test_view(%arg0 : memref<32xi8>) -> memref<4xf32> attributes {__placeholder__byre.entry_point} {
  %c16 = arith.constant 16 : index
  %1 = memref.view %arg0[%c16][] : memref<32xi8> to memref<4xf32>
  return %1 : memref<4xf32>
}
// CHECK: byre.alias
//   CHECK-SAME: offset = 16

// -----

func.func @test_collapse_shape(%arg0 : memref<2x2xf32>) -> memref<4xf32> attributes {__placeholder__byre.entry_point} {
  %0 = memref.collapse_shape %arg0[[0, 1]] : memref<2x2xf32> into memref<4xf32>
  return %0 : memref<4xf32>
}
// CHECK: byre.alias
//   CHECK-SAME: offset = 0

// -----

func.func @test_expand_shape(%arg0 : memref<4xf32>) -> memref<2x2xf32> attributes {__placeholder__byre.entry_point} {
  %0 = memref.expand_shape %arg0[[0, 1]] output_shape [2, 2] : memref<4xf32> into memref<2x2xf32>
  return %0 : memref<2x2xf32>
}
// CHECK: byre.alias
//   CHECK-SAME: offset = 0

// -----

memref.global "private" constant @constant : memref<4xf32> = dense<0.0>
func.func @test_get_global() -> memref<4xf32> attributes {__placeholder__byre.entry_point} {
  %0 = memref.get_global @constant : memref<4xf32>
  return %0 : memref<4xf32>
}
// CHECK: byre.compute @FillOp
//   CHECK-SAME: dense<0.000000e+00>
