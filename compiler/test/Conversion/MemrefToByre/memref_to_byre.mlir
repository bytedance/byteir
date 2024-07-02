// RUN: byteir-opt -memref-to-byre --split-input-file %s | FileCheck %s

func.func @test_copy(%arg0: memref<4xf32, "cpu">, %arg1: memref<4xf32, "gpu">) attributes {__placeholder__byre.entry_point} {
  memref.copy %arg0, %arg1 : memref<4xf32, "cpu"> to memref<4xf32, "gpu">
  return
}
// CHECK: byre.copy
//   CHECK-SAME {callee = "cpu2gpu"} : memref<4xf32, "cpu">, memref<4xf32, "gpu">

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
