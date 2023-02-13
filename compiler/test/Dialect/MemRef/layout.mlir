// RUN: byteir-opt -apply-memref-affine-layout %s | FileCheck %s

// CHECK-DAG: [[MAP_0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG: [[MAP_1:.*]] = affine_map<(d0, d1) -> (d1, d0)>

#map0 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func.func @test_map_attr
func.func @test_map_attr(%arg0: memref<100x?xf32>, %arg1: memref<100x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c1 : memref<100x?xf32>
  %1 = memref.alloc(%0) {layout = affine_map<(d0, d1) -> (d0 * 32 + d1)>} : memref<100x?xf32, 1>
// CHECK: %alloc = memref.alloc(%dim) : memref<100x?xf32, [[MAP_0]], 1>
  %2 = memref.load %arg0[%c0, %c0] : memref<100x?xf32>
  %3 = memref.load %1[%c1, %c0] : memref<100x?xf32, 1>
// CHECK: memref.load %alloc[%c1, %c0] : memref<100x?xf32, [[MAP_0]], 1>
  %4 = arith.addf %2, %3 : f32
  memref.store %4, %arg1[%c0, %c1] : memref<100x?xf32>
  return
}

// CHECK-LABEL: func.func @test_registered_affine_str_attr
func.func @test_registered_affine_str_attr(%arg0: memref<100x?xf32>, %arg1: memref<100x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c1 : memref<100x?xf32>
  %1 = memref.alloc(%0) {layout = "test_affine_layout"} : memref<100x?xf32, 1>
// CHECK: %alloc = memref.alloc(%dim) : memref<100x?xf32, [[MAP_1]], 1>
  %2 = memref.load %arg0[%c0, %c0] : memref<100x?xf32>
  %3 = memref.load %1[%c1, %c0] : memref<100x?xf32, 1>
// CHECK: memref.load %alloc[%c1, %c0] : memref<100x?xf32, [[MAP_1]], 1>
  %4 = arith.addf %2, %3 : f32
  memref.store %4, %arg1[%c0, %c1] : memref<100x?xf32>
  return
}

// CHECK-LABEL: func.func @test_non_registered_str_attr
func.func @test_non_registered_str_attr(%arg0: memref<100x?xf32>, %arg1: memref<100x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c1 : memref<100x?xf32>
  %1 = memref.alloc(%0) {layout = "other_layout"} : memref<100x?xf32, 1>
// CHECK: memref.alloc(%dim) {layout = "other_layout"} : memref<100x?xf32, 1>
  %2 = memref.load %arg0[%c0, %c0] : memref<100x?xf32>
  %3 = memref.load %1[%c1, %c0] : memref<100x?xf32, 1>
  %4 = arith.addf %2, %3 : f32
  memref.store %4, %arg1[%c0, %c1] : memref<100x?xf32>
  return
}

