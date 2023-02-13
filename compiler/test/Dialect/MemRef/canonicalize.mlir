// RUN: byteir-opt %s -canonicalize | FileCheck %s


#map0 = affine_map<(d0, d1) -> (d0 * 32 + d1)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>

// CHECK-LABEL: func.func @test_map
func.func @test_map(%arg0: memref<100x?xf32>, %arg1: memref<100x?xf32>, %arg2: index) {
  %c32 = arith.constant 32 : index
  %0 = memref.alloc(%c32) : memref<100x?xf32, #map0, 1>
// CHECK: memref.alloc() : memref<100x32xf32, #map, 1>
  %1 = memref.load %arg0[%arg2, %arg2] : memref<100x?xf32>
  %2 = memref.load %0[%arg2, %arg2] : memref<100x?xf32, #map0, 1>
  %3 = arith.addf %1, %2 : f32
  memref.store %3, %arg1[%arg2, %arg2] : memref<100x?xf32>
  return
}

// CHECK-LABEL: func.func @matmul_tiled_hoist
func.func @matmul_tiled_hoist(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %0 = memref.alloc(%c8) : memref<?x64xf32, 3>
// CHECK-DAG: memref.alloc() : memref<8x64xf32, 3>
  %1 = memref.alloc() : memref<64x64xf32, 4>
  %2 = memref.alloc(%c8) : memref<?x64xf32, 5>
// CHECK-DAG: memref.alloc() : memref<8x64xf32, 5>
  scf.for %arg3 = %c0 to %c128 step %c8 {
    %3 = memref.subview %arg0[%arg3, 0] [%c8, 64] [1, 1] : memref<128x64xf32> to memref<?x64xf32, #map1>
    %4 = memref.subview %arg2[%arg3, 0] [%c8, 64] [1, 1] : memref<128x64xf32> to memref<?x64xf32, #map1>
    linalg.copy ins(%3: memref<?x64xf32, #map1>) outs(%0: memref<?x64xf32, 3>)
    linalg.copy ins(%arg1: memref<64x64xf32>) outs(%1: memref<64x64xf32, 4>) 
    linalg.matmul {anchor} ins(%0, %1 : memref<?x64xf32, 3>, memref<64x64xf32, 4>) outs(%2 : memref<?x64xf32, 5>)
    linalg.copy ins(%2: memref<?x64xf32, 5>) outs(%4: memref<?x64xf32, #map1>)
  }
  return
}

// CHECK-LABEL: func.func @matmul_tiled_non_hoist
func.func @matmul_tiled_non_hoist(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %c128 step %c8 {
    %0 = memref.subview %arg0[%arg3, 0] [%c8, 64] [1, 1] : memref<128x64xf32> to memref<?x64xf32, #map1>
    %1 = memref.subview %arg2[%arg3, 0] [%c8, 64] [1, 1] : memref<128x64xf32> to memref<?x64xf32, #map1>
    %2 = memref.alloc(%c8) : memref<?x64xf32, 3>
// CHECK-DAG: memref.alloc() : memref<8x64xf32, 3>
    linalg.copy ins(%0: memref<?x64xf32, #map1>) outs(%2: memref<?x64xf32, 3>)
    %3 = memref.alloc() : memref<64x64xf32, 4>
    linalg.copy ins(%arg1: memref<64x64xf32>) outs(%3: memref<64x64xf32, 4>)
    %4 = memref.alloc(%c8) : memref<?x64xf32, 5>
// CHECK-DAG: memref.alloc() : memref<8x64xf32, 5>
    linalg.matmul {anchor} ins(%2, %3 : memref<?x64xf32, 3>, memref<64x64xf32, 4>) outs(%4 : memref<?x64xf32, 5>)
    linalg.copy ins(%4: memref<?x64xf32, 5>) outs(%1: memref<?x64xf32, #map1>)
  }
  return
}
