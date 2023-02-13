// RUN: byteir-opt %s -rewrite-affine-to-memref | FileCheck %s

// CHECK-LABEL: fusion_broadcast
func.func @fusion_broadcast(%arg0: memref<6x12x96xf32>, %arg1: memref<6x12x96x96xf32>) -> memref<6x12x96x96xf32> {
  %0 = memref.alloc() : memref<6x12x96x96xf32>
  affine.for %arg2 = 0 to 6 {
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 96 {
        affine.for %arg5 = 0 to 96 {
          %1 = affine.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<6x12x96x96xf32>
          %2 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<6x12x96xf32>
          // CHECK: memref.load
          // CHECK-NEXT: memref.load
          %3 = arith.subf %1, %2 : f32
          %4 = math.exp %3 : f32
          affine.store %4, %0[%arg2, %arg3, %arg4, %arg5] : memref<6x12x96x96xf32>
          // CHECK: memref.store
        }
      }
    }
  }
  return %0 : memref<6x12x96x96xf32>
}
