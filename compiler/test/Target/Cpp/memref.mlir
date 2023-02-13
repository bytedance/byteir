// RUN: byteir-translate -emit-cpp %s | FileCheck %s

// CHECK-LABEL: func_memref
// CHECK-SAME: (float* [[V1:[^ ]*]], float* [[V2:[^ ]*]])
func.func @func_memref(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>) {
// CHECK-NEXT: size_t [[V3:[^ ]*]] = 1;
  %c1= arith.constant 1 : index
// CHECK-NEXT: size_t [[V4:[^ ]*]] = 2;
  %c2= arith.constant 2 : index
// CHECK-NEXT: float [[V5:[^ ]*]]
  %0 = memref.load %arg0[%c1] : memref<4xf32>
// CHECK-NEXT: float [[V6:[^ ]*]]
  %1 = memref.load %arg1[%c1] : memref<4xf32>
// CHECK-NEXT: float [[V7:[^ ]*]]
  %2 = arith.addf %0, %1 : f32
// CHECK-NEXT: [[V7]]
  memref.store %2, %arg0[%c1] : memref<4xf32>
// CHECK-NEXT: return;
  return 
}