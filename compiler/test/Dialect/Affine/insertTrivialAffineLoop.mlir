// RUN: byteir-opt %s -insert-trivial-affine-loop | FileCheck %s


func.func  @scalar_func(%arg0: memref<f32>) -> memref<f32> {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = memref.alloc() : memref<f32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %1 = affine.load %arg0[] : memref<f32>
  %2 = arith.cmpf une, %1, %cst_0 : f32
  %3 = arith.select %2, %1, %cst : f32
  affine.store %3, %0[] : memref<f32>
  return %0 : memref<f32>
}
// CHECK-LABEL: func.func @scalar_func
// CHECK: affine.for {{.*}} = 0 to 1
// CHECK-NEXT: affine.load
// CHECK-NEXT: arith.cmpf
// CHECK-NEXT: arith.select
// CHECK-NEXT: affine.store 