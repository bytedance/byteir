// RUN: byteir-opt %s -insert-trivial-scf-loop | FileCheck %s


func.func  @scalar_func(%arg0: memref<f32>) -> memref<f32> {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = memref.alloc() : memref<f32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %1 = memref.load %arg0[] : memref<f32>
  %2 = arith.cmpf une, %1, %cst_0 : f32
  %3 = arith.select %2, %1, %cst : f32
  memref.store %3, %0[] : memref<f32>
  return %0 : memref<f32>
}
// CHECK-LABEL: func.func @scalar_func
// CHECK: scf.for {{.*}} = %c0 to %c1 step %c1
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.cmpf
// CHECK-NEXT: select
// CHECK-NEXT: memref.store 