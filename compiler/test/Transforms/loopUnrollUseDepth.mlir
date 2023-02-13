// RUN: byteir-opt %s -unroll="unroll-factor=2 depth=0" -cse | FileCheck %s 

func.func @anchored(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i0 = %lb to %ub step %step {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  }
  return
}
// CHECK-LABEL: func.func @anchored
// CHECK: scf.for
// CHECK:   memref.store
// CHECK:   memref.store

func.func @anchored_2_loop(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i0 = %lb to %ub step %step {
    scf.for %i1 = %lb to %ub step %step {
      memref.store %0, %arg0[%i1] : memref<?xf32>
    }
  }
  return
}
// CHECK-LABEL: func.func @anchored_2_loop
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     memref.store
// CHECK:   scf.for
// CHECK:     memref.store
