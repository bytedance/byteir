// RUN: byteir-opt %s -unroll="unroll-factor=2" -cse | FileCheck %s -check-prefix=UNROLL2
// RUN: byteir-opt %s -unroll="unroll-full" -cse | FileCheck %s -check-prefix=UNROLLFULL

func.func @anchored(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i0 = %lb to %ub step %step {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  } {__byteir_unroll__}
  return
}
// UNROLL2-LABEL: func.func @anchored
// UNROLL2: scf.for
// UNROLL2:   memref.store
// UNROLL2:   memref.store

// UNROLLFULL-LABEL: func.func @anchored
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store

func.func @anchored_2_loop(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i0 = %lb to %ub step %step {
    scf.for %i1 = %lb to %ub step %step {
      memref.store %0, %arg0[%i1] : memref<?xf32>
    } {__byteir_unroll__}
  } {__byteir_unroll__}
  return
}
// UNROLL2-LABEL: func.func @anchored_2_loop
// UNROLL2: scf.for
// UNROLL2:   scf.for
// UNROLL2:     memref.store
// UNROLL2:     memref.store
// UNROLL2:   scf.for
// UNROLL2:     memref.store
// UNROLL2:     memref.store

// UNROLLFULL-LABEL: func.func @anchored_2_loop
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
// UNROLLFULL: memref.store
