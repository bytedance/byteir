// RUN: byteir-opt %s --host-opt --byre-opt | FileCheck %s

// CHECK-LABEL: func.func @Unknown

module {
  func.func private @Unknown0(%arg0: memref<i64>, %arg1: memref<i64>) -> memref<1x97xf32> attributes {__byteir_hlo_aggressive_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.32830644E-10 : f32
    %c12345_i32 = arith.constant 12345 : i32
    %c1103515245_i32 = arith.constant 1103515245 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c97 = arith.constant 97 : index
    %alloc = memref.alloc() : memref<1x97xf32>
    scf.for %arg2 = %c0 to %c97 step %c1 {
      %0 = memref.load %arg0[] : memref<i64>
      %1 = memref.load %arg1[] : memref<i64>
      %2 = arith.trunci %0 : i64 to i32
      %3 = arith.trunci %1 : i64 to i32
      %4 = arith.addi %2, %3 : i32
      %5 = arith.muli %4, %c1103515245_i32 : i32
      %6 = arith.addi %5, %c12345_i32 : i32
      %7 = arith.index_cast %arg2 : index to i32
      %8 = arith.addi %7, %6 : i32
      %9 = arith.muli %8, %c1103515245_i32 : i32
      %10 = arith.addi %9, %c12345_i32 : i32
      %11 = arith.uitofp %10 : i32 to f32
      %12 = arith.mulf %11, %cst_0 : f32
      %13 = arith.addf %12, %cst : f32
      memref.store %13, %alloc[%c0, %arg2] : memref<1x97xf32>
    }
    return %alloc : memref<1x97xf32>
  }
  func.func @main() -> memref<1x97xf32> attributes {__placeholder__byre.entry_point} {
    %alloc = memref.alloc() : memref<i64>
    byre.compute @GetSeed(%alloc) {memory_effects = [2 : i32]} : memref<i64>
    %alloc_0 = memref.alloc() : memref<i64>
    byre.compute @NextOffset(%alloc_0) {memory_effects = [2 : i32]} : memref<i64>
    %0 = call @Unknown0(%alloc, %alloc_0) : (memref<i64>, memref<i64>) -> memref<1x97xf32>
    return %0 : memref<1x97xf32>
  }
}