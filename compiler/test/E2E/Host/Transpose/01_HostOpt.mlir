// RUN: byteir-opt %s --host-opt -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" --byre-opt | FileCheck %s

// CHECK-LABEL: func.func @main

module {
  func.func private @Unknown0(%arg0: memref<1x32x64x64xf32>) -> memref<1x64x64x32xf32> attributes {__byteir_hlo_aggressive_fusion__} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c2048 = arith.constant 2048 : index
    %alloc = memref.alloc() : memref<1x64x64x32xf32>
    scf.for %arg1 = %c0 to %c2048 step %c1 {
      %0 = arith.remsi %arg1, %c8 : index
      %1 = arith.divsi %arg1, %c8 : index
      %2 = arith.remsi %1, %c64 : index
      %3 = arith.divsi %1, %c64 : index
      %4 = arith.muli %0, %c8 : index
      %5 = arith.muli %3, %c8 : index
      %6 = vector.transfer_read %arg0[%c0, %5, %2, %4], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x64xf32>, vector<1x8x1x8xf32>
      %7 = vector.transpose %6, [0, 2, 3, 1] : vector<1x8x1x8xf32> to vector<1x1x8x8xf32>
      vector.transfer_write %7, %alloc[%c0, %2, %4, %5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<1x64x64x32xf32>
    }
    return %alloc : memref<1x64x64x32xf32>
  }
  func.func @main(%arg0: memref<1x32x64x64xf32>) -> memref<1x64x64x32xf32> attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg0) : (memref<1x32x64x64xf32>) -> memref<1x64x64x32xf32>
    return %0 : memref<1x64x64x32xf32>
  }
}