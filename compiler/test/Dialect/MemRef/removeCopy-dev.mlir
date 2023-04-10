// RUN: byteir-opt %s -remove-copy -cse -canonicalize-ext -split-input-file | FileCheck %s

// CHECK-LABEL: transpose_split
func.func @transpose_split(%arg0: memref<11x13x15x17xf32>) -> memref<11x15x17x13xf32> {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c11 = arith.constant 11 : index
  %c15 = arith.constant 15 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<11x15x17x13xf32>
  %subview = memref.subview %arg0[0, 0, 0, 0] [11, 8, 15, 17] [1, 1, 1, 1] : memref<11x13x15x17xf32> to memref<11x8x15x17xf32, strided<[3315, 255, 17, 1]>>
  %alloc_0 = memref.alloc() : memref<11x15x17x8xf32>
  %alloc_1 = memref.alloc() : memref<11x15x16x8xf32>
  scf.for %arg1 = %c0 to %c11 step %c1 {
    scf.for %arg2 = %c0 to %c15 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c8 {
        %0 = vector.transfer_read %arg0[%arg1, %c0, %arg2, %arg3], %cst {in_bounds = [true, true, true, true]} : memref<11x13x15x17xf32>, vector<1x8x1x8xf32>
        %1 = vector.transpose %0, [0, 2, 3, 1] : vector<1x8x1x8xf32> to vector<1x1x8x8xf32>
        vector.transfer_write %1, %alloc_1[%arg1, %arg2, %arg3, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<11x15x16x8xf32>
      }
    }
  }
  %subview_2 = memref.subview %alloc_0[0, 0, 0, 0] [11, 15, 16, 8] [1, 1, 1, 1] : memref<11x15x17x8xf32> to memref<11x15x16x8xf32, strided<[2040, 136, 8, 1]>>
  memref.copy %alloc_1, %subview_2 : memref<11x15x16x8xf32> to memref<11x15x16x8xf32, strided<[2040, 136, 8, 1]>>
  %subview_3 = memref.subview %subview[0, 0, 0, 16] [11, 8, 15, 1] [1, 1, 1, 1] : memref<11x8x15x17xf32, strided<[3315, 255, 17, 1]>> to memref<11x8x15x1xf32, strided<[3315, 255, 17, 1], offset: 16>>
  %subview_4 = memref.subview %alloc_0[0, 0, 16, 0] [11, 15, 1, 8] [1, 1, 1, 1] : memref<11x15x17x8xf32> to memref<11x15x1x8xf32, strided<[2040, 136, 8, 1], offset: 128>>
  scf.for %arg1 = %c0 to %c11 step %c1 {
    scf.for %arg2 = %c0 to %c15 step %c1 {
      %subview_16 = memref.subview %subview_3[%arg1, 0, %arg2, 0] [1, 8, 1, 1] [1, 1, 1, 1] : memref<11x8x15x1xf32, strided<[3315, 255, 17, 1], offset: 16>> to memref<1x8x1x1xf32, strided<[3315, 255, 17, 1], offset: ?>>
      %subview_17 = memref.subview %subview_4[%arg1, %arg2, 0, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<11x15x1x8xf32, strided<[2040, 136, 8, 1], offset: 128>> to memref<1x1x1x8xf32, strided<[2040, 136, 8, 1], offset: ?>>
      %0 = vector.transfer_read %subview_16[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, false]} : memref<1x8x1x1xf32, strided<[3315, 255, 17, 1], offset: ?>>, vector<1x8x1x8xf32>
      %1 = vector.transpose %0, [0, 2, 3, 1] : vector<1x8x1x8xf32> to vector<1x1x8x8xf32>
      vector.transfer_write %1, %subview_17[%c0, %c0, %c0, %c0] {in_bounds = [true, true, false, true]} : vector<1x1x8x8xf32>, memref<1x1x1x8xf32, strided<[2040, 136, 8, 1], offset: ?>>
      %subview_18 = memref.subview %subview_4[%arg1, %arg2, 0, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<11x15x1x8xf32, strided<[2040, 136, 8, 1], offset: 128>> to memref<1x1x1x8xf32, strided<[2040, 136, 8, 1], offset: ?>>
      memref.copy %subview_17, %subview_18 : memref<1x1x1x8xf32, strided<[2040, 136, 8, 1], offset: ?>> to memref<1x1x1x8xf32, strided<[2040, 136, 8, 1], offset: ?>>
    }
  }
  %subview_5 = memref.subview %alloc_0[0, 0, 16, 0] [11, 15, 1, 8] [1, 1, 1, 1] : memref<11x15x17x8xf32> to memref<11x15x1x8xf32, strided<[2040, 136, 8, 1], offset: 128>>
  memref.copy %subview_4, %subview_5 : memref<11x15x1x8xf32, strided<[2040, 136, 8, 1], offset: 128>> to memref<11x15x1x8xf32, strided<[2040, 136, 8, 1], offset: 128>>
  %subview_6 = memref.subview %alloc[0, 0, 0, 0] [11, 15, 17, 8] [1, 1, 1, 1] : memref<11x15x17x13xf32> to memref<11x15x17x8xf32, strided<[3315, 221, 13, 1]>>
  memref.copy %alloc_0, %subview_6 : memref<11x15x17x8xf32> to memref<11x15x17x8xf32, strided<[3315, 221, 13, 1]>>
  %subview_7 = memref.subview %arg0[0, 8, 0, 0] [11, 5, 15, 17] [1, 1, 1, 1] : memref<11x13x15x17xf32> to memref<11x5x15x17xf32, strided<[3315, 255, 17, 1], offset: 2040>>
  %subview_8 = memref.subview %alloc[0, 0, 0, 8] [11, 15, 17, 5] [1, 1, 1, 1] : memref<11x15x17x13xf32> to memref<11x15x17x5xf32, strided<[3315, 221, 13, 1], offset: 8>>
  %subview_9 = memref.subview %subview_7[0, 0, 0, 0] [11, 5, 15, 16] [1, 1, 1, 1] : memref<11x5x15x17xf32, strided<[3315, 255, 17, 1], offset: 2040>> to memref<11x5x15x16xf32, strided<[3315, 255, 17, 1], offset: 2040>>
  %subview_10 = memref.subview %subview_8[0, 0, 0, 0] [11, 15, 16, 5] [1, 1, 1, 1] : memref<11x15x17x5xf32, strided<[3315, 221, 13, 1], offset: 8>> to memref<11x15x16x5xf32, strided<[3315, 221, 13, 1], offset: 8>>
  scf.for %arg1 = %c0 to %c11 step %c1 {
    scf.for %arg2 = %c0 to %c15 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c8 {
        %subview_16 = memref.subview %subview_9[%arg1, 0, %arg2, %arg3] [1, 5, 1, 8] [1, 1, 1, 1] : memref<11x5x15x16xf32, strided<[3315, 255, 17, 1], offset: 2040>> to memref<1x5x1x8xf32, strided<[3315, 255, 17, 1], offset: ?>>
        %subview_17 = memref.subview %subview_10[%arg1, %arg2, %arg3, 0] [1, 1, 8, 5] [1, 1, 1, 1] : memref<11x15x16x5xf32, strided<[3315, 221, 13, 1], offset: 8>> to memref<1x1x8x5xf32, strided<[3315, 221, 13, 1], offset: ?>>
        %0 = vector.transfer_read %subview_16[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, false, true, true]} : memref<1x5x1x8xf32, strided<[3315, 255, 17, 1], offset: ?>>, vector<1x8x1x8xf32>
        %1 = vector.transpose %0, [0, 2, 3, 1] : vector<1x8x1x8xf32> to vector<1x1x8x8xf32>
        vector.transfer_write %1, %subview_17[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, false]} : vector<1x1x8x8xf32>, memref<1x1x8x5xf32, strided<[3315, 221, 13, 1], offset: ?>>
        %subview_18 = memref.subview %subview_10[%arg1, %arg2, %arg3, 0] [1, 1, 8, 5] [1, 1, 1, 1] : memref<11x15x16x5xf32, strided<[3315, 221, 13, 1], offset: 8>> to memref<1x1x8x5xf32, strided<[3315, 221, 13, 1], offset: ?>>
        memref.copy %subview_17, %subview_18 : memref<1x1x8x5xf32, strided<[3315, 221, 13, 1], offset: ?>> to memref<1x1x8x5xf32, strided<[3315, 221, 13, 1], offset: ?>>
      }
    }
  }
  %subview_11 = memref.subview %subview_8[0, 0, 0, 0] [11, 15, 16, 5] [1, 1, 1, 1] : memref<11x15x17x5xf32, strided<[3315, 221, 13, 1], offset: 8>> to memref<11x15x16x5xf32, strided<[3315, 221, 13, 1], offset: 8>>
  memref.copy %subview_10, %subview_11 : memref<11x15x16x5xf32, strided<[3315, 221, 13, 1], offset: 8>> to memref<11x15x16x5xf32, strided<[3315, 221, 13, 1], offset: 8>>
  %subview_12 = memref.subview %subview_7[0, 0, 0, 16] [11, 5, 15, 1] [1, 1, 1, 1] : memref<11x5x15x17xf32, strided<[3315, 255, 17, 1], offset: 2040>> to memref<11x5x15x1xf32, strided<[3315, 255, 17, 1], offset: 2056>>
  %subview_13 = memref.subview %subview_8[0, 0, 16, 0] [11, 15, 1, 5] [1, 1, 1, 1] : memref<11x15x17x5xf32, strided<[3315, 221, 13, 1], offset: 8>> to memref<11x15x1x5xf32, strided<[3315, 221, 13, 1], offset: 216>>
  scf.for %arg1 = %c0 to %c11 step %c1 {
    scf.for %arg2 = %c0 to %c15 step %c1 {
      %subview_16 = memref.subview %subview_12[%arg1, 0, %arg2, 0] [1, 5, 1, 1] [1, 1, 1, 1] : memref<11x5x15x1xf32, strided<[3315, 255, 17, 1], offset: 2056>> to memref<1x5x1x1xf32, strided<[3315, 255, 17, 1], offset: ?>>
      %subview_17 = memref.subview %subview_13[%arg1, %arg2, 0, 0] [1, 1, 1, 5] [1, 1, 1, 1] : memref<11x15x1x5xf32, strided<[3315, 221, 13, 1], offset: 216>> to memref<1x1x1x5xf32, strided<[3315, 221, 13, 1], offset: ?>>
      %0 = vector.transfer_read %subview_16[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, false, true, false]} : memref<1x5x1x1xf32, strided<[3315, 255, 17, 1], offset: ?>>, vector<1x8x1x8xf32>
      %1 = vector.transpose %0, [0, 2, 3, 1] : vector<1x8x1x8xf32> to vector<1x1x8x8xf32>
      vector.transfer_write %1, %subview_17[%c0, %c0, %c0, %c0] {in_bounds = [true, true, false, false]} : vector<1x1x8x8xf32>, memref<1x1x1x5xf32, strided<[3315, 221, 13, 1], offset: ?>>
      %subview_18 = memref.subview %subview_13[%arg1, %arg2, 0, 0] [1, 1, 1, 5] [1, 1, 1, 1] : memref<11x15x1x5xf32, strided<[3315, 221, 13, 1], offset: 216>> to memref<1x1x1x5xf32, strided<[3315, 221, 13, 1], offset: ?>>
      memref.copy %subview_17, %subview_18 : memref<1x1x1x5xf32, strided<[3315, 221, 13, 1], offset: ?>> to memref<1x1x1x5xf32, strided<[3315, 221, 13, 1], offset: ?>>
    }
  }
  %subview_14 = memref.subview %subview_8[0, 0, 16, 0] [11, 15, 1, 5] [1, 1, 1, 1] : memref<11x15x17x5xf32, strided<[3315, 221, 13, 1], offset: 8>> to memref<11x15x1x5xf32, strided<[3315, 221, 13, 1], offset: 216>>
  memref.copy %subview_13, %subview_14 : memref<11x15x1x5xf32, strided<[3315, 221, 13, 1], offset: 216>> to memref<11x15x1x5xf32, strided<[3315, 221, 13, 1], offset: 216>>
  %subview_15 = memref.subview %alloc[0, 0, 0, 8] [11, 15, 17, 5] [1, 1, 1, 1] : memref<11x15x17x13xf32> to memref<11x15x17x5xf32, strided<[3315, 221, 13, 1], offset: 8>>
  memref.copy %subview_8, %subview_15 : memref<11x15x17x5xf32, strided<[3315, 221, 13, 1], offset: 8>> to memref<11x15x17x5xf32, strided<[3315, 221, 13, 1], offset: 8>>
  return %alloc : memref<11x15x17x13xf32>
}