// RUN: byteir-opt %s --allow-unregistered-dialect -remove-copy -cse -canonicalize-ext -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @max_pool
func.func @max_pool(%arg0: memref<4x126x126x16xf32>) -> memref<4x63x63x16xf32> {
  // CHECK-NOT: memref.copy
  %cst = arith.constant 0xFF800000 : f32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() : memref<2x2xf32>
  %alloc_0 = memref.alloc() : memref<4x63x63x16xf32>
  %0 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %alloc_0) -> (memref<4x63x63x16xf32>) {
    %subview = memref.subview %arg0[%arg1, 0, 0, 0] [1, 126, 126, 16] [1, 1, 1, 1] : memref<4x126x126x16xf32> to memref<1x126x126x16xf32, strided<[254016, 2016, 16, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<1x126x126x16xf32>
    memref.copy %subview, %alloc_1 : memref<1x126x126x16xf32, strided<[254016, 2016, 16, 1], offset: ?>> to memref<1x126x126x16xf32>
    %alloc_2 = memref.alloc() : memref<1x63x63x16xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<1x63x63x16xf32>)
    %alloc_3 = memref.alloc() : memref<1x63x63x16xf32>
    memref.copy %alloc_2, %alloc_3 : memref<1x63x63x16xf32> to memref<1x63x63x16xf32>
    linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%alloc_1, %alloc : memref<1x126x126x16xf32>, memref<2x2xf32>) outs(%alloc_3 : memref<1x63x63x16xf32>)
    %alloc_4 = memref.alloc() {alignment = 128 : i64} : memref<4x63x63x16xf32>
    memref.copy %arg2, %alloc_4 : memref<4x63x63x16xf32> to memref<4x63x63x16xf32>
    %subview_5 = memref.subview %alloc_4[%arg1, 0, 0, 0] [1, 63, 63, 16] [1, 1, 1, 1] : memref<4x63x63x16xf32> to memref<1x63x63x16xf32, strided<[63504, 1008, 16, 1], offset: ?>>
    memref.copy %alloc_3, %subview_5 : memref<1x63x63x16xf32> to memref<1x63x63x16xf32, strided<[63504, 1008, 16, 1], offset: ?>>
    scf.yield %alloc_4 : memref<4x63x63x16xf32>
  }
  return %0 : memref<4x63x63x16xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @add2_generic
func.func @add2_generic(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
  // CHECK-NOT: memref.copy
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() : memref<4x4xf32>
  %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %alloc) -> (memref<4x4xf32>) {
    %subview = memref.subview %arg0[%arg3, 0] [1, 4] [1, 1] : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
    %alloc_0 = memref.alloc() : memref<1x4xf32>
    memref.copy %subview, %alloc_0 : memref<1x4xf32, strided<[4, 1], offset: ?>> to memref<1x4xf32>
    %subview_1 = memref.subview %arg1[%arg3, 0] [1, 4] [1, 1] : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1x4xf32>
    memref.copy %subview_1, %alloc_2 : memref<1x4xf32, strided<[4, 1], offset: ?>> to memref<1x4xf32>
    %subview_3 = memref.subview %arg2[%arg3, 0] [1, 4] [1, 1] : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
    %alloc_4 = memref.alloc() : memref<1x4xf32>
    memref.copy %subview_3, %alloc_4 : memref<1x4xf32, strided<[4, 1], offset: ?>> to memref<1x4xf32>
    %alloc_5 = memref.alloc() : memref<1x4xf32>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_0, %alloc_2, %alloc_4 : memref<1x4xf32>, memref<1x4xf32>, memref<1x4xf32>) outs(%alloc_5 : memref<1x4xf32>) {
    ^bb0(%in: f32, %in_8: f32, %in_9: f32, %out: f32):
      %1 = arith.addf %in, %in_8 : f32
      %2 = arith.addf %1, %in_9 : f32
      linalg.yield %2 : f32
    }
    %alloc_6 = memref.alloc() {alignment = 128 : i64} : memref<4x4xf32>
    memref.copy %arg4, %alloc_6 : memref<4x4xf32> to memref<4x4xf32>
    %subview_7 = memref.subview %alloc_6[%arg3, 0] [1, 4] [1, 1] : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
    memref.copy %alloc_5, %subview_7 : memref<1x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
    scf.yield %alloc_6 : memref<4x4xf32>
  }
  return %0 : memref<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @fuse_2_matmul_add
func.func @fuse_2_matmul_add(%arg0: memref<1024x32xf32>, %arg1: memref<32x512xf32>, %arg2: memref<1024x32xf32>, %arg3: memref<32x512xf32>) -> memref<1024x512xf32> {
  // CHECK-NOT: memref.copy
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %alloc = memref.alloc() : memref<1024x512xf32>
  %0 = scf.for %arg4 = %c0 to %c512 step %c8 iter_args(%arg5 = %alloc) -> (memref<1024x512xf32>) {
    %1 = scf.for %arg6 = %c0 to %c1024 step %c4 iter_args(%arg7 = %arg5) -> (memref<1024x512xf32>) {
      %subview = memref.subview %arg0[%arg6, 0] [4, 32] [1, 1] : memref<1024x32xf32> to memref<4x32xf32, strided<[32, 1], offset: ?>>
      %alloc_0 = memref.alloc() : memref<4x32xf32>
      memref.copy %subview, %alloc_0 : memref<4x32xf32, strided<[32, 1], offset: ?>> to memref<4x32xf32>
      %subview_1 = memref.subview %arg1[0, %arg4] [32, 8] [1, 1] : memref<32x512xf32> to memref<32x8xf32, strided<[512, 1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<32x8xf32>
      memref.copy %subview_1, %alloc_2 : memref<32x8xf32, strided<[512, 1], offset: ?>> to memref<32x8xf32>
      %alloc_3 = memref.alloc() : memref<4x8xf32>
      %alloc_4 = memref.alloc() : memref<4x8xf32>
      memref.copy %alloc_3, %alloc_4 : memref<4x8xf32> to memref<4x8xf32>
      linalg.matmul ins(%alloc_0, %alloc_2 : memref<4x32xf32>, memref<32x8xf32>) outs(%alloc_4 : memref<4x8xf32>)
      %subview_5 = memref.subview %arg2[%arg6, 0] [4, 32] [1, 1] : memref<1024x32xf32> to memref<4x32xf32, strided<[32, 1], offset: ?>>
      %alloc_6 = memref.alloc() : memref<4x32xf32>
      memref.copy %subview_5, %alloc_6 : memref<4x32xf32, strided<[32, 1], offset: ?>> to memref<4x32xf32>
      %subview_7 = memref.subview %arg3[0, %arg4] [32, 8] [1, 1] : memref<32x512xf32> to memref<32x8xf32, strided<[512, 1], offset: ?>>
      %alloc_8 = memref.alloc() : memref<32x8xf32>
      memref.copy %subview_7, %alloc_8 : memref<32x8xf32, strided<[512, 1], offset: ?>> to memref<32x8xf32>
      %alloc_9 = memref.alloc() : memref<4x8xf32>
      %alloc_10 = memref.alloc() : memref<4x8xf32>
      memref.copy %alloc_9, %alloc_10 : memref<4x8xf32> to memref<4x8xf32>
      linalg.matmul ins(%alloc_6, %alloc_8 : memref<4x32xf32>, memref<32x8xf32>) outs(%alloc_10 : memref<4x8xf32>)
      %alloc_11 = memref.alloc() : memref<4x8xf32>
      linalg.elemwise_binary ins(%alloc_4, %alloc_10 : memref<4x8xf32>, memref<4x8xf32>) outs(%alloc_11 : memref<4x8xf32>)
      %alloc_12 = memref.alloc() {alignment = 128 : i64} : memref<1024x512xf32>
      memref.copy %arg7, %alloc_12 : memref<1024x512xf32> to memref<1024x512xf32>
      %subview_13 = memref.subview %alloc_12[%arg6, %arg4] [4, 8] [1, 1] : memref<1024x512xf32> to memref<4x8xf32, strided<[512, 1], offset: ?>>
      memref.copy %alloc_11, %subview_13 : memref<4x8xf32> to memref<4x8xf32, strided<[512, 1], offset: ?>>
      scf.yield %alloc_12 : memref<1024x512xf32>
    } {__byteir_parallel__}
    scf.yield %1 : memref<1024x512xf32>
  } {__byteir_parallel__}
  return %0 : memref<1024x512xf32>
}

// -----

// CHECK-LABEL: func.func @fuse_dot_attention
func.func @fuse_dot_attention(%arg0: memref<1024x32xf32>, %arg1: memref<32x512xf32>, %arg2: memref<512x32xf32>) -> memref<1024x32xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %alloc = memref.alloc() : memref<1024x32xf32>
  %alloc_0 = memref.alloc() : memref<1024xf32>
  %alloc_1 = memref.alloc() : memref<1024xf32>
  %0:3 = scf.for %arg3 = %c0 to %c512 step %c8 iter_args(%arg4 = %alloc, %arg5 = %alloc_0, %arg6 = %alloc_1) -> (memref<1024x32xf32>, memref<1024xf32>, memref<1024xf32>) {
    %1:3 = scf.for %arg7 = %c0 to %c1024 step %c4 iter_args(%arg8 = %arg4, %arg9 = %arg5, %arg10 = %arg6) -> (memref<1024x32xf32>, memref<1024xf32>, memref<1024xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [4, 32] [1, 1] : memref<1024x32xf32> to memref<4x32xf32, strided<[32, 1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<4x32xf32>
      memref.copy %subview, %alloc_2 : memref<4x32xf32, strided<[32, 1], offset: ?>> to memref<4x32xf32>
      %subview_3 = memref.subview %arg1[0, %arg3] [32, 8] [1, 1] : memref<32x512xf32> to memref<32x8xf32, strided<[512, 1], offset: ?>>
      %alloc_4 = memref.alloc() : memref<32x8xf32>
      memref.copy %subview_3, %alloc_4 : memref<32x8xf32, strided<[512, 1], offset: ?>> to memref<32x8xf32>
      %alloc_5 = memref.alloc() : memref<4x8xf32>
      %alloc_6 = memref.alloc() : memref<4x8xf32>
      memref.copy %alloc_5, %alloc_6 : memref<4x8xf32> to memref<4x8xf32>
      linalg.matmul ins(%alloc_2, %alloc_4 : memref<4x32xf32>, memref<32x8xf32>) outs(%alloc_6 : memref<4x8xf32>)
      %subview_7 = memref.subview %arg9[%arg7] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
      %alloc_8 = memref.alloc() : memref<4xf32>
      memref.copy %subview_7, %alloc_8 : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
      %subview_9 = memref.subview %arg10[%arg7] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
      %alloc_10 = memref.alloc() : memref<4xf32>
      memref.copy %subview_9, %alloc_10 : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
      %alloc_11 = memref.alloc() : memref<4x8xf32>
      %alloc_12 = memref.alloc() : memref<4xf32>
      memref.copy %alloc_8, %alloc_12 : memref<4xf32> to memref<4xf32>
      %alloc_13 = memref.alloc() : memref<4xf32>
      memref.copy %alloc_10, %alloc_13 : memref<4xf32> to memref<4xf32>
      %alloc_14 = memref.alloc() : memref<4xf32>
      linalg_ext.softmax dimension(1) ins(%alloc_6 : memref<4x8xf32>) outs(%alloc_11, %alloc_12, %alloc_13, %alloc_14 : memref<4x8xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
      %subview_15 = memref.subview %arg2[%arg3, 0] [8, 32] [1, 1] : memref<512x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
      %alloc_16 = memref.alloc() : memref<8x32xf32>
      memref.copy %subview_15, %alloc_16 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32>
      %subview_17 = memref.subview %arg8[%arg7, 0] [4, 32] [1, 1] : memref<1024x32xf32> to memref<4x32xf32, strided<[32, 1], offset: ?>>
      %alloc_18 = memref.alloc() : memref<4x32xf32>
      memref.copy %subview_17, %alloc_18 : memref<4x32xf32, strided<[32, 1], offset: ?>> to memref<4x32xf32>
      %alloc_19 = memref.alloc() : memref<4x4xf32>
      linalg_ext.diag ins(%alloc_14 : memref<4xf32>) outs(%alloc_19 : memref<4x4xf32>)
      %alloc_20 = memref.alloc() : memref<4x32xf32>
      %alloc_21 = memref.alloc() : memref<4x32xf32>
      memref.copy %alloc_20, %alloc_21 : memref<4x32xf32> to memref<4x32xf32>
      linalg.matmul ins(%alloc_19, %alloc_18 : memref<4x4xf32>, memref<4x32xf32>) outs(%alloc_21 : memref<4x32xf32>)
      %alloc_22 = memref.alloc() : memref<4x32xf32>
      memref.copy %alloc_21, %alloc_22 : memref<4x32xf32> to memref<4x32xf32>
      linalg.matmul ins(%alloc_11, %alloc_16 : memref<4x8xf32>, memref<8x32xf32>) outs(%alloc_22 : memref<4x32xf32>)
      %alloc_23 = memref.alloc() {alignment = 128 : i64} : memref<1024x32xf32>
      memref.copy %arg8, %alloc_23 : memref<1024x32xf32> to memref<1024x32xf32>
      %subview_24 = memref.subview %alloc_23[%arg7, 0] [4, 32] [1, 1] : memref<1024x32xf32> to memref<4x32xf32, strided<[32, 1], offset: ?>>
      memref.copy %alloc_22, %subview_24 : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1], offset: ?>>
      %alloc_25 = memref.alloc() {alignment = 128 : i64} : memref<1024xf32>
      memref.copy %arg9, %alloc_25 : memref<1024xf32> to memref<1024xf32>
      %subview_26 = memref.subview %alloc_25[%arg7] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
      memref.copy %alloc_12, %subview_26 : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
      %alloc_27 = memref.alloc() {alignment = 128 : i64} : memref<1024xf32>
      memref.copy %arg10, %alloc_27 : memref<1024xf32> to memref<1024xf32>
      %subview_28 = memref.subview %alloc_27[%arg7] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
      // CHECK: memref.copy
      memref.copy %alloc_13, %subview_28 : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
      // CHECK-NOT: memref.copy
      scf.yield %alloc_23, %alloc_25, %alloc_27 : memref<1024x32xf32>, memref<1024xf32>, memref<1024xf32>
    } {__byteir_parallel__}
    scf.yield %1#0, %1#1, %1#2 : memref<1024x32xf32>, memref<1024xf32>, memref<1024xf32>
  }
  return %0#0 : memref<1024x32xf32>
}

// -----


func.func @tile_linalg_matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) -> memref<128x128xf32> {
  // CHECK-NOT: memref.copy
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %0 = scf.for %arg3 = %c0 to %c128 step %c8 iter_args(%arg4 = %arg2) -> (memref<128x128xf32>) {
    %1 = scf.for %arg5 = %c0 to %c128 step %c4 iter_args(%arg6 = %arg4) -> (memref<128x128xf32>) {
      %2 = scf.for %arg7 = %c0 to %c128 step %c2 iter_args(%arg8 = %arg6) -> (memref<128x128xf32>) {
        %subview = memref.subview %arg0[%arg7, %arg3] [2, 8] [1, 1] : memref<128x128xf32> to memref<2x8xf32, strided<[128, 1], offset: ?>>
        %alloc = memref.alloc() : memref<2x8xf32>
        memref.copy %subview, %alloc : memref<2x8xf32, strided<[128, 1], offset: ?>> to memref<2x8xf32>
        %subview_0 = memref.subview %arg1[%arg3, %arg5] [8, 4] [1, 1] : memref<128x128xf32> to memref<8x4xf32, strided<[128, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<8x4xf32>
        memref.copy %subview_0, %alloc_1 : memref<8x4xf32, strided<[128, 1], offset: ?>> to memref<8x4xf32>
        %subview_2 = memref.subview %arg8[%arg7, %arg5] [2, 4] [1, 1] : memref<128x128xf32> to memref<2x4xf32, strided<[128, 1], offset: ?>>
        %alloc_3 = memref.alloc() : memref<2x4xf32>
        memref.copy %subview_2, %alloc_3 : memref<2x4xf32, strided<[128, 1], offset: ?>> to memref<2x4xf32>
        %alloc_4 = memref.alloc() : memref<2x4xf32>
        memref.copy %alloc_3, %alloc_4 : memref<2x4xf32> to memref<2x4xf32>
        linalg.matmul ins(%alloc, %alloc_1 : memref<2x8xf32>, memref<8x4xf32>) outs(%alloc_4 : memref<2x4xf32>)
        %alloc_5 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
        memref.copy %arg8, %alloc_5 : memref<128x128xf32> to memref<128x128xf32>
        %subview_6 = memref.subview %alloc_5[%arg7, %arg5] [2, 4] [1, 1] : memref<128x128xf32> to memref<2x4xf32, strided<[128, 1], offset: ?>>
        memref.copy %alloc_4, %subview_6 : memref<2x4xf32> to memref<2x4xf32, strided<[128, 1], offset: ?>>
        scf.yield %alloc_5 : memref<128x128xf32>
      }
      scf.yield %2 : memref<128x128xf32>
    }
    scf.yield %1 : memref<128x128xf32>
  }
  return %0 : memref<128x128xf32>
}

// -----

// CHECK-LABEL: func.func @softmax
func.func @softmax(%arg0: memref<1024x64xf32>) -> memref<1024x64xf32> {
  // CHECK-NOT: memref.copy
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1024 = arith.constant 1024 : index
  %alloc = memref.alloc() : memref<1024x64xf32>
  %alloc_0 = memref.alloc() : memref<1024xf32>
  %alloc_1 = memref.alloc() : memref<1024xf32>
  %alloc_2 = memref.alloc() : memref<1024xf32>
  %0:4 = scf.for %arg1 = %c0 to %c1024 step %c4 iter_args(%arg2 = %alloc, %arg3 = %alloc_0, %arg4 = %alloc_1, %arg5 = %alloc_2) -> (memref<1024x64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) {
    %subview = memref.subview %arg0[%arg1, 0] [4, 64] [1, 1] : memref<1024x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<4x64xf32>
    memref.copy %subview, %alloc_3 : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32>
    %subview_4 = memref.subview %arg3[%arg1] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
    %alloc_5 = memref.alloc() : memref<4xf32>
    memref.copy %subview_4, %alloc_5 : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
    %subview_6 = memref.subview %arg4[%arg1] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
    %alloc_7 = memref.alloc() : memref<4xf32>
    memref.copy %subview_6, %alloc_7 : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
    %alloc_8 = memref.alloc() : memref<4x64xf32>
    %alloc_9 = memref.alloc() : memref<4xf32>
    memref.copy %alloc_5, %alloc_9 : memref<4xf32> to memref<4xf32>
    %alloc_10 = memref.alloc() : memref<4xf32>
    memref.copy %alloc_7, %alloc_10 : memref<4xf32> to memref<4xf32>
    %alloc_11 = memref.alloc() : memref<4xf32>
    linalg_ext.softmax dimension(1) ins(%alloc_3 : memref<4x64xf32>) outs(%alloc_8, %alloc_9, %alloc_10, %alloc_11 : memref<4x64xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
    %alloc_12 = memref.alloc() {alignment = 128 : i64} : memref<1024x64xf32>
    memref.copy %arg2, %alloc_12 : memref<1024x64xf32> to memref<1024x64xf32>
    %subview_13 = memref.subview %alloc_12[%arg1, 0] [4, 64] [1, 1] : memref<1024x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
    memref.copy %alloc_8, %subview_13 : memref<4x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
    %alloc_14 = memref.alloc() {alignment = 128 : i64} : memref<1024xf32>
    memref.copy %arg3, %alloc_14 : memref<1024xf32> to memref<1024xf32>
    %subview_15 = memref.subview %alloc_14[%arg1] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %alloc_9, %subview_15 : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %alloc_16 = memref.alloc() {alignment = 128 : i64} : memref<1024xf32>
    memref.copy %arg4, %alloc_16 : memref<1024xf32> to memref<1024xf32>
    %subview_17 = memref.subview %alloc_16[%arg1] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %alloc_10, %subview_17 : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %alloc_18 = memref.alloc() {alignment = 128 : i64} : memref<1024xf32>
    memref.copy %arg5, %alloc_18 : memref<1024xf32> to memref<1024xf32>
    %subview_19 = memref.subview %alloc_18[%arg1] [4] [1] : memref<1024xf32> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %alloc_11, %subview_19 : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
    scf.yield %alloc_12, %alloc_14, %alloc_16, %alloc_18 : memref<1024x64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>
  }
  return %0#0 : memref<1024x64xf32>
}

// -----

// CHECK-LABEL: func.func @copy_collapse_shape
func.func @copy_collapse_shape(%arg0: memref<90x10xf32>) -> memref<90xf32> {
  %subview = memref.subview %arg0[0, 5] [90, 1] [1, 1] : memref<90x10xf32> to memref<90x1xf32, strided<[10, 1], offset: 5>>
  %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<90x1xf32, strided<[10, 1], offset: 5>> into memref<90xf32, strided<[10], offset: 5>>
  %alloc = memref.alloc() : memref<90xf32>
  // CHECK: memref.copy
  memref.copy %collapse_shape, %alloc : memref<90xf32, strided<[10], offset: 5>> to memref<90xf32>
  return %alloc : memref<90xf32>
}

// -----

// CHECK-LABEL: func.func @transpose_split
func.func @transpose_split(%arg0: memref<11x13x15x17xf32>) -> memref<11x15x17x13xf32> {
  // CHECK-NOT: memref.copy
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

// -----

// CHECK-LABEL: func.func @view_of_view
//   CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<11x15x17x13xf32>
//   CHECK-NEXT: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]][0, 0, 16, 0] [11, 15, 1, 8] [1, 1, 1, 1]
//   CHECK-NEXT: "foo.bar"(%[[SUBVIEW]])
//   CHECK-NEXT: return %[[ALLOC]] : memref<11x15x17x13xf32>
func.func @view_of_view() -> memref<11x15x17x13xf32> {
  %dst = memref.alloc() : memref<11x15x17x13xf32>
  %src = memref.alloc() : memref<11x15x17x8xf32>
  %src_sub = memref.subview %src[0, 0, 16, 0] [11, 15, 1, 8] [1, 1, 1, 1] : memref<11x15x17x8xf32> to memref<11x15x1x8xf32, strided<[2040, 136, 8, 1], offset: 128>>
  "foo.bar"(%src_sub) : (memref<11x15x1x8xf32, strided<[2040, 136, 8, 1], offset: 128>>) -> ()
  %dst_sub = memref.subview %dst[0, 0, 0, 0] [11, 15, 17, 8] [1, 1, 1, 1] : memref<11x15x17x13xf32> to memref<11x15x17x8xf32, strided<[3315, 221, 13, 1]>>
  memref.copy %src, %dst_sub : memref<11x15x17x8xf32> to memref<11x15x17x8xf32, strided<[3315, 221, 13, 1]>>
  return %dst : memref<11x15x17x13xf32>
}

// -----

func.func @cannot_remove_copy_0() -> memref<16xf32> {
  %src = memref.alloc() : memref<4x8xf32>
  %src_sub = memref.subview %src[0, 4] [4, 4] [1, 1] : memref<4x8xf32> to memref<4x4xf32, strided<[8, 1], offset: 4>>
  "foo.bar"(%src_sub) : (memref<4x4xf32, strided<[8, 1], offset: 4>>) -> ()
  %dst = memref.alloc() : memref<4x4xf32>
  memref.copy %src_sub, %dst : memref<4x4xf32, strided<[8, 1], offset: 4>> to memref<4x4xf32>
  %dst_collapsed = memref.collapse_shape %dst [[0, 1]] : memref<4x4xf32> into memref<16xf32>
  return %dst_collapsed: memref<16xf32>
}
// CHECK-LABEL: cannot_remove_copy_0
//   CHECK: memref.copy

// -----

func.func @cannot_remove_copy_1() -> memref<4x4xf32> {
  %src = memref.alloc() : memref<4x8xf32>
  %src_sub = memref.subview %src[0, 4] [4, 4] [1, 1] : memref<4x8xf32> to memref<4x4xf32, strided<[8, 1], offset: 4>>
  "foo.bar"(%src_sub) : (memref<4x4xf32, strided<[8, 1], offset: 4>>) -> ()
  %alloc = memref.alloc() : memref<4x4xf32>
  memref.copy %src_sub, %alloc : memref<4x4xf32, strided<[8, 1], offset: 4>> to memref<4x4xf32>
  %dst = memref.alloc() : memref<4x4xf32>
  "lmhlo.log"(%alloc, %dst) : (memref<4x4xf32>, memref<4x4xf32>) -> ()
  return %dst: memref<4x4xf32>
}
// CHECK-LABEL: cannot_remove_copy_1
//   CHECK: memref.copy

// -----
func.func private @foo(%arg0: memref<4x8xf32>, %arg1: memref<4x8xf32>) -> memref<4x8xf32>

func.func @copy_then_call() -> memref<4x8xf32> {
  %cst_0 = arith.constant 0.0 : f32
  %cst_1 = arith.constant 1.0 : f32
  %cst_2 = arith.constant 2.0 : f32
  %src = memref.alloc() : memref<4x8xf32>
  linalg.fill ins(%cst_0: f32) outs(%src: memref<4x8xf32>)
  %src0 = memref.alloc() : memref<4x4xf32>
  linalg.fill ins(%cst_1: f32) outs(%src0: memref<4x4xf32>)
  %src1 = memref.alloc() : memref<4x4xf32>
  linalg.fill ins(%cst_2: f32) outs(%src1: memref<4x4xf32>)

  %arg0 = memref.alloc() : memref<4x8xf32>
  memref.copy %src, %arg0 : memref<4x8xf32> to memref<4x8xf32>
  %subview0 = memref.subview %arg0[0, 0] [4, 4] [1, 1] : memref<4x8xf32> to memref<4x4xf32, strided<[8, 1], offset: 0>>
  memref.copy %src0, %subview0 : memref<4x4xf32> to memref<4x4xf32, strided<[8, 1], offset: 0>>

  %arg1 = memref.alloc() : memref<4x8xf32>
  memref.copy %src, %arg1 : memref<4x8xf32> to memref<4x8xf32>
  %subview1 = memref.subview %arg1[0, 4] [4, 4] [1, 1] : memref<4x8xf32> to memref<4x4xf32, strided<[8, 1], offset: 4>>
  memref.copy %src1, %subview1 : memref<4x4xf32> to memref<4x4xf32, strided<[8, 1], offset: 4>>

  %0 = call @foo(%arg0, %arg1) : (memref<4x8xf32>, memref<4x8xf32>) -> memref<4x8xf32>

  return %0 : memref<4x8xf32>
}
// CHECK-LABEL: copy_then_call
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<4x8xf32>
// CHECK: linalg.fill {{.*}} outs(%[[SRC]] : memref<4x8xf32>)
// CHECK: %[[SRC0:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK: linalg.fill {{.*}} outs(%[[SRC0]] : memref<4x4xf32>)
// CHECK: %[[SRC1:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK: linalg.fill {{.*}} outs(%[[SRC1]] : memref<4x4xf32>)
// CHECK: %[[ARG0:.*]] = memref.alloc() : memref<4x8xf32>
// CHECK: memref.copy %[[SRC]], %[[ARG0]]
// CHECK: %[[SUBVIEW0:.*]] = memref.subview %[[ARG0]]
// CHECK: memref.copy %[[SRC0]], %[[SUBVIEW0]]
// CHECK: %[[SUBVIEW1:.*]] = memref.subview %[[SRC]]
// CHECK: memref.copy %[[SRC1]], %[[SUBVIEW1]]
// CHECK: call @foo(%[[ARG0]], %[[SRC]])
