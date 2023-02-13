// RUN: byteir-opt %s -empty-tensor-to-alloc-tensor -byteir-one-shot-bufferize -cse --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//CHECK-LABEL: func.func @linalg_genric
func.func @linalg_genric(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: bufferization.to_tensor
  // CHECK: memref.alloc()
  %0 = tensor.empty() : tensor<2x128x128xf32>
  // CHECK: linalg.generic
  //   CHECK-SAME: memref<2x128x128xf32>
  // CHECK-NOT: bufferization.to_memref
  %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%0 : tensor<2x128x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
    %2 = arith.addf %in, %in_0 : f32
    %3 = arith.addf %2, %in_1 : f32
    %4 = arith.addf %3, %in_2 : f32
    linalg.yield %4 : f32
  } -> tensor<2x128x128xf32>
  return %1 : tensor<2x128x128xf32>
}

// -----

//CHECK-LABEL: func.func @tensor_pad
func.func @tensor_pad(%arg0: tensor<2x34xi32>) -> tensor<2x64xi32> {
  %c3_i32 = arith.constant 3 : i32
  // CHECK-NOT: bufferization.to_tensor
  // CHECK: linalg.map
  //   CHECK-SAME: memref<2x64xi32>
  // CHECK-NOT: bufferization.to_memref
  %0 = tensor.pad %arg0 low[0, 0] high[0, 30] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %c3_i32 : i32
  } : tensor<2x34xi32> to tensor<2x64xi32>
  return %0 : tensor<2x64xi32>
}

// -----

//CHECK-LABEL: func.func @arith_const_tensor
func.func @arith_const_tensor() -> tensor<2x1xf32> {
  %cst = arith.constant dense<[[10.0], [11.0]]> : tensor<2x1xf32>
  // CHECK: memref.get_global
  return %cst : tensor<2x1xf32>
}

// -----

//CHECK-LABEL: func.func @tensor_splat
func.func @tensor_splat() -> tensor<2x1xf32> {
  %s = arith.constant 10.1 : f32
  %t = tensor.splat %s : tensor<2x1xf32>
  // CHECK: memref.get_global
  return %t : tensor<2x1xf32>
}
