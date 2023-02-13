// RUN: byteir-opt %s -insert-unique-id -split-input-file | FileCheck %s
// RUN: byteir-opt %s -insert-unique-id="anchor-attr=__test__" -split-input-file  | FileCheck %s --check-prefix ANCHOR

func.func @mhlo_add(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> attributes {__test__} {
  %res = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %res : tensor<4xf32>
}

// CHECK-LABEL: func.func @mhlo_add
// CHECK: {__byteir_unique_id__ = "mhlo.add_0"}
// CHECK: {__byteir_unique_id__ = "func.return_1"}

// ANCHOR-LABEL: func.func @mhlo_add
// ANCHOR: {__byteir_unique_id__ = "mhlo.add_0"}
// ANCHOR: {__byteir_unique_id__ = "func.return_1"}

// -----

func.func @linalg_max_pool(%arg0: tensor<4x128x128x16xf32>) -> tensor<4x64x64x16xf32> {
  %cst = arith.constant dense<0xFF800000> : tensor<f32>
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = tensor.empty() : tensor<4x64x64x16xf32>
  %extracted = tensor.extract %cst[] : tensor<f32>
  %2 = linalg.fill ins(%extracted : f32) outs(%1 : tensor<4x64x64x16xf32>) -> tensor<4x64x64x16xf32>
  %3 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %0 : tensor<4x128x128x16xf32>, tensor<2x2xf32>) outs(%2 : tensor<4x64x64x16xf32>) -> tensor<4x64x64x16xf32>
  return %3 : tensor<4x64x64x16xf32>
}

// CHECK-LABEL: func.func @linalg_max_pool
// CHECK: arith.constant {__byteir_unique_id__ = "arith.constant_0"}
// CHECK: tensor.empty() {__byteir_unique_id__ = "tensor.empty_1"}
// CHECK: tensor.empty() {__byteir_unique_id__ = "tensor.empty_2"}
// CHECK: {__byteir_unique_id__ = "tensor.extract_3"}
// CHECK: linalg.fill {__byteir_unique_id__ = "linalg.fill_5"}
// CHECK: linalg.pooling_nhwc_max {__byteir_unique_id__ = "linalg.pooling_nhwc_max_8"
// CHECK: return {__byteir_unique_id__ = "func.return_9"}

// ANCHOR-LABEL: func.func @linalg_max_pool
// ANCHOR: arith.constant dense<0xFF800000> : tensor<f32>
// ANCHOR: tensor.empty() : tensor<2x2xf32>
// ANCHOR: tensor.empty() : tensor<4x64x64x16xf32>
// ANCHOR: linalg.fill ins