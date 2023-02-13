// RUN: byteir-opt %s -test-dtype-convert --canonicalize -canonicalize-ext -split-input-file | FileCheck %s

func.func @convert_foldable_dtypes(%arg0: tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf32> attributes {__byteir_unit_test__} {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<4x126x126x16xf32>, tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf32>
  %1 = "mhlo.add"(%0, %0) : (tensor<4x126x126x16xf32>, tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf32>
  return %1 : tensor<4x126x126x16xf32>
}
// CHECK-LABEL: func.func @convert_foldable_dtypes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf32> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.convert %[[ARG0]] : (tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf16>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.add %[[V0]], %[[V0]] : tensor<4x126x126x16xf16>
// CHECK-NEXT:   %[[V2:.*]] = mhlo.add %[[V1]], %[[V1]] : tensor<4x126x126x16xf16>
// CHECK-NEXT:   %[[V3:.*]] = mhlo.convert %[[V2]] : (tensor<4x126x126x16xf16>) -> tensor<4x126x126x16xf32>
// CHECK-NEXT:   return %[[V3]] : tensor<4x126x126x16xf32>

// -----

func.func @convert_dtypes(%arg0: tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf32> attributes {__byteir_unit_test__} {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<4x126x126x16xf32>, tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf32>
  %1 = "mhlo.subtract"(%0, %0) : (tensor<4x126x126x16xf32>, tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf32>
  return %1 : tensor<4x126x126x16xf32>
}

// CHECK-LABEL: func.func @convert_dtypes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf32> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.convert %[[ARG0]] : (tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf16>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.add %[[V0]], %[[V0]] : tensor<4x126x126x16xf16>
// CHECK-NEXT:   %[[V2:.*]] = mhlo.convert %[[V1]] : (tensor<4x126x126x16xf16>) -> tensor<4x126x126x16xf32>
// CHECK-NEXT:   %[[V3:.*]] = mhlo.subtract %[[V2]], %[[V2]] : tensor<4x126x126x16xf32>
// CHECK-NEXT:   return %[[V3]] : tensor<4x126x126x16xf32>

// -----

func.func @max_pool(%arg0: tensor<4x126x126x16xf32>) -> tensor<4x63x63x16xf32> attributes {__byteir_unit_test__} {
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.reduce_window"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x126x126x16xf32>, tensor<f32>) -> tensor<4x63x63x16xf32>
  return %1 : tensor<4x63x63x16xf32>
}
// CHECK-LABEL: func.func @max_pool
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x126x126x16xf32>) -> tensor<4x63x63x16xf32> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.constant dense<0xFC00> : tensor<f16>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.convert %[[ARG0]] : (tensor<4x126x126x16xf32>) -> tensor<4x126x126x16xf16>
// CHECK-NEXT:   %[[V2:.*]] = "mhlo.reduce_window"(%[[V1]], %[[V0]]) ({
// CHECK-NEXT:   ^bb0(%[[ARG1:.*]]: tensor<f16>, %[[ARG2:.*]]: tensor<f16>):
// CHECK-NEXT:     %[[V4:.*]] = mhlo.maximum %[[ARG1]], %[[ARG2]] : tensor<f16>
// CHECK-NEXT:     mhlo.return %[[V4]] : tensor<f16>
// CHECK-NEXT:   }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x126x126x16xf16>, tensor<f16>) -> tensor<4x63x63x16xf16>
// CHECK-NEXT:   %[[V3:.*]] = mhlo.convert %[[V2]] : (tensor<4x63x63x16xf16>) -> tensor<4x63x63x16xf32>
// CHECK-NEXT:   return %[[V1:.*]] : tensor<4x63x63x16xf32>

// -----

func.func @custom_call(%arg0: tensor<?xf32>) -> tensor<?xf32> attributes {__byteir_unit_test__} {
  %0 = "mhlo.subtract"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "mhlo.custom_call"(%0, %0) {call_target_name = "f16_custom_call", has_side_effect = false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @custom_call
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>) -> tensor<?xf32> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.subtract %[[ARG0]], %[[ARG0]] : tensor<?xf32>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.convert %[[V0]] : (tensor<?xf32>) -> tensor<?xf16>
// CHECK-NEXT:   %[[V2:.*]] = mhlo.custom_call @f16_custom_call(%[[V1]], %[[V1]]) : (tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
// CHECK-NEXT:   %[[V3:.*]] = mhlo.convert %[[V2]] : (tensor<?xf16>) -> tensor<?xf32>
// CHECK-NEXT:   return %[[V3]] : tensor<?xf32>

// -----

func.func @uselessConvert(%arg0: tensor<?xf16>) -> tensor<?xf16> attributes {__byteir_unit_test__} {
  %0 = "mhlo.subtract"(%arg0, %arg0) : (tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
  %1 = mhlo.convert %0 : (tensor<?xf16>) -> tensor<?xf32>
  %2 = "mhlo.custom_call"(%1, %1) {call_target_name = "f16_custom_call", has_side_effect = false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = mhlo.convert %2 : (tensor<?xf32>) -> tensor<?xf16>
  return %3 : tensor<?xf16>
}

// CHECK-LABEL: func.func @uselessConvert
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf16>) -> tensor<?xf16> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.subtract %[[ARG0]], %[[ARG0]] : tensor<?xf16>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.custom_call @f16_custom_call(%[[V0]], %[[V0]]) : (tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
// CHECK-NEXT:   return %[[V1]] : tensor<?xf16>

// -----

func.func @mixp_custom_call(%arg0: tensor<?xf16>) -> tensor<?xf16> attributes {__byteir_unit_test__} {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
  %1 = "mhlo.custom_call"(%0, %0) {call_target_name = "mixp_custom_call", has_side_effect = false} : (tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
  return %1 : tensor<?xf16>
}

// CHECK-LABEL: func.func @mixp_custom_call
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf16>) -> tensor<?xf16> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.add %[[ARG0]], %[[ARG0]] : tensor<?xf16>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.convert %[[V0]] : (tensor<?xf16>) -> tensor<?xf32>
// CHECK-NEXT:   %[[V2:.*]] = mhlo.custom_call @mixp_custom_call(%[[V1]], %[[V0]]) : (tensor<?xf32>, tensor<?xf16>) -> tensor<?xf16>
// CHECK-NEXT:   return %[[V2]] : tensor<?xf16>

// -----

func.func @mixp_custom_call2(%arg0: tensor<?xf32>) -> tensor<?xf16> attributes {__byteir_unit_test__} {
  %0 = mhlo.convert %arg0 : (tensor<?xf32>) -> tensor<?xf16>
  %1 = "mhlo.custom_call"(%0, %0) {call_target_name = "mixp_custom_call", has_side_effect = false} : (tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
  return %1 : tensor<?xf16>
}

// CHECK-LABEL: func.func @mixp_custom_call2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>) -> tensor<?xf16> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.convert %[[ARG0]] : (tensor<?xf32>) -> tensor<?xf16>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.custom_call @mixp_custom_call(%[[ARG0]], %[[V0]]) : (tensor<?xf32>, tensor<?xf16>) -> tensor<?xf16>
// CHECK-NEXT:   return %[[V1]] : tensor<?xf16>
