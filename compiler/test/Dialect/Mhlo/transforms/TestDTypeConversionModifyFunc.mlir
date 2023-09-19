// RUN: byteir-opt %s -test-dtype-convert-modify-func --canonicalize -split-input-file | FileCheck %s

func.func @convert_foldable_dtypes(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> attributes {__byteir_unit_test__} {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "mhlo.add"(%0, %0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func.func @convert_foldable_dtypes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf16>, %[[ARG1:.*]]: tensor<?xf16>) -> tensor<?xf16> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.add %[[ARG0]], %[[ARG1]] : tensor<?xf16>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.add %[[V0]], %[[V0]] : tensor<?xf16>
// CHECK-NEXT:   return %[[V1]] : tensor<?xf16>

// -----

func.func @custom_call(%arg0: tensor<?xf32>) -> tensor<?xf32> attributes {__byteir_unit_test__} {
  %0 = "mhlo.subtract"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "mhlo.custom_call"(%0, %0) {call_target_name = "f16_custom_call", has_side_effect = false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @custom_call
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>) -> tensor<?xf16> attributes {__byteir_unit_test__} {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.subtract %[[ARG0]], %[[ARG0]] : tensor<?xf32>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.convert %[[V0]] : (tensor<?xf32>) -> tensor<?xf16>
// CHECK-NEXT:   %[[V2:.*]] =  mhlo.custom_call @f16_custom_call(%[[V1]], %[[V1]]) : (tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
// CHECK-NEXT:   return %[[V2]] : tensor<?xf16>