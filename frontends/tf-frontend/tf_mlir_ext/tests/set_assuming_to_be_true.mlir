// RUN: tf-ext-opt -remove-cstr-reshapable -remove-shape-constraints -canonicalize %s | FileCheck %s

func.func @remove_cstr_reshapable(%arg0: index, %arg1: tensor<3xi32>, %arg2: tensor<32x?xf16>) -> tensor<?x?x64xf16> {
  %0 = mhlo.cstr_reshapable %arg0, %arg1 : (index, tensor<3xi32>) -> !shape.witness
  %1 = shape.assuming %0 -> (tensor<?x?x64xf16>) {
    %2 = mhlo.compute_reshape_shape %arg0, %arg1 : (index, tensor<3xi32>) -> tensor<3xi32>
    %3 = mhlo.dynamic_reshape %arg2, %2 : (tensor<32x?xf16>, tensor<3xi32>) -> tensor<?x?x64xf16>
    shape.assuming_yield %3 : tensor<?x?x64xf16>
  }
  return %1 : tensor<?x?x64xf16> 
}
// CHECK-LABEL: @remove_cstr_reshapable
// CHECK-NEXT: %0 = mhlo.compute_reshape_shape %arg0, %arg1 : (index, tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT: %1 = mhlo.dynamic_reshape %arg2, %0 : (tensor<32x?xf16>, tensor<3xi32>) -> tensor<?x?x64xf16>
// CHECK-NEXT: return %1 : tensor<?x?x64xf16>

func.func @remove_cstr_broadcastable(%arg0: tensor<3xindex>, %arg1: tensor<3xindex>, %arg2: tensor<?x?x64xf16>, %arg3: tensor<32x?x64xf16>) -> tensor<32x?x64xf16> {
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<3xindex>, tensor<3xindex>
  %1 = shape.assuming %0 -> (tensor<32x?x64xf16>) {
    %2 = shape.broadcast %arg0, %arg1 : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
    %3 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x64xf16>, tensor<3xindex>) -> tensor<32x?x64xf16>
    %4 = "mhlo.dynamic_broadcast_in_dim"(%arg3, %2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<32x?x64xf16>, tensor<3xindex>) -> tensor<32x?x64xf16>
    %5 = mhlo.multiply %3, %4 : tensor<32x?x64xf16>
    shape.assuming_yield %5 : tensor<32x?x64xf16>
  }
  return %1 : tensor<32x?x64xf16> 
}
// CHECK-LABEL: @remove_cstr_broadcastable
// CHECK-NEXT: %0 = shape.broadcast %arg0, %arg1 : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-NEXT: %1 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x64xf16>, tensor<3xindex>) -> tensor<32x?x64xf16>
// CHECK-NEXT: %2 = "mhlo.dynamic_broadcast_in_dim"(%arg3, %0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<32x?x64xf16>, tensor<3xindex>) -> tensor<32x?x64xf16>
// CHECK-NEXT: %3 = mhlo.multiply %1, %2 : tensor<32x?x64xf16>
// CHECK-NEXT: return %3 : tensor<32x?x64xf16>
