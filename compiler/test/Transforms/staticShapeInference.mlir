// RUN: byteir-opt %s -static-shape-infer --canonicalize | FileCheck %s

// CHECK-LABEL: @InferShapedTypeOpInterface
func.func @InferShapedTypeOpInterface(%arg0 : tensor<8x4xi32>, %arg1 : tensor<8x4xi32>) -> tensor<?x4xi1> {
// CHECK-NEXT: %0 = mhlo.compare LT, %arg0, %arg1 : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<8x4xi1>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<?x4xi1>
// CHECK-NEXT: return %0 : tensor<8x4xi1>
  return %0 : tensor<?x4xi1>
}

// CHECK-LABEL: @InferSlice
func.func @InferSlice(%arg0: tensor<2x56x56x96xf16>) -> tensor<2x?x56x96xf16> {
// CHECK-NEXT: %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 56, 56, 96]> : tensor<4xi64>, start_indices = dense<[0, 3, 0, 0]> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<2x56x56x96xf16>) -> tensor<2x53x56x96xf16>
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 56, 56, 96]> : tensor<4xi64>, start_indices = dense<[0, 3, 0, 0]> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<2x56x56x96xf16>) -> tensor<2x?x56x96xf16>
// CHECK-NEXT: return %0 : tensor<2x53x56x96xf16>
  return %0 : tensor<2x?x56x96xf16>
}

// CHECK-LABEL: @InferReshapeSlice
func.func @InferReshapeSlice(%arg0: tensor<2x2x28x56x96xf16>) -> tensor<2x?x56x96xf16>  {
  %0 = mhlo.reshape %arg0 : (tensor<2x2x28x56x96xf16>) -> tensor<2x56x56x96xf16>
// CHECK: %[[V1:.*]] = "mhlo.slice"(%0) {limit_indices = dense<[2, 56, 56, 96]> : tensor<4xi64>, start_indices = dense<[0, 3, 0, 0]> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<2x56x56x96xf16>) -> tensor<2x53x56x96xf16>
  %1 = "mhlo.slice"(%0) {limit_indices = dense<[2, 56, 56, 96]> : tensor<4xi64>, start_indices = dense<[0, 3, 0, 0]> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<2x56x56x96xf16>) -> tensor<2x?x56x96xf16>
// CHECK-NEXT: return %[[V1]] : tensor<2x53x56x96xf16>
  return %1 : tensor<2x?x56x96xf16>
}

// CHECK-LABEL: @InferWithArgAttr
func.func @InferWithArgAttr(%arg0: tensor<?x8xf16> {byteir.static_shape = [2, 8]}, %arg1: tensor<?x8xf16> {byteir.static_shape = [2, 8]}) -> tensor<?x8xf16> {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x8xf16>
// CHECK-NEXT: %0 = mhlo.add %arg0, %arg1 : tensor<2x8xf16>
  %1 = mhlo.multiply %arg0, %0 : tensor<?x8xf16>
// CHECK-NEXT: %1 = mhlo.multiply %arg0, %0 : tensor<2x8xf16>
  return %1 : tensor<?x8xf16>
// CHECK-NEXT: return %1 : tensor<2x8xf16>
}

// CHECK-LABEL: @several_ops
func.func @several_ops(%arg0: tensor<?x4xf32> {byteir.static_shape = [8, 4]}, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-NEXT: %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
// CHECK-NEXT: %1 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<8x4xf32>
  %3 = mhlo.add %0, %2 : tensor<?x4xf32>
// CHECK-NEXT: %2 = mhlo.add %0, %1 : tensor<8x4xf32>
  return %3 : tensor<?x4xf32>
// CHECK-NEXT: return %2 : tensor<8x4xf32>
}
