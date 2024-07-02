// RUN: byteir-opt %s -test-print-symbolic-shape -split-input-file | FileCheck %s


func.func @several_ops(%arg0: tensor<?x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %3 = mhlo.add %0, %2 : tensor<?x4xf32>
  return %3 : tensor<?x4xf32>
}
// CHECK-LABEL: ============= auxiliary shape function for @several_ops =============
// CHECK-NEXT: func.func private @_shape_infer_several_ops
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[V0:.+]] = shape.const_shape [2] : tensor<1xindex>
// CHECK-DAG:   %[[V2:.+]] = tensor.dim %arg0, %[[C0]] : tensor<?x4xf32>
// CHECK-DAG:   %[[V3:.+]] = tensor.from_elements %[[V2]], %[[C4]] : tensor<2xindex>  
// CHECK-DAG:   %[[V4:.+]] = shape.value_as_shape %[[V3]] : tensor<2xindex> -> !shape.shape
// CHECK-DAG:   %[[V5:.+]] = shape.value_as_shape %[[V0]] : tensor<1xindex> -> !shape.shape
// CHECK-DAG:   return %[[V4]], %[[V5]], %[[V4]], %[[V4]],

// CHECK-LABEL: ============= symbolic shape table for @several_ops =============
// CHECK-NEXT: original value: %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-NEXT: symbolic shape: %from_elements = tensor.from_elements %dim, %c4 : tensor<2xindex>

// CHECK:      original value: %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-NEXT: symbolic shape: %0 = shape.const_shape [2] : tensor<1xindex>

// CHECK:      original value: %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
// CHECK-NEXT: symbolic shape: %from_elements = tensor.from_elements %dim, %c4 : tensor<2xindex>

// CHECK:      original value: %3 = mhlo.add %0, %2 : tensor<?x4xf32>
// CHECK-NEXT: symbolic shape: %from_elements = tensor.from_elements %dim, %c4 : tensor<2xindex>

// CHECK-LABEL: ============= symbolic expr sources table for @several_ops =============
// CHECK-NEXT: original value: %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-NEXT: symbolic shape sources: 
// CHECK-NEXT: <block argument> of type 'tensor<?x4xf32>' at index: 0
// CHECK:      original value: %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-NEXT: symbolic shape sources: 
// CHECK:      original value: %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
// CHECK-NEXT: symbolic shape sources: 
// CHECK-NEXT: <block argument> of type 'tensor<?x4xf32>' at index: 0
// CHECK:      original value: %3 = mhlo.add %0, %2 : tensor<?x4xf32>
// CHECK-NEXT: symbolic shape sources: 
// CHECK-NEXT: <block argument> of type 'tensor<?x4xf32>' at index: 0
