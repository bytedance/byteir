// RUN: byteir-opt %s -insert-input-shape-constraint="mode=all-dynamic-batch-same" --canonicalize | FileCheck %s

module attributes {torch.debug_module_name = "TestModule"} {
  func.func @forward(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %0 = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
    %1 = shape.shape_of %arg1 : tensor<?x4xf32> -> tensor<2xindex>
    %2 = shape.cstr_broadcastable %0, %1 : tensor<2xindex>, tensor<2xindex>
    %3 = shape.assuming %2 -> (tensor<?x4xf32>) {
      %4 = shape.broadcast %0, %1 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %5 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %4) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
      %6 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %4) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
      %7 = mhlo.add %5, %6 : tensor<?x4xf32>
      shape.assuming_yield %7 : tensor<?x4xf32>
    }
    return %3 : tensor<?x4xf32>
  }
}
// CHECK-LABEL:  func.func @forward
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[DIM:.*]] = tensor.dim %arg0, %c0
// CHECK-NEXT:     %[[DIM1:.*]] = tensor.dim %arg1, %c0
// CHECK-NEXT:     "shape_ext.meet"(%[[DIM1]], %[[DIM]])
