// RUN: byteir-opt %s --insert-detensorize-transform="func-anchor=test_func_anchor match-prefix=test_prefix" --transform-dialect-interpreter --canonicalize --cse --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @elementwise
//   CHECK-SAME: (%[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<f32>)
func.func @elementwise(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> attributes {test_func_anchor, other} {
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<f32>)
                             outs(%arg1: tensor<f32>) -> tensor<f32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][] : tensor<f32>
  // CHECK: %[[UNARY:.+]] = math.exp %[[EXTRACT]] : f32
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<f32>, tensor<f32>)
                             outs(%arg1: tensor<f32>) -> tensor<f32>
  // CHECK: %[[BINARY:.+]] = arith.addf %[[UNARY]], %[[EXTRACT]] : f32
  // CHECK: %[[RET:.+]] = tensor.from_elements %[[BINARY]] : tensor<f32>
  return %1 : tensor<f32>
  // CHECK: return %[[RET]] : tensor<f32>
}

// CHECK: transform.sequence failures(propagate) {
// CHECK: ^bb0(%arg0: !pdl.operation): 
// CHECK: %0 = transform.structured.match attributes {test_prefix_0}
// CHECK: transform.structured.detensorize %0