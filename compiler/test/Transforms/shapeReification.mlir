// RUN: byteir-opt %s --split-input-file -byteir-shape-reification -canonicalize -cse | FileCheck %s

func.func @several_ops(%arg0: tensor<?x2xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4xf32>) -> (!shape.shape, !shape.shape, !shape.shape, !shape.shape) {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x2xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
  %3 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %4 = shape.shape_of %3 : tensor<2xindex> -> tensor<1xindex>
  %5 = shape.value_as_shape %4 : tensor<1xindex> -> !shape.shape
  %6 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>} :
(tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %7 = shape.shape_of %6 : tensor<?x4xf32> -> tensor<2xindex>
  %8 = shape.value_as_shape %7 : tensor<2xindex> -> !shape.shape
  %9 = mhlo.add %0, %6 : tensor<?x4xf32>
  %10 = shape.shape_of %9 : tensor<?x4xf32> -> tensor<2xindex>
  %11 = shape.value_as_shape %10 : tensor<2xindex> -> !shape.shape
  return %2, %5, %8, %11 : !shape.shape, !shape.shape, !shape.shape, !shape.shape
}
// CHECK-LABEL: func.func @several_ops
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C2:.+]] = shape.const_shape [2] : tensor<1xindex>
// CHECK-DAG:     %[[V0:.+]] = tensor.dim %arg0, %[[C0]] : tensor<?x2xf32>
// CHECK-DAG:     %[[V1:.+]] = tensor.from_elements %[[V0]], %[[C1]] : tensor<2xindex>
// CHECK-DAG:     %[[V2:.+]] = shape.value_as_shape %[[V1]] : tensor<2xindex> -> !shape.shape
// CHECK-DAG:     %[[V3:.+]] = shape.value_as_shape %[[C2]] : tensor<1xindex> -> !shape.shape
// CHECK-DAG:     return %[[V2]], %[[V3]], %[[V2]], %[[V2]] : !shape.shape, !shape.shape, !shape.shape, !shape.shape

// -----

// CHECK-LABEL: @infer_shape_using_dim_op
func.func @infer_shape_using_dim_op(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>, %arg2: tensor<4x4xf32>) -> !shape.shape {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  %1 = "mhlo.dot"(%0, %arg2) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[V0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x4xf32>
  // CHECK-DAG: %[[V1:.*]] = tensor.from_elements %[[V0]], %[[C1]] : tensor<2xindex>
  %2 = shape.shape_of %1 : tensor<?x4xf32> -> tensor<2xindex>
  // CHECK-DAG: %[[V2:.*]] = shape.value_as_shape %[[V1]] : tensor<2xindex> -> !shape.shape
  %3 = shape.value_as_shape %2 : tensor<2xindex> -> !shape.shape
  return %3 : !shape.shape
}

// -----

func.func @dynamic_stitch(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?x4xf32>, %arg3: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.custom_call"(%arg0, %arg1, %arg2, %arg3) {call_target_name = "tf.DynamicStitch", has_side_effect = false} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %c0 = arith.constant 0 : index
  %1 = tensor.dim %0, %c0 : tensor<?x4xf32>
  // CHECK-DAG: %[[V0:.*]] = tensor.dim %arg0, %c0 : tensor<?xi32>
  // CHECK-DAG: %[[V1:.*]] = tensor.dim %arg1, %c0 : tensor<?xi32>
  // CHECK-DAG: %[[V2:.*]] = shape.add %[[V0]], %[[V1]] : index, index -> index
  // CHECK-DAG: "shape_ext.tie"(%0, %[[V2]]) : (tensor<?x4xf32>, index) -> ()
  "shape_ext.tie"(%0, %1) : (tensor<?x4xf32>, index) -> ()
  return %0 : tensor<?x4xf32>
}

// -----

func.func @gelu(%arg0: tensor<?x3072xf32>) -> tensor<?x3072xf32> {
  %0 = mhlo.custom_call @byteir.gelu(%arg0) {backend_config = "", byteir_attrs = {approximate = "erf"}} : (tensor<?x3072xf32>) -> tensor<?x3072xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %0, %c0 : tensor<?x3072xf32>
  "shape_ext.tie"(%0, %dim) : (tensor<?x3072xf32>, index) -> ()
  // CHECK-DAG: %[[V0:.*]] = tensor.dim %arg0, %c0 : tensor<?x3072xf32>
  // CHECK-DAG: "shape_ext.tie"(%0, %[[V0]]) : (tensor<?x3072xf32>, index) -> ()
  return %0 : tensor<?x3072xf32>
}

// -----

// CHECK-LABEL: func.func @dot_general
func.func @dot_general(%arg0: tensor<?x?x4xf32>, %arg1: tensor<?x4x128xf32>) -> tensor<3xindex> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x4xf32>, tensor<?x4x128xf32>) -> tensor<?x?x128xf32>
  // CHECK-DAG: %[[V0:.+]] = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x4xf32>, tensor<?x4x128xf32>) -> tensor<?x?x128xf32>
  // CHECK-DAG: %[[V1:.+]] = tensor.dim %arg0, %[[C0]] : tensor<?x?x4xf32>
  // CHECK-DAG: %[[V2:.+]] = tensor.dim %arg0, %[[C1]] : tensor<?x?x4xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?x128xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?x128xf32>
  "shape_ext.tie"(%0, %1, %2) : (tensor<?x?x128xf32>, index, index) -> ()
  // CHECK-DAG: "shape_ext.tie"(%[[V0]], %[[V1]], %[[V2]])
  %3 = shape.shape_of %0 : tensor<?x?x128xf32> -> tensor<3xindex>
  return %3 : tensor<3xindex>
}

// -----

// TODO: Check this after nested function call is supported
func.func private @inner_func(%arg0 : tensor<?x4xf32>, %arg1 : tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

func.func @outer_func(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> (!shape.shape, !shape.shape) {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
  %3 = call @inner_func(%0, %arg0) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %4 = shape.shape_of %3 : tensor<?x4xf32> -> tensor<2xindex>
  %5 = shape.value_as_shape %4 : tensor<2xindex> -> !shape.shape
  return %2, %5 : !shape.shape, !shape.shape
}
// CHECK-LABEL: func.func @outer_func
// CHECK: %[[V0:.*]] = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK: %[[V1:.*]] = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
// CHECK: return %[[V1]], %[[V1]] : !shape.shape, !shape.shape

// -----

func.func private @Unknown1(%arg0: tensor<?x10xf32>, %arg1: tensor<?x20xf32>) -> tensor<?x20xf32> attributes {__byteir_matmul_epilogue_fusion__} {
  %0 = mhlo.constant dense_resource<__elided__> : tensor<10x20xf32>
  %1 = "mhlo.dot"(%arg0, %0) : (tensor<?x10xf32>, tensor<10x20xf32>) -> tensor<?x20xf32>
  %2 = mhlo.add %1, %arg1 : tensor<?x20xf32>
  return %2 : tensor<?x20xf32>
}

func.func private @Unknown0(%arg0: tensor<?x10xf32>, %arg1: tensor<20xf32>, %arg2: tensor<?x20xf32>, %arg3: tensor<?x20xf32>) -> (tensor<?x20xf32>, tensor<?x20xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %c20 = arith.constant 20 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x10xf32>
  %from_elements = tensor.from_elements %dim, %c20 : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %from_elements) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20xf32>, tensor<2xindex>) -> tensor<?x20xf32>
  %2 = mhlo.add %arg2, %1 : tensor<?x20xf32>
  %3 = call @Unknown1(%arg0, %2) : (tensor<?x10xf32>, tensor<?x20xf32>) -> tensor<?x20xf32>
  %4 = mhlo.maximum %2, %3 : tensor<?x20xf32>
  return %4, %3 : tensor<?x20xf32>, tensor<?x20xf32>
}

func.func @forward(%arg0: tensor<?x10xf32>, %arg1: tensor<?x20xf32>, %arg2: tensor<20x?xf32>) -> tensor<2xindex> attributes {__placeholder__byre.entry_point} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = mhlo.constant dense_resource<__elided__> : tensor<10x20xf32>
  %1 = mhlo.constant dense_resource<__elided__> : tensor<20xf32>
  %2 = "mhlo.dot"(%arg0, %0) : (tensor<?x10xf32>, tensor<10x20xf32>) -> tensor<?x20xf32>
  %3:2 = call @Unknown0(%arg0, %1, %2, %arg1) : (tensor<?x10xf32>, tensor<20xf32>, tensor<?x20xf32>, tensor<?x20xf32>) -> (tensor<?x20xf32>, tensor<?x20xf32>)
  %4 = "mhlo.dot"(%3#0, %arg2) : (tensor<?x20xf32>, tensor<20x?xf32>) -> tensor<?x?xf32>
  %5 = shape.shape_of %4 : tensor<?x?xf32> -> tensor<2xindex>
  return %5 : tensor<2xindex>
}

// CHECK-LABEL: func.func @forward
// CHECK: %[[DIM:.*]] = tensor.dim %arg0, %c0 : tensor<?x10xf32>
// CHECK: %[[DIM0:.*]] = tensor.dim %arg2, %c1 : tensor<20x?xf32>
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[DIM:.*]], %[[DIM0:.*]] : tensor<2xindex>
// CHECK: return %[[SHAPE:.*]] : tensor<2xindex>