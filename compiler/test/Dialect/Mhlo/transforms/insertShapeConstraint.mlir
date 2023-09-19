// RUN: byteir-opt %s -insert-shape-constraint -canonicalize-ext -cse | FileCheck %s

// CHECK-LABEL: @DynamicPartition_constraint
func.func @DynamicPartition_constraint(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<?x4xf32> {
  %c0 = arith.constant 0 : index
  %0 = mhlo.constant dense<[[-0.916170597, -0.884184718, 1.60242105, -1.19678485], [0.33643803, -0.431175768, 1.71861267, 0.126368985], [-1.07191086, -1.00517535, -0.666032254, 0.776807785], [1.53380013, 0.83925873, -0.24277249, 1.53341103]]> : tensor<4x4xf32>
  %1 = mhlo.constant dense<[2.63629675, 2.68127704, 2.14741468, -1.6519475]> : tensor<4xf32>
  %2 = mhlo.constant dense<[[0.473984271, 0.173930168, 0.465745121, 1.14254773], [-0.384602815, -0.673360229, 1.13109767, 0.761463344], [-0.171464354, -0.908823907, 1.19337058, -1.78143835], [1.40376866, -0.529214859, -1.9030931, 1.25083804]]> : tensor<4x4xf32>
  %3 = mhlo.constant dense<[0.478572756, 0.458867788, -1.44476604, 0.189240679]> : tensor<4xf32>
  // CHECK-DAG: %[[C0:.*]] = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %4 = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  // CHECK-DAG: %[[V0:.*]]:2 = mhlo.custom_call @tf.DynamicPartition(%arg0, %arg1) {byteir_attrs = {num_partitions = 2 : i64}} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  %5:2 = "mhlo.custom_call"(%arg0, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "tf.DynamicPartition", has_side_effect = false} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  // CHECK-DAG: %[[V1:.*]] = tensor.dim %[[V0]]#0, %c0 : tensor<?x4xf32>
  // CHECK-DAG: %[[V2:.*]] = tensor.dim  %[[V0]]#1, %c0 : tensor<?x4xf32>
  // CHECK-DAG: %[[V3:.*]] = shape.add %[[V1]], %[[V2]] : index, index -> index
  // CHECK-DAG: "shape_ext.meet"(%[[V3]], %c4) : (index, index) -> ()
  %6 = tensor.dim %5#1, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%5#1, %6) : (tensor<?x4xf32>, index) -> ()
  %7 = tensor.dim %5#0, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%5#0, %7) : (tensor<?x4xf32>, index) -> ()
  %8 = "mhlo.dot"(%5#0, %0) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %9 = tensor.dim %8, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%8, %9) : (tensor<?x4xf32>, index) -> ()
  %10 = shape.shape_of %8 : tensor<?x4xf32> -> tensor<2xindex>
  %11 = "mhlo.dynamic_broadcast_in_dim"(%1, %10) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %12 = tensor.dim %11, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%11, %12) : (tensor<?x4xf32>, index) -> ()
  %13 = mhlo.add %8, %11 : tensor<?x4xf32>
  %14 = tensor.dim %13, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%13, %14) : (tensor<?x4xf32>, index) -> ()
  %15 = "mhlo.dot"(%5#1, %2) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %16 = tensor.dim %15, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%15, %16) : (tensor<?x4xf32>, index) -> ()
  %17 = shape.shape_of %15 : tensor<?x4xf32> -> tensor<2xindex>
  %18 = "mhlo.dynamic_broadcast_in_dim"(%3, %17) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %19 = tensor.dim %18, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%18, %19) : (tensor<?x4xf32>, index) -> ()
  %20 = mhlo.add %15, %18 : tensor<?x4xf32>
  %21 = tensor.dim %20, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%20, %21) : (tensor<?x4xf32>, index) -> ()
  // CHECK-DAG: %[[V4:.*]]:2 = mhlo.custom_call @tf.DynamicPartition(%[[C0]], %arg1) {byteir_attrs = {num_partitions = 2 : i64}} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %22:2 = "mhlo.custom_call"(%4, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "tf.DynamicPartition", has_side_effect = false} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  // CHECK-DAG: %[[V5:.*]] = tensor.dim %[[V4]]#0, %c0 : tensor<?xi32>
  // CHECK-DAG: %[[V6:.*]] = tensor.dim %[[V4]]#1, %c0 : tensor<?xi32>
  // CHECK-DAG: %[[V7:.*]] = shape.add %[[V5]], %[[V6]] : index, index -> index
  // CHECK-DAG: "shape_ext.meet"(%[[V7]], %c4) : (index, index) -> ()
  %23 = tensor.dim %22#1, %c0 : tensor<?xi32>
  "shape_ext.tie"(%22#1, %23) : (tensor<?xi32>, index) -> ()
  %24 = tensor.dim %22#0, %c0 : tensor<?xi32>
  "shape_ext.tie"(%22#0, %24) : (tensor<?xi32>, index) -> ()
  %25 = "mhlo.custom_call"(%22#0, %22#1, %13, %20) {call_target_name = "tf.DynamicStitch", has_side_effect = false} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %26 = tensor.dim %25, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%25, %26) : (tensor<?x4xf32>, index) -> ()
  return %25 : tensor<?x4xf32>
}

func.func @einsum_shape_constraint_1(%arg0: tensor<2x3x?x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "dcbq,btd->bqtc"} : (tensor<2x3x?x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32>
  return %0 : tensor<2x2x2x3xf32>
}
// CHECK-LABEL: einsum_shape_constraint_1
// CHECK-DAG: %[[C0:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[V0:.+]] = tensor.dim %arg0, %[[C0]]
// CHECK-DAG: "shape_ext.meet"(%[[C0]], %[[V0]]) : (index, index) -> ()

func.func @dot_general(%arg0: tensor<?x?x4xf32>, %arg1: tensor<?x4x128xf32>) -> tensor<3xindex> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x4xf32>, tensor<?x4x128xf32>) -> tensor<?x?x128xf32>
  %c0 = arith.constant 0 : index
  %1 = tensor.dim %0, %c0 : tensor<?x?x128xf32>
  %c1 = arith.constant 1 : index
  %2 = tensor.dim %0, %c1 : tensor<?x?x128xf32>
  "shape_ext.tie"(%0, %1, %2) : (tensor<?x?x128xf32>, index, index) -> ()
  %3 = shape.shape_of %0 : tensor<?x?x128xf32> -> tensor<3xindex>
  return %3 : tensor<3xindex>
}
// CHECK-LABEL: func.func @dot_general
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[V0:.+]] = tensor.dim %arg0, %[[C0]] : tensor<?x?x4xf32>
// CHECK-DAG:     %[[V1:.+]] = tensor.dim %arg1, %[[C0]] : tensor<?x4x128xf32>
// CHECK-DAG:     "shape_ext.meet"(%[[V0]], %[[V1]]) : (index, index) -> ()
// CHECK-DAG:     "shape_ext.meet"(%[[C2]], %[[C2]]) : (index, index) -> ()

func.func @dynamic_reshape(%arg0: tensor<?x128xf16>, %arg1: tensor<?xf16>) -> tensor<3xindex> {
  %c0 = "arith.constant"() {value = 0 : index} : () -> index
  %0 = "tensor.dim"(%arg0, %c0) : (tensor<?x128xf16>, index) -> index
  %c1 = "arith.constant"() {value = 1 : index} : () -> index
  %c128 = "arith.constant"() {value = 128 : index} : () -> index
  %1 = "tensor.from_elements"(%0, %c1, %c128) : (index, index, index) -> tensor<3xindex>
  %2 = "mhlo.dynamic_reshape"(%arg1, %1) : (tensor<?xf16>, tensor<3xindex>) -> tensor<?x1x128xf16>
  %3 = shape.shape_of %2 : tensor<?x1x128xf16> -> tensor<3xindex>
  return %3 : tensor<3xindex>
}
// CHECK-LABEL: func @dynamic_reshape
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[DIM:.+]] = tensor.dim %arg0, %[[C0]] : tensor<?x128xf16>
// CHECK-DAG:     %[[FROM_ELEMENTS:.+]] = tensor.from_elements %[[DIM]], %[[C1]], %[[C2]] : tensor<3xindex>
// CHECK-DAG:     %[[V0:.+]] = mhlo.dynamic_reshape %arg1, %[[FROM_ELEMENTS]] : (tensor<?xf16>, tensor<3xindex>) -> tensor<?x1x128xf16>
// CHECK-DAG:     %[[DIM0:.+]] = tensor.dim %arg1, %[[C0]] : tensor<?xf16>
// CHECK-DAG:     %[[DIM1:.+]] = tensor.dim %[[V0]], %[[C0]] : tensor<?x1x128xf16>
// CHECK-DAG:     %[[V2:.+]] = shape.mul %[[DIM1]], %[[C1]] : index, index -> index
// CHECK-DAG:     %[[V3:.+]] = shape.mul %[[V2]], %[[C2]] : index, index -> index
// CHECK-DAG:     "shape_ext.meet"(%[[DIM0]], %[[V3]])

func.func @reshape_scalar(%arg0 : tensor<1x1xf32>) -> tensor<f32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x1xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}
// CHECK-LABEL: @reshape_scalar(%arg0: tensor<1x1xf32>) -> tensor<f32>
// CHECK: "shape_ext.meet"


func.func @concat(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x?x?xf32>) -> tensor<?x4x?xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg1, %c1 : tensor<2x?x?xf32>
  %dim_0 = tensor.dim %arg1, %c2 : tensor<2x?x?xf32>
  "shape_ext.tie"(%arg1, %dim, %dim_0) : (tensor<2x?x?xf32>, index, index) -> ()
  %dim_1 = tensor.dim %arg0, %c0 : tensor<?x4x?xf32>
  %dim_2 = tensor.dim %arg0, %c2 : tensor<?x4x?xf32>
  "shape_ext.tie"(%arg0, %dim_1, %dim_2) : (tensor<?x4x?xf32>, index, index) -> ()
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x?x?xf32>) -> tensor<?x4x?xf32>
  %dim_3 = tensor.dim %0, %c0 : tensor<?x4x?xf32>
  %dim_4 = tensor.dim %0, %c2 : tensor<?x4x?xf32>
  "shape_ext.tie"(%0, %dim_3, %dim_4) : (tensor<?x4x?xf32>, index, index) -> ()
  return %0 : tensor<?x4x?xf32>
}
// CHECK-LABEL: func.func @concat
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[DIM:.+]] = tensor.dim %arg1, %c1
// CHECK-DAG:     %[[DIM1:.+]] = tensor.dim %arg0, %c2
// CHECK-DAG:     %[[DIM2:.+]] = tensor.dim %arg1, %c2
// CHECK-DAG:     "shape_ext.meet"(%[[C4]], %[[DIM]])
// CHECK-DAG:     "shape_ext.meet"(%[[DIM1]], %[[DIM2]])
