// RUN: byteir-opt %s -bounded-shape-infer | FileCheck %s

func.func @SameOperandsAndResultShape(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}) -> tensor<?x4xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}
//CHECK-LABEL: func.func @SameOperandsAndResultShape(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {byteir.bounded_shape = [8, 4]}) -> tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {
//CHECK-NEXT:  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  return %0 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>

func.func @InferShapedTypeOpInterface(%arg0 : tensor<8x4xi32>, %arg1 : tensor<8x4xi32>) -> tensor<?x4xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<?x4xi1>
  return %0 : tensor<?x4xi1>
}
//CHECK-LABEL:func.func @InferShapedTypeOpInterface(%arg0: tensor<8x4xi32>, %arg1: tensor<8x4xi32>) -> tensor<?x4xi1, {byteir.bounded_shape = [8, 4]}> {
//CHECK-NEXT:  %0 = mhlo.compare LT, %arg0, %arg1 : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<?x4xi1, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  return %0 : tensor<?x4xi1, {byteir.bounded_shape = [8, 4]}>

func.func @several_ops(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %3 = mhlo.add %0, %2 : tensor<?x4xf32>
  return %3 : tensor<?x4xf32>
}
//CHECK-LABEL: func.func @several_ops(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {
//CHECK-NEXT:  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>, tensor<4x4xf32>) -> tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  %1 = shape.shape_of %0 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> -> tensor<2xindex>
//CHECK-NEXT:  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  %3 = mhlo.add %0, %2 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  return %3 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>

func.func @registered_shape_infer(%arg0 : tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}) -> tensor<?x2xi64> {
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "byteir.non_zero"} : (tensor<?x4xf32>) -> tensor<?x2xi64>
  return %0 : tensor<?x2xi64>
}
//CHECK-LABEL: func.func @registered_shape_infer(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {byteir.bounded_shape = [8, 4]}) -> tensor<?x2xi64, {byteir.bounded_shape = [32, 2]}> {
//CHECK-NEXT: %0 = mhlo.custom_call @byteir.non_zero(%arg0) : (tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>) -> tensor<?x2xi64, {byteir.bounded_shape = [32, 2]}>
//CHECK-NEXT: return %0 : tensor<?x2xi64, {byteir.bounded_shape = [32, 2]}>

func.func @main_sub_0(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [4, 4]}) -> tensor<?xf32> {
  %0 = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x4xf32>, tensor<f32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
//CHECK-LABEL: func.func @main_sub_0(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}> {byteir.bounded_shape = [4, 4]}) -> tensor<?xf32, {byteir.bounded_shape = [4]}> {
//CHECK-NEXT:  %0 = mhlo.constant dense<-0.000000e+00> : tensor<f32>
//CHECK-NEXT:  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<f32>) -> tensor<?xf32, {byteir.bounded_shape = [4]}>
//CHECK-NEXT:  return %1 : tensor<?xf32, {byteir.bounded_shape = [4]}>

func.func @concat(%arg0 : tensor<?x3xf32> {byteir.bounded_shape = [3, 3]}, %arg1 : tensor<?x3xf32> {byteir.bounded_shape = [3, 3]}) -> tensor <?x6xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x6xf32>
  return %0 : tensor<?x6xf32>
}
//CHECK-LABEL: func.func @concat(%arg0: tensor<?x3xf32, {byteir.bounded_shape = [3, 3]}> {byteir.bounded_shape = [3, 3]}, %arg1: tensor<?x3xf32, {byteir.bounded_shape = [3, 3]}> {byteir.bounded_shape = [3, 3]}) -> tensor<?x6xf32, {byteir.bounded_shape = [3, 6]}> {
//CHECK-NEXT:  %0 = "mhlo.concatenate"(%arg0, %arg1) <{dimension = 1 : i64}> : (tensor<?x3xf32, {byteir.bounded_shape = [3, 3]}>, tensor<?x3xf32, {byteir.bounded_shape = [3, 3]}>) -> tensor<?x6xf32, {byteir.bounded_shape = [3, 6]}>


func.func @dynamic_broadcast_in_dim(%arg0 : tensor<?x32xf32> {byteir.bounded_shape = [32, 32]}) -> tensor <1x?x30x32xf32> {
  %c30 = arith.constant 30 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x32xf32>
  %1 = tensor.from_elements %c1, %0, %c30, %c32 : tensor<4xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<?x32xf32>, tensor<4xindex>) -> tensor<1x?x30x32xf32>
  return %2 : tensor<1x?x30x32xf32>
}
//CHECK-LABEL: func.func @dynamic_broadcast_in_dim(%arg0: tensor<?x32xf32, {byteir.bounded_shape = [32, 32]}> {byteir.bounded_shape = [32, 32]}) -> tensor<1x?x30x32xf32, {byteir.bounded_shape = [1, 32, 30, 32]}> {
//CHECK: %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %from_elements) <{broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>}> : (tensor<?x32xf32, {byteir.bounded_shape = [32, 32]}>, tensor<4xindex>) -> tensor<1x?x30x32xf32, {byteir.bounded_shape = [1, 32, 30, 32]}>

func.func @torch_index_select(%arg0: tensor<10x128xf16>, %arg1: tensor<?xi32> {byteir.bounded_shape = [10]}) -> tensor<?x128xf16> {
  %6 = "mhlo.torch_index_select"(%arg0, %arg1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<10x128xf16>, tensor<?xi32>) -> tensor<?x128xf16>
  return %6 : tensor<?x128xf16>
}
// CHECK-LABEL: func.func @torch_index_select(%arg0: tensor<10x128xf16>, %arg1: tensor<?xi32, {byteir.bounded_shape = [10]}> {byteir.bounded_shape = [10]}) -> tensor<?x128xf16, {byteir.bounded_shape = [10, 128]}>
// CHECK-NEXT:  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<10x128xf16>, tensor<?xi32, {byteir.bounded_shape = [10]}>) -> tensor<?x128xf16, {byteir.bounded_shape = [10, 128]}>
// CHECK-NEXT:  return %0 : tensor<?x128xf16, {byteir.bounded_shape = [10, 128]}>

func.func @dot_general(%arg0 : tensor<?x1x8xf16> {byteir.bounded_shape = [1, 1, 8]}, %arg1 : tensor<?x8x128xf16> {byteir.bounded_shape = [1, 8, 128]}) -> tensor<?x1x128xf16> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x1x8xf16>, tensor<?x8x128xf16>) -> tensor<?x1x128xf16>
  return %0 : tensor<?x1x128xf16>
}
// CHECK-LABEL: func.func @dot_general(%arg0: tensor<?x1x8xf16, {byteir.bounded_shape = [1, 1, 8]}> {byteir.bounded_shape = [1, 1, 8]}, %arg1: tensor<?x8x128xf16, {byteir.bounded_shape = [1, 8, 128]}> {byteir.bounded_shape = [1, 8, 128]}) -> tensor<?x1x128xf16, {byteir.bounded_shape = [1, 1, 128]}>

func.func @real_dynamic_slice(%arg0 : tensor<?x26x128xf16> {byteir.bounded_shape = [1, 26, 128]}) -> tensor<?x12x128xf16> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c12 = arith.constant 12 : index
    %cst_0 = arith.constant dense<0> : tensor<3xindex>
    %cst_1 = arith.constant dense<1> : tensor<3xindex>
    %0 = tensor.dim %arg0, %c0 : tensor<?x26x128xf16>
    %1 = tensor.from_elements %0, %c12, %c128 : tensor<3xindex>
    %2 = "mhlo.real_dynamic_slice"(%arg0, %cst_0, %1, %cst_1) : (tensor<?x26x128xf16>, tensor<3xindex>, tensor<3xindex>, tensor<3xindex>) -> tensor<?x12x128xf16>
    return %2 : tensor<?x12x128xf16>
}
// CHECK-LABEL: func.func @real_dynamic_slice(%arg0: tensor<?x26x128xf16, {byteir.bounded_shape = [1, 26, 128]}> {byteir.bounded_shape = [1, 26, 128]}) -> tensor<?x12x128xf16, {byteir.bounded_shape = [1, 12, 128]}>


func.func @compatible_operands_and_result_type(%arg0 : tensor<?x32x32x12xf16> {byteir.bounded_shape = [6, 32, 32, 12]}) -> tensor<?x12x32x32xf16> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x32x32x12xf16>) -> tensor<?x12x32x32xf16>
  %1 = "mhlo.reverse"(%0) {dimensions = dense<3> : tensor<1xi64>} : (tensor<?x12x32x32xf16>) -> tensor<?x12x32x32xf16>
  return %1 : tensor<?x12x32x32xf16>
}
// CHECK-LABEL: func.func @compatible_operands_and_result_type(%arg0: tensor<?x32x32x12xf16, {byteir.bounded_shape = [6, 32, 32, 12]}> {byteir.bounded_shape = [6, 32, 32, 12]}) -> tensor<?x12x32x32xf16, {byteir.bounded_shape = [6, 12, 32, 32]}>


// ERROR: error: 'mhlo.convolution' op requires attribute 'batch_group_count', 这不是给了吗，不懂
func.func @convolution(%arg0 : tensor<?x12x26x4xf16> {byteir.bounded_shape = [16, 12, 26, 4]}) -> tensor<?x12x26x12xf16> {
  %0 = mhlo.constant dense<1.00000> : tensor<5x7x4x12xf16>
  %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[4, 4], [6, 6]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x12x26x4xf16>, tensor<5x7x4x12xf16>) -> tensor<?x12x26x12xf16>
  return %1 : tensor<?x12x26x12xf16>
}
// CHECK-LABEL:  func.func @convolution(%arg0: tensor<?x12x26x4xf16, {byteir.bounded_shape = [16, 12, 26, 4]}> {byteir.bounded_shape = [16, 12, 26, 4]}) -> tensor<?x12x26x12xf16, {byteir.bounded_shape = [16, 12, 26, 12]}>

func.func @gelu(%arg0: tensor<?x3072xf32> {byteir.bounded_shape = [1500, 3072]}) -> tensor<?x3072xf32> {
  %0 = mhlo.custom_call @byteir.gelu(%arg0) {backend_config = "", byteir_attrs = {approximate = "erf"}} : (tensor<?x3072xf32>) -> tensor<?x3072xf32>
  return %0 : tensor<?x3072xf32>
}
// CHECK-LABEL:  func.func @gelu(%arg0: tensor<?x3072xf32, {byteir.bounded_shape = [1500, 3072]}> {byteir.bounded_shape = [1500, 3072]}) -> tensor<?x3072xf32, {byteir.bounded_shape = [1500, 3072]}>
