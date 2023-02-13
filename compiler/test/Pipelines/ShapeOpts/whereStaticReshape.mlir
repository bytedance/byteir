// RUN: byteir-opt %s -shape-opt | FileCheck %s

func.func @where_static_reshape(%arg0: tensor<6144x12xf16>, %arg1: tensor<6144xi1>) -> tensor<96x384xf16> {
  %c0 = arith.constant 0 : index
  %0 = mhlo.constant dense<[-1, 32, 32, 12]> : tensor<4xi32>
  %1 = mhlo.constant dense<[-1, 384, 32]> : tensor<3xi32>
  %2 = "mhlo.custom_call"(%arg1) {api_version = 1 : i32, backend_config = "", byteir_attrs = {}, call_target_name = "tf.Where", called_computations = [], has_side_effect = false} : (tensor<6144xi1>) -> tensor<?x1xi64>
  %3 = tensor.dim %2, %c0 : tensor<?x1xi64>
  %4 = tensor.from_elements %3 : tensor<1xindex>
  %5 = shape.shape_of %2 : tensor<?x1xi64> -> tensor<2xindex>
  %6 = shape.num_elements %5 : tensor<2xindex> -> index
  %7 = mhlo.cstr_reshapable %6, %4 : (index, tensor<1xindex>) -> !shape.witness
  %8 = shape.assuming %7 -> (tensor<?xi64>) {
    %22 = mhlo.compute_reshape_shape %6, %4 : (index, tensor<1xindex>) -> tensor<1xindex>
    %23 = "mhlo.dynamic_reshape"(%2, %22) : (tensor<?x1xi64>, tensor<1xindex>) -> tensor<?xi64>
    shape.assuming_yield %23 : tensor<?xi64>
  }
  %9 = "mhlo.torch_index_select"(%arg0, %8) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<6144x12xf16>, tensor<?xi64>) -> tensor<?x12xf16>
  %10 = shape.shape_of %9 : tensor<?x12xf16> -> tensor<2xindex>
  %11 = shape.num_elements %10 : tensor<2xindex> -> index
  %12 = mhlo.cstr_reshapable %11, %0 : (index, tensor<4xi32>) -> !shape.witness
  %13 = shape.assuming %12 -> (tensor<?x32x32x12xf16>) {
    %22 = mhlo.compute_reshape_shape %11, %0 : (index, tensor<4xi32>) -> tensor<4xi32>
    %23 = "mhlo.dynamic_reshape"(%9, %22) : (tensor<?x12xf16>, tensor<4xi32>) -> tensor<?x32x32x12xf16>
    shape.assuming_yield %23 : tensor<?x32x32x12xf16>
  }
  %14 = "mhlo.transpose"(%13) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x32x32x12xf16>) -> tensor<?x12x32x32xf16>
  %15 = "mhlo.reverse"(%14) {dimensions = dense<3> : tensor<1xi64>} : (tensor<?x12x32x32xf16>) -> tensor<?x12x32x32xf16>
  %16 = shape.shape_of %15 : tensor<?x12x32x32xf16> -> tensor<4xindex>
  %17 = shape.num_elements %16 : tensor<4xindex> -> index
  %18 = mhlo.cstr_reshapable %17, %1 : (index, tensor<3xi32>) -> !shape.witness
  %19 = shape.assuming %18 -> (tensor<?x384x32xf16>) {
    %22 = mhlo.compute_reshape_shape %17, %1 : (index, tensor<3xi32>) -> tensor<3xi32>
    %23 = "mhlo.dynamic_reshape"(%15, %22) : (tensor<?x12x32x32xf16>, tensor<3xi32>) -> tensor<?x384x32xf16>
    shape.assuming_yield %23 : tensor<?x384x32xf16>
  }
  %20 = "mhlo.transpose"(%19) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x384x32xf16>) -> tensor<?x32x384xf16>
  %21 = "mhlo.reshape"(%20) : (tensor<?x32x384xf16>) -> tensor<96x384xf16>
  return %21 : tensor<96x384xf16>
}
// CHECK-LABEL: @where_static_reshape(%arg0: tensor<6144x12xf16>, %arg1: tensor<6144xi1>) -> tensor<96x384xf16>
// CHECK-NEXT:    %0 = mhlo.custom_call @tf.Where(%arg1) {backend_config = "", byteir_attrs = {}} : (tensor<6144xi1>) -> tensor<3072x1xi64>
// CHECK-NEXT:    %1 = mhlo.reshape %0 : (tensor<3072x1xi64>) -> tensor<3072xi64>
// CHECK-NEXT:    %2 = "mhlo.torch_index_select"(%arg0, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<6144x12xf16>, tensor<3072xi64>) -> tensor<3072x12xf16>
// CHECK-NEXT:    %3 = mhlo.reshape %2 : (tensor<3072x12xf16>) -> tensor<3x32x32x12xf16>
// CHECK-NEXT:    %4 = "mhlo.transpose"(%3) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<3x32x32x12xf16>) -> tensor<3x12x32x32xf16>
// CHECK-NEXT:    %5 = "mhlo.reverse"(%4) {dimensions = dense<3> : tensor<1xi64>} : (tensor<3x12x32x32xf16>) -> tensor<3x12x32x32xf16>
// CHECK-NEXT:    %6 = mhlo.reshape %5 : (tensor<3x12x32x32xf16>) -> tensor<3x384x32xf16>
// CHECK-NEXT:    %7 = "mhlo.transpose"(%6) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<3x384x32xf16>) -> tensor<3x32x384xf16>
// CHECK-NEXT:    %8 = mhlo.reshape %7 : (tensor<3x32x384xf16>) -> tensor<96x384xf16>
// CHECK-NEXT:    return %8 : tensor<96x384xf16>
