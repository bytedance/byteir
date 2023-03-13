// RUN: byteir-opt %s -hlo-fusion-to-linalg --canonicalize-ext -byteir-bufferize-opt \
// RUN:      -convert-linalg-to-loops --canonicalize-ext -lower-affine -to-llvm -convert-math-to-libm -convert-func-to-llvm \
// RUN: | byteir-cpu-runner -e main -entry-point-result=void \
// RUN:     --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

module attributes {byteir.llvm_module} {
func.func @fastgelu(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
  %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {approximate="tanh"}, call_target_name = "byteir.gelu", called_computations = [], has_side_effect = false} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

func.func @gelu(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
  %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {approximate="erf"}, call_target_name = "byteir.gelu", called_computations = [], has_side_effect = false} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

func.func @check_and_print(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x3xf32>) {
  %eps = mhlo.constant dense<1.000000e-06> : tensor<2x3xf32>
  %diff = mhlo.subtract %arg0, %arg1 : tensor<2x3xf32>
  %abs_diff = mhlo.abs %diff : tensor<2x3xf32>
  %cmp = mhlo.compare LT, %abs_diff, %eps : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
  %cmp_i32 = mhlo.convert %cmp : (tensor<2x3xi1>) -> tensor<2x3xi32>
  %cmp_unranked = tensor.cast %cmp_i32 : tensor<2x3xi32> to tensor<*xi32>
  call @printMemrefI32(%cmp_unranked) : (tensor<*xi32>) -> ()
  return
}

func.func @main() {
  %src = arith.constant dense<[
    [-2.0, -1.0, 0.0],
    [1.0, 2.0, 3.0]
  ]> : tensor<2x3xf32>

  %fastgelu_expected = arith.constant dense<[
    [-0.0454023060802099, -0.1588080096158213, 0.0],
    [0.8411919903841787, 1.9545976939197902, 2.9963626078936834]
  ]> : tensor<2x3xf32>
  %fastgelu_actual = call @fastgelu(%src) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  call @check_and_print(%fastgelu_actual, %fastgelu_expected) : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1] data =
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]

  %gelu_expected = arith.constant dense<[
    [-0.04550028, -0.15865526, 0.0],
    [0.8413447, 1.9544997, 2.9959502]
  ]> : tensor<2x3xf32>
  %gelu_actual = call @gelu(%src) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  call @check_and_print(%gelu_actual, %gelu_expected) : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1] data =
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  return
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
}