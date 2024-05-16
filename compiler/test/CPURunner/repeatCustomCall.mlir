// RUN: byteir-opt %s -hlo-fusion-to-linalg --canonicalize-ext -byteir-bufferize-opt \
// RUN:      -convert-linalg-to-loops --canonicalize-ext -lower-affine -to-llvm -convert-math-to-libm -convert-func-to-llvm \
// RUN: | byteir-cpu-runner -e main -entry-point-result=void \
// RUN:     --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

module attributes {byteir.llvm_module} {
func.func @convert_repeat_static(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi64>) -> tensor<5x3xf32> attributes {__byteir_hlo_aggressive_fusion__} {
  %0 = mhlo.custom_call @byteir.repeat(%arg0, %arg1) {ada.op_unique_key = "ada.ref_value.0", backend_config = "", byteir_attrs = {Trepeats = i64}, device = "host"} : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}
func.func @convert_repeat_input_dynamic(%arg0: tensor<?x3xf32>, %arg1: tensor<?xi64>) -> tensor<5x3xf32> attributes {__byteir_hlo_aggressive_fusion__} {
  %0 = mhlo.custom_call @byteir.repeat(%arg0, %arg1) {ada.op_unique_key = "ada.ref_value.0", backend_config = "", byteir_attrs = {Trepeats = i64}, device = "host"} : (tensor<?x3xf32>, tensor<?xi64>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}
func.func @convert_repeat_output_dynamic(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi64>) -> tensor<?x3xf32> attributes {__byteir_hlo_aggressive_fusion__} {
  %0 = mhlo.custom_call @byteir.repeat(%arg0, %arg1) {ada.op_unique_key = "ada.ref_value.0", backend_config = "", byteir_attrs = {Trepeats = i64}, device = "host"} : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}
func.func @convert_repeat_dynamic(%arg0: tensor<?x3xf32>, %arg1: tensor<?xi64>) -> tensor<?x3xf32> attributes {__byteir_hlo_aggressive_fusion__} {
  %0 = mhlo.custom_call @byteir.repeat(%arg0, %arg1) {ada.op_unique_key = "ada.ref_value.0", backend_config = "", byteir_attrs = {Trepeats = i64}, device = "host"} : (tensor<?x3xf32>, tensor<?xi64>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

func.func @check_and_print(%arg0 : tensor<5x3xf32>, %arg1 : tensor<5x3xf32>) {
  %eps = mhlo.constant dense<1.000000e-06> : tensor<5x3xf32>
  %diff = mhlo.subtract %arg0, %arg1 : tensor<5x3xf32>
  %abs_diff = mhlo.abs %diff : tensor<5x3xf32>
  %cmp = mhlo.compare LT, %abs_diff, %eps : (tensor<5x3xf32>, tensor<5x3xf32>) -> tensor<5x3xi1>
  %cmp_i32 = mhlo.convert %cmp : (tensor<5x3xi1>) -> tensor<5x3xi32>
  %cmp_unranked = tensor.cast %cmp_i32 : tensor<5x3xi32> to tensor<*xi32>
  call @printMemrefI32(%cmp_unranked) : (tensor<*xi32>) -> ()
  return
}

func.func @main() {
  %src = arith.constant dense<[
    [-2.0, -1.0, 0.0],
    [1.0, 2.0, 3.0]
  ]> : tensor<2x3xf32>
  %multiplyer = arith.constant dense<[3, 2]> : tensor<2xi64>
  %expected = arith.constant dense<[
    [-2.0, -1.0, 0.0],
    [-2.0, -1.0, 0.0],
    [-2.0, -1.0, 0.0],
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0]
  ]> : tensor<5x3xf32>
  %casted_src = tensor.cast %src : tensor<2x3xf32> to tensor<?x3xf32>
  %casted_multiplyer = tensor.cast %multiplyer : tensor<2xi64> to tensor<?xi64>

  %static_actual = call @convert_repeat_static(%src, %multiplyer) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<5x3xf32>
  call @check_and_print(%static_actual, %expected) : (tensor<5x3xf32>, tensor<5x3xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]

  %input_dynamic_actual = call @convert_repeat_input_dynamic(%casted_src, %casted_multiplyer) : (tensor<?x3xf32>, tensor<?xi64>) -> tensor<5x3xf32>
  call @check_and_print(%input_dynamic_actual, %expected) : (tensor<5x3xf32>, tensor<5x3xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]

  %output_dynamic_actual = call @convert_repeat_output_dynamic(%src, %multiplyer) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<?x3xf32>
  %casted_output_dynamic_actual = tensor.cast %output_dynamic_actual : tensor<?x3xf32> to tensor<5x3xf32>
  call @check_and_print(%casted_output_dynamic_actual, %expected) : (tensor<5x3xf32>, tensor<5x3xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]

  %dynamic_actual = call @convert_repeat_dynamic(%casted_src, %casted_multiplyer) : (tensor<?x3xf32>, tensor<?xi64>) -> tensor<?x3xf32>
  %casted_dynamic_actual = tensor.cast %dynamic_actual : tensor<?x3xf32> to tensor<5x3xf32>
  call @check_and_print(%casted_dynamic_actual, %expected) : (tensor<5x3xf32>, tensor<5x3xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]
  //   CHECK-NEXT: [1, 1, 1]

  return
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
}
