// RUN: onnx-frontend-opt -shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test the default behavior of shape inference when onnx and mhlo dialects
/// are mixed together.
//===----------------------------------------------------------------------===//

module {
  func.func @test_onnx_with_custom_call(%arg0: tensor<10x10xf32>) -> tensor<10xf32> {
    %0 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<10x10xf32>) -> tensor<*xf32>
    %1 = "onnx.Log"(%0) : (tensor<*xf32>) -> tensor<10x10xf32>
    %2 = "mhlo.custom_call"(%1) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64, keep_dims = false, select_last_index = false}, call_target_name = "byteir.arg_max", called_computations = [], has_side_effect = false} : (tensor<10x10xf32>) -> tensor<10xf32>
    return %2 : tensor<10xf32>
  }
// CHECK-LABEL:  @test_onnx_with_custom_call(%arg0: tensor<10x10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:     %0 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:     %1 = "onnx.Log"(%0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:     %2 = mhlo.custom_call @byteir.arg_max(%1) {backend_config = "", byteir_attrs = {axis = 3 : i64, keep_dims = false, select_last_index = false}} : (tensor<10x10xf32>) -> tensor<10xf32>
// CHECK-NEXT:     return %2 : tensor<10xf32>
// CHECK-NEXT:   }
}
