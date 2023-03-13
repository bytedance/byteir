// RUN: onnx-frontend-opt -of-canonicalize %s -split-input-file | FileCheck %s

func.func @test_softmax(%230: tensor<1x2xf32>) -> tensor<?x2xf32> {
  %231 = "onnx.Shape"(%230) {start = 0 : si64} : (tensor<1x2xf32>) -> tensor<2xi64>
  %232 = "onnx.Flatten"(%230) {axis = 1 : si64} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %233 = "onnx.Softmax"(%232) {axis = -1 : si64} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %234 = "onnx.Reshape"(%233, %231) {allowzero = 0 : si64} : (tensor<1x2xf32>, tensor<2xi64>) -> tensor<?x2xf32>
  return %234 : tensor<?x2xf32>
// CHECK-LABEL:  @test_softmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2xf32>) -> tensor<1x2xf32> {
// CHECK-NEXT:   [[VAR_0_:%.+]] = "onnx.Flatten"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:   [[VAR_1_:%.+]] = "onnx.Softmax"(%0) {axis = -1 : si64} : (tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:   return [[VAR_1_]] : tensor<1x2xf32>
}