// RUN: onnx-frontend-opt -of-modify-entry-point %s -split-input-file | FileCheck %s

module {
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
  func.func @main_graph(%arg0: tensor<1x5x5x3xf32> {onnx.name = "input_0"}, %arg1: tensor<1x5x5x3xf32> {onnx.name = "input_1"}) -> (tensor<1x5x5x3xf32> {onnx.name = "output_0"}) {
    %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "Add_0"} : (tensor<1x5x5x3xf32>, tensor<1x5x5x3xf32>) -> tensor<1x5x5x3xf32>
    return %0 : tensor<1x5x5x3xf32>
  }
// CHECK-LABEL:  module {
// CHECK-LABEL:    @main
// CHECK-SAME:     (%arg0: tensor<1x5x5x3xf32> {onnx.name = "input_0"}, %arg1: tensor<1x5x5x3xf32> {onnx.name = "input_1"}) -> (tensor<1x5x5x3xf32> {onnx.name = "output_0"}) attributes {byteir.entry_point = {inputs = ["input_0", "input_1"], outputs = ["output_0"]}} {
// CHECK-NEXT:       %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "Add_0"} : (tensor<1x5x5x3xf32>, tensor<1x5x5x3xf32>) -> tensor<1x5x5x3xf32>
// CHECK-NEXT:       return %0 : tensor<1x5x5x3xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
}

