// RUN: byteir-opt %s --allow-unregistered-dialect

module {
func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> attributes {byteir.entry_point = {inputs = ["xx", "yy"], outputs = ["module1_output"]}} {
    %0 = "foo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
}
}
