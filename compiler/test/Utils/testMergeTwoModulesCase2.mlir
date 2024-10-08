// RUN: byteir-opt %s --allow-unregistered-dialect

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) attributes {byteir.entry_point = {inputs = ["module0_input0", "module0_input1"], outputs = ["x", "y"]}} {
    return %arg0, %arg1 : tensor<f32>, tensor<f32>
}
