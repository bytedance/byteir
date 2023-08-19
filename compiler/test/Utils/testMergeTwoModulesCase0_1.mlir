// RUN: byteir-opt %s --allow-unregistered-dialect

module {
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = "foo.add"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
}
}
