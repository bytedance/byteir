// RUN: onnx-frontend %S/dynamic_shape_relu.onnx --input-name-and-shapes=X,1,128,80 -- | FileCheck %s

// CHECK-LABEL:  func.func @main
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x128x80xf32> {onnx.name = "X"}) -> (tensor<1x128x80xf32> {onnx.name = "Y"}) attributes {byteir.entry_point = {inputs = ["X"], outputs = ["Y"]}} {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1x128x80xf32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.maximum [[PARAM_0_]], [[VAR_0_]] : tensor<1x128x80xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x128x80xf32>