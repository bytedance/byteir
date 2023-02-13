// RUN: tf-ext-opt --rewrite-func-attr-to-byteir %s | FileCheck %s

func.func @main(%arg0: tensor<f32> {tf.device = "/device:CPU:0"}, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<f32> attributes {tf.entry_function = {controls = "", inputs = "oracle_pred:0,style_0_score:0,style_1_score:0,style_2_score:0,style_3_score:0", outputs = "oracle_pred:0"}} {
  return %arg0 : tensor<f32>
}
// CHECK: func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<f32> attributes {byteir.entry_point = {inputs = ["oracle_pred:0", "style_0_score:0", "style_1_score:0", "style_2_score:0", "style_3_score:0"], outputs = ["oracle_pred:0"]}} {
