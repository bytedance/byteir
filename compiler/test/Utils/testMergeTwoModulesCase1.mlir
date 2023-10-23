// RUN: byteir-opt %s --test-merge-two-modules="second-module-path=%S/testMergeTwoModulesCase1_1.mlir" --allow-unregistered-dialect | FileCheck %s

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) attributes {byteir.entry_point = {inputs = ["module0_input0", "module0_input1"], outputs = ["module1_input1", "module1_input0"]}} {
    return %arg0, %arg1 : tensor<f32>, tensor<f32>
}
// CHECK: func.func @main
// CHECK-SAME: byteir.entry_point = {inputs = ["module0_input0", "module0_input1"], outputs = ["module1_output"]}
// CHECK-NEXT: %0:2 = call @__byteir__merge_model_0
// CHECK-NEXT: call @__byteir__merge_model_1(%0#1, %0#0)
// CHECK-DAG: func.func private @__byteir__merge_model_1
// CHECK-DAG: func.func private @__byteir__merge_model_0
