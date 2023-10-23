// RUN: byteir-opt %s --test-merge-two-modules="second-module-path=%S/testMergeTwoModulesCase0_1.mlir" --allow-unregistered-dialect | FileCheck %s

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
}
// CHECK: func.func @main
// CHECK-NEXT: call @__byteir__merge_model_0
// CHECK-NEXT: call @__byteir__merge_model_1
// CHECK-DAG: func.func private @__byteir__merge_model_1
// CHECK-DAG: func.func private @__byteir__merge_model_0
