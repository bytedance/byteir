// RUN: byteir-opt -hlo-fusion-to-linalg -cse -split-input-file %s | FileCheck %s

func.func private @NextOffsetFunc() -> tensor<i64> attributes {byre_compute_name = "NextOffset", byre_force_compute_name}
func.func private @GetSeedFunc() -> tensor<i64> attributes {byre_compute_name = "GetSeed", byre_force_compute_name}
func.func @convert_rngUniform_static() -> tensor<8x1024x768xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = call @GetSeedFunc() : () -> tensor<i64>
    %3 = call @NextOffsetFunc() : ()-> tensor<i64>
    %4 = mhlo.custom_call @byteir.rng_uniform(%1, %0, %2, %3) {backend_config = ""} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>) -> tensor<8x1024x768xf32>
    return %4 : tensor<8x1024x768xf32>
  }
// CHECK-LABEL: func.func @convert_rngUniform_static
// CHECK: linalg.generic

func.func @convert_rngNormal_static() -> tensor<8x1024x768xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = call @GetSeedFunc() : () -> tensor<i64>
    %3 = call @NextOffsetFunc() : ()-> tensor<i64>
    %4 = mhlo.custom_call @byteir.rng_normal(%1, %0, %2, %3) {backend_config = ""} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>) -> tensor<8x1024x768xf32>
    return %4 : tensor<8x1024x768xf32>
  }
// CHECK-LABEL: func.func @convert_rngNormal_static
// CHECK: linalg.generic
