// RUN: byteir-opt %s -convert-op-to-customcall --split-input-file | FileCheck %s

func.func @convert_rng_static() -> tensor<8x1024x768xf32> {
  %16 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %17 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %18 = mhlo.constant dense<[8, 1024, 768]> : tensor<3xi64>
  %26 = "mhlo.rng"(%17, %16, %18) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<8x1024x768xf32>
  return %26 : tensor<8x1024x768xf32>
}
// CHECK-LABEL: func.func private @NextOffsetFunc() -> tensor<i64> attributes {byre_compute_name = "NextOffset", byre_force_compute_name}
// CHECK-LABEL: func.func private @GetSeedFunc() -> tensor<i64> attributes {byre_compute_name = "GetSeed", byre_force_compute_name}
// CHECK-LABEL: func.func @convert_rng_static
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  call @GetSeedFunc
// CHECK-NEXT:  call @NextOffsetFunc
// CHECK-NEXT:  mhlo.custom_call
// CHECK-SAME:  @byteir.rng_uniform

// -----

func.func @convert_two_rng_static() -> (tensor<8x1024x768xf32>, tensor<8x1024x768xf32>) {
  %16 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %17 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %18 = mhlo.constant dense<[8, 1024, 768]> : tensor<3xi64>
  %26 = "mhlo.rng"(%17, %16, %18) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<8x1024x768xf32>
  %27 = "mhlo.rng"(%17, %16, %18) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<8x1024x768xf32>
  return %26, %27 : tensor<8x1024x768xf32>, tensor<8x1024x768xf32>
}
// CHECK-LABEL: func.func private @NextOffsetFunc() -> tensor<i64> attributes {byre_compute_name = "NextOffset", byre_force_compute_name}
// CHECK-LABEL: func.func private @GetSeedFunc() -> tensor<i64> attributes {byre_compute_name = "GetSeed", byre_force_compute_name}
// CHECK-LABEL: func.func @convert_two_rng_static
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  call @GetSeedFunc
// CHECK-NEXT:  call @NextOffsetFunc
// CHECK-NEXT:  mhlo.custom_call
// CHECK-SAME:  @byteir.rng_uniform
// CHECK-NEXT:  call @NextOffsetFunc
// CHECK-NEXT:  mhlo.custom_call
// CHECK-SAME:  @byteir.rng_uniform

// -----

func.func @convert_rng_dynamic(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %16 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %17 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %shape = shape.shape_of %arg0 : tensor<?x?x?xf32> -> tensor<3xindex>
  %shape1 = arith.index_cast %shape : tensor<3xindex> to tensor<3xi64>
  %26 = "mhlo.rng"(%17, %16, %shape1) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %26 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func.func private @NextOffsetFunc() -> tensor<i64> attributes {byre_compute_name = "NextOffset", byre_force_compute_name}
// CHECK-LABEL: func.func private @GetSeedFunc() -> tensor<i64> attributes {byre_compute_name = "GetSeed", byre_force_compute_name}
// CHECK-LABEL: func.func @convert_rng_dynamic
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  call @GetSeedFunc
// CHECK-NEXT:  shape.shape_of
// CHECK-NEXT:  arith.index_cast
// CHECK-NEXT:  call @NextOffsetFunc
// CHECK-NEXT:  mhlo.custom_call
// CHECK-SAME:  @byteir.rng_uniform

// -----

func.func @flash_attn_fwd(%arg0: tensor<2x256x12x128xf16>, %arg1: tensor<2x256x12x128xf16>, %arg2: tensor<2x256x12x128xf16>) -> (tensor<2x256x12x128xf16>, tensor<2x12x256xf32>, tensor<2x12x256x256xf16>, tensor<2xi64>) {
  %0:4 = mhlo.custom_call @byteir.flash_attn_fwd(%arg0, %arg1, %arg2) {backend_config = "", byteir_attrs = {casual = false, dropout_p = 1.000000e-01 : f64, return_softmax = false, softmax_scale = 1.000000e+00 : f64}} : (tensor<2x256x12x128xf16>, tensor<2x256x12x128xf16>, tensor<2x256x12x128xf16>) -> (tensor<2x256x12x128xf16>, tensor<2x12x256xf32>, tensor<2x12x256x256xf16>, tensor<2xi64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<2x256x12x128xf16>, tensor<2x12x256xf32>, tensor<2x12x256x256xf16>, tensor<2xi64>
}

// CHECK-LABEL: func.func private @NextOffsetFunc() -> tensor<i64> attributes {byre_compute_name = "NextOffset", byre_force_compute_name}
// CHECK-LABEL: func.func private @GetSeedFunc() -> tensor<i64> attributes {byre_compute_name = "GetSeed", byre_force_compute_name}
// CHECK-LABEL: func.func @flash_attn_fwd
// CHECK-NEXT:  call @GetSeedFunc
// CHECK-NEXT:  call @NextOffsetFunc
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.concatenate
// CHECK-NEXT:  mhlo.custom_call
// CHECK-SAME:  @byteir.flash_attn_fwd
// CHECK-SAME:  byteir_attrs = {casual = false, dropout_p = 1.000000e-01 : f64, return_softmax = false, softmax_scale = 1.000000e+00 : f64}
