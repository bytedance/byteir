// RUN: byteir-opt -hlo-opt %s | FileCheck %s

func.func @uniform_rngf32() -> tensor<2x128x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<[2, 128, 128]> : tensor<3xi64>
    %3 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x128x128xf32>
    %4 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x128x128xf32>
    %5 = mhlo.add %3, %4 : tensor<2x128x128xf32>
    return %5 : tensor<2x128x128xf32>
}

// CHECK-LABEL: func.func private @RngUniform_f32f32_f320
//   CHECK-DAG: __byre__high = 1.000000e+00
//   CHECK-DAG: __byre__low = 0.000000e+00
//   CHECK-DAG: byre_compute_name = "RngUniform_f32f32_f32"
//   CHECK-DAG: byre_force_compute_name

// CHECK-LABEL: func.func private @RngUniform_f32f32_f321
//   CHECK-DAG: __byre__high = 1.000000e+00
//   CHECK-DAG: __byre__low = 0.000000e+00
//   CHECK-DAG: byre_compute_name = "RngUniform_f32f32_f32"
//   CHECK-DAG: byre_force_compute_name

// CHECK-LABEL: func.func @uniform_rngf32
//   CHECK: %[[VAR_0:.*]] = call @RngUniform_f32f32_f320
//   CHECK: %[[VAR_1:.*]] = call @RngUniform_f32f32_f321
//   CHECK: %[[VAR_2:.*]] = mhlo.add %[[VAR_0]], %[[VAR_1]]
//   CHECK: return %[[VAR_2]]

func.func @uniform_rngf64() -> tensor<2x128x128xf64> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f64>
    %2 = mhlo.constant dense<[2, 128, 128]> : tensor<3xi64>
    %3 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<2x128x128xf64>
    %4 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<2x128x128xf64>
    %5 = mhlo.add %3, %4 : tensor<2x128x128xf64>
    return %5 : tensor<2x128x128xf64>
}

// CHECK-LABEL: func.func private @RngUniform_f64f64_f642
//   CHECK-DAG: __byre__high = 1.000000e+00
//   CHECK-DAG: __byre__low = 0.000000e+00
//   CHECK-DAG: byre_compute_name = "RngUniform_f64f64_f64"
//   CHECK-DAG: byre_force_compute_name

// CHECK-LABEL: func.func private @RngUniform_f64f64_f643
//   CHECK-DAG: __byre__high = 1.000000e+00
//   CHECK-DAG: __byre__low = 0.000000e+00
//   CHECK-DAG: byre_compute_name = "RngUniform_f64f64_f64"
//   CHECK-DAG: byre_force_compute_name

// CHECK-LABEL: func.func @uniform_rngf64
//   CHECK: %[[VAR_0:.*]] = call @RngUniform_f64f64_f642
//   CHECK: %[[VAR_1:.*]] = call @RngUniform_f64f64_f643
//   CHECK: %[[VAR_2:.*]] = mhlo.add %[[VAR_0]], %[[VAR_1]]
//   CHECK: return %[[VAR_2]]
