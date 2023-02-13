// RUN: byteir-opt -cluster-constraint %s | FileCheck %s

func.func @uniform_rng() -> tensor<2x128x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<[2, 128, 128]> : tensor<3xi64>
    %3 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x128x128xf32>
    return %3 : tensor<2x128x128xf32>
}

// CHECK-LABEL: func.func @uniform_rng
//   CHECK: "mhlo.fusion"
//   CHECK:   "mhlo.rng"
//   CHECK-DAG: __byre__high = 1.000000e+00
//   CHECK-DAG: __byre__low = 0.000000e+00
//   CHECK-DAG: byre_compute_name = "RngUniform"
//   CHECK-DAG: byre_force_compute_name
