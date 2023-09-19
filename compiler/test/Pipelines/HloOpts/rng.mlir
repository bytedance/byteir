// RUN: byteir-opt -hlo-opt %s | FileCheck %s

module @uniform_rng {
    func.func @uniform_rngf32() -> tensor<2x128x128xf32> {
        %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
        %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
        %2 = mhlo.constant dense<[2, 128, 128]> : tensor<3xi64>
        %3 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x128x128xf32>
        %4 = mhlo.add %3, %3 : tensor<2x128x128xf32>
        return %4 : tensor<2x128x128xf32>
    }
    func.func @uniform_rngf64() -> tensor<2x128x128xf64> {
        %0 = mhlo.constant dense<0.000000e+00> : tensor<f64>
        %1 = mhlo.constant dense<1.000000e+00> : tensor<f64>
        %2 = mhlo.constant dense<[2, 128, 128]> : tensor<3xi64>
        %3 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<2x128x128xf64>
        %4 = mhlo.add %3, %3 : tensor<2x128x128xf64>
        return %4 : tensor<2x128x128xf64>
    }
}

// CHECK-LABEL: func.func private @NextOffsetFunc() -> tensor<i64> attributes {byre_compute_name = "NextOffset", byre_force_compute_name}
// CHECK-LABEL: func.func private @GetSeedFunc() -> tensor<i64> attributes {byre_compute_name = "GetSeed", byre_force_compute_name}

// CHECK-LABEL: func.func private @Unknown0
//   CHECK-DAG: mhlo.constant
//   CHECK-DAG: mhlo.constant
//   CHECK-DAG: mhlo.custom_call
//   CHECK-DAG: mhlo.add

// CHECK-LABEL: func.func @uniform_rngf32
//   CHECK: %[[VAR_0:.*]] = call @GetSeedFunc()
//   CHECK: %[[VAR_1:.*]] = call @NextOffsetFunc()
//   CHECK: %[[VAR_2:.*]] = call @Unknown0(%[[VAR_0]], %[[VAR_1]]) 
//   CHECK: return %[[VAR_2]]

// CHECK-LABEL: func.func private @Unknown1
//   CHECK-DAG: mhlo.constant
//   CHECK-DAG: mhlo.constant
//   CHECK-DAG: mhlo.custom_call
//   CHECK-DAG: mhlo.add

// CHECK-LABEL: func.func @uniform_rngf64
//   CHECK: %[[VAR_3:.*]] = call @GetSeedFunc()
//   CHECK: %[[VAR_4:.*]] = call @NextOffsetFunc()
//   CHECK: %[[VAR_5:.*]] = call @Unknown1(%[[VAR_3]], %[[VAR_4]]) 
//   CHECK: return %[[VAR_5]]
