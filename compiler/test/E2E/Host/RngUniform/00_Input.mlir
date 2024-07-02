// RUN: byteir-opt %s --hlo-graph-opt --hlo-fusion-opt="target=cpu" --linalg-tensor-opt="target=cpu" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --linalg-memref-opt --scf-opt="target=cpu" | FileCheck %s

// CHECK-LABEL: func.func @main

func.func @main() -> tensor<1x97xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<[1, 97]> : tensor<2xi64>
    %3 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
    return %3 : tensor<1x97xf32>
}