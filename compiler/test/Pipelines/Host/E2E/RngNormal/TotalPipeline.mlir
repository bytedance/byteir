// RUN: byteir-opt %s --hlo-opt="target=CPU" --linalg-tensor-opt="target=CPU" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --scf-opt="target=CPU" --host-opt --byre-opt --to-llvm | byteir-translate --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @_mlir_ciface_Unknown

func.func @main() -> tensor<1x97xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<[1, 97]> : tensor<2xi64>
    %3 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<NORMAL>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
    return %3 : tensor<1x97xf32>
}