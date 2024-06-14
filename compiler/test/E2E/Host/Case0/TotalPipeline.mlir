// RUN: byteir-opt %s --hlo-graph-opt --hlo-opt="target=CPU" --linalg-tensor-opt="target=CPU" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --scf-opt="target=CPU" --host-opt --byre-opt --to-llvm | byteir-translate --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: constant
// CHECK-LABEL: define void @_mlir_ciface_Unknown

func.func @main(%arg0: tensor<1xi64>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>, %arg3: tensor<1x128xi32>) -> (tensor<1x128xi32>, tensor<1x128xi32>) {
    %0 = mhlo.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]]> : tensor<1x128xi32>
    %1 = mhlo.add %arg0, %arg1 : tensor<1xi64>
    %2 = mhlo.add %arg2, %1 : tensor<1xi64>
    %3 = "mhlo.convert"(%2) : (tensor<1xi64>) -> tensor<1xi32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1x128xi32>
    %5 = "mhlo.compare"(%0, %4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1x128xi32>, tensor<1x128xi32>) -> tensor<1x128xi1>
    %6 = "mhlo.convert"(%5) : (tensor<1x128xi1>) -> tensor<1x128xi32>
    %7 = mhlo.multiply %6, %arg3 {device = "host"} : tensor<1x128xi32>
    return %6, %7 : tensor<1x128xi32>, tensor<1x128xi32>
}