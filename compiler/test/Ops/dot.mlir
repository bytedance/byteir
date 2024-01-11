// RUN: byteir-opt -hlo-transpose-dot-to-dot-general -fusion-outlining -byre-tensor-opt --byteir-bufferize-opt -convert-to-byre %s | FileCheck %s

func.func @dot(%arg0 : tensor<64x128xf32> {__placeholder__byre.argname = "A"}, %arg1 : tensor<64x32xf32> {__placeholder__byre.argname = "B"}) -> (tensor<128x32xf32> {__placeholder__byre.argname = "C"}) attributes {__placeholder__byre.entry_point} {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %1 = "mhlo.dot"(%0, %arg1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
    return %1 : tensor<128x32xf32>
}
// CHECK-LABEL: func.func @dot
// CHECK:  byre.compute @MatmulOp
//   CHECK-DAG: lhs_contracting_dimension = 0 : i64
//   CHECK-DAG: rhs_contracting_dimension = 0 : i64
//   CHECK-DAG: memory_effects = [1 : i32, 1 : i32, 2 : i32]
