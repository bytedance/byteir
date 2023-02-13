// RUN: byteir-opt %s -hlo-aggressive-fusion | FileCheck %s

func.func @mhlo_aggressive_fusion(%arg0 : tensor<32x32xf32>, %arg1 : tensor<32xi64>, %arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = "mhlo.torch_index_select"(%0, %arg1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<32x32xf32>, tensor<32xi64>) -> tensor<32x32xf32>
  %2 = "mhlo.add"(%1, %arg2) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @mhlo_aggressive_fusion
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.torch_index_select
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_hlo_aggressive_fusion__}
// CHECK:  return
