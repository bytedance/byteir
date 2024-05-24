// RUN: byteir-opt %s -canonicalize-ext | FileCheck %s

func.func @broadcast_to_broadcast_in_dim(%arg0: tensor<3xf32>) -> tensor<1x2x3xf32> {
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<3xf32>) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}
// CHECK-LABEL: func.func @broadcast_to_broadcast_in_dim
// CHECK-NEXT:  mhlo.broadcast_in_dim
// CHECK-NEXT:  return
