// RUN: byteir-opt %s -convert-hlo-to-lhlo -cse | FileCheck %s


func.func private @Unknown0(%arg0: tensor<32xf32>, %arg1: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<128x32xf32>
  %1 = mhlo.add %arg1, %0 : tensor<128x32xf32>
  return %1 : tensor<128x32xf32>
}
// CHECK-LABEL: func.func private @Unknown0
// CHECK-NEXT: memref.alloc
// CHECK-NEXT: lmhlo.broadcast_in_dim
// CHECK-NEXT: memref.alloc
// CHECK-NEXT: lmhlo.add
// CHECK-NEXT: return


func.func @mlp(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<32xf32>) -> tensor<128x32xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
  %1 = call @Unknown0(%arg2, %0) : (tensor<32xf32>, tensor<128x32xf32>) -> tensor<128x32xf32>
  return %1 : tensor<128x32xf32>
}
// CHECK-LABEL: func.func @mlp
// CHECK-NEXT: memref.alloc
// CHECK-NEXT: lmhlo.dot
// CHECK-NEXT: call
// CHECK-NEXT: return
