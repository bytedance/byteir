// RUN: byteir-opt %s --decompose-mhlo-custom-call-ops | FileCheck %s

func.func @byteir.addn(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.custom_call @byteir.addn(%arg0, %arg1, %arg2) {byteir_attrs = {}} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @byteir.addn
// CHECK-NOT:  byteir.addn
// CHECK:      mhlo.add
// CHECK:      mhlo.add
// CHECK:      return

