// RUN: byteir-opt -hlo-legalize-to-linalg %s | FileCheck %s

// CHECK-LABEL: mhlo_add
func.func @mhlo_add(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "mhlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
  // CHECK: linalg.generic
  // CHECK: addf
}
