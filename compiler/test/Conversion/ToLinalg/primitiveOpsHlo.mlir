// RUN: byteir-opt -hlo-fusion-to-linalg="enable-primitive-ops=true" %s | FileCheck %s

// CHECK-LABEL: mhlo_add
func.func @mhlo_add(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "mhlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
  // CHECK: tensor.empty
  // CHECK: linalg.map
  // CHECK: arith.addf
}
