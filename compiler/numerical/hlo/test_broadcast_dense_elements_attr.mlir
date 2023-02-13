// RUN: byteir-opt %s -test-broadcast-dense-elements-attr -o %t
// RUN: FileCheck %s < %t
// RUN: python3 %S/numerical_test.py %s %t

func.func @case3() -> tensor<2x1x5xi64> {
  %0 = mhlo.constant dense<[[[2, 3]]]> : tensor<1x1x2xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1, 2, 0]> : tensor<3xi64>} : (tensor<1x1x2xi64>) -> (tensor<2x1x5xi64>)
  return %1 : tensor<2x1x5xi64>
}
// CHECK-LABEL: @case3
// CHECK{LITERAL}: [[[2, 2, 2, 2, 2]], [[3, 3, 3, 3, 3]]]
// CHECK-NOT: mhlo.broadcast_in_dim
