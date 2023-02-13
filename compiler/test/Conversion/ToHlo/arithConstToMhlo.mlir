// RUN: byteir-opt %s -convert-arith-to-mhlo | FileCheck %s

// CHECK-LABEL: func.func @const
func.func @const() -> tensor<4x4xf32> {
  // CHECK: mhlo.constant
  %0 = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @not_mhlo_const
func.func @not_mhlo_const() -> i32 {
  // CHECK-NOT: mhlo.constant
  %0 = arith.constant 1 : i32
  return %0 : i32
}
