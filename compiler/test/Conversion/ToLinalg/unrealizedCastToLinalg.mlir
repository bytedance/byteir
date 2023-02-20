// RUN: byteir-opt -hlo-legalize-to-linalg -unrealized-cast-to-linalg -linalg-fuse-elementwise-ops %s | FileCheck %s

// CHECK-LABEL: test_unrealized_conversion_cast
func.func @test_unrealized_conversion_cast(%arg0: tensor<2176xi32>) -> (tensor<2176xui32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<2176xi32> to tensor<2176xui32>
  return %0 : tensor<2176xui32>
}
// CHECK: linalg.generic
// CHECK:  builtin.unrealized_conversion_cast
// CHECK-SAME:  i32 to ui32
// CHECK-NEXT:  linalg.yield

// CHECK-LABEL: test_shift_right_logical
func.func @test_shift_right_logical(%arg0: tensor<17x128x768xui32>, %arg1: tensor<17x128x768xui32>) -> (tensor<17x128x768xui32>) {
  %0 = mhlo.shift_right_logical %arg0, %arg1 : tensor<17x128x768xui32>
  return %0 : tensor<17x128x768xui32>
}
// CHECK: linalg.generic
// CHECK:  builtin.unrealized_conversion_cast
// CHECK-NEXT:  builtin.unrealized_conversion_cast
// CHECK-NEXT:  arith.shrui
// CHECK-NEXT:  arith.cmpi ult
// CHECK-NEXT:  arith.select
// CHECK-NEXT:  builtin.unrealized_conversion_cast
// CHECK-NEXT:  linalg.yield


