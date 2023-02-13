// RUN: byteir-opt %s -test-convert-func-to-custom | FileCheck %s

func.func private @test.test_name.123(%arg0: tensor<4x?xf32>, %arg1: tensor<4x?xf32>, %arg2: tensor<4x?xi64>) -> tensor<2x128x128xf32> 
// CHECK-NOT: func.func private @test.test_name.123
func.func private @test.other_name(%arg0: tensor<4x?xf32>, %arg1: tensor<4x?xf32>, %arg2: tensor<4x?xi64>) -> tensor<2x128x128xf32> 
// CHECK-LABEL: func.func private @test.other_name

func.func @main1(%arg0: tensor<4x?xf32>, %arg1: tensor<4x?xf32>, %arg2: tensor<4x?xi64>) -> tensor<2x128x128xf32> {
  %0 = call @test.test_name.123(%arg0, %arg1, %arg2) : (tensor<4x?xf32>, tensor<4x?xf32>, tensor<4x?xi64>) -> tensor<2x128x128xf32>
  return %0 : tensor<2x128x128xf32>
}
// CHECK-LABEL:  func.func @main1
// CHECK-NEXT: mhlo.custom_call
// CHEKC-SAME: call_target_name = "TestName"
// CHECK-NOT: call @test.test_name.123


func.func @main2(%arg0: tensor<4x?xf32>, %arg1: tensor<4x?xi64>) -> tensor<2x128x128xf32> {
  %0 = call @test.other_name(%arg0, %arg0, %arg1) : (tensor<4x?xf32>, tensor<4x?xf32>, tensor<4x?xi64>) -> tensor<2x128x128xf32>
  return %0 : tensor<2x128x128xf32>
}
// CHECK-LABEL:  func.func @main2
// CHECK-NEXT: call @test.other_name
