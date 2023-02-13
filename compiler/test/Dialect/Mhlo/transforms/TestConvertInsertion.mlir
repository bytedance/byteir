// RUN: byteir-opt %s -test-insert-convert | FileCheck %s

func.func private @test_func(%arg0: tensor<4x?xf32>, %arg1: tensor<4x?xf32>, %arg2: tensor<4x?xi64>) -> tensor<2x128x128xf32> attributes {__byteir_unit_test__}
// CHECK-LABEL: func.func private @test_func
// CHECK-SAME: (tensor<4x?xf16>, tensor<4x?xf16>, tensor<4x?xi64>) -> tensor<2x128x128xf16>

func.func @main1(%arg0: tensor<4x?xf32>, %arg1: tensor<4x?xf32>, %arg2: tensor<4x?xi64>) -> tensor<2x128x128xf32> {
  %0 = call @test_func(%arg0, %arg1, %arg2) : (tensor<4x?xf32>, tensor<4x?xf32>, tensor<4x?xi64>) -> tensor<2x128x128xf32>
  return %0 : tensor<2x128x128xf32>
}
// CHECK-LABEL:  func.func @main1
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x?xf32>, %[[ARG1:.*]]: tensor<4x?xf32>, %[[ARG2:.*]]: tensor<4x?xi64>) -> tensor<2x128x128xf32> {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.convert %[[ARG0]] : (tensor<4x?xf32>) -> tensor<4x?xf16>
// CHECK-NEXT:   %[[V1:.*]] = mhlo.convert %[[ARG1]] : (tensor<4x?xf32>) -> tensor<4x?xf16>
// CHECK-NEXT:   %[[V2:.*]] = call @test_func(%[[V0]], %[[V1]], %[[ARG2]]) : (tensor<4x?xf16>, tensor<4x?xf16>, tensor<4x?xi64>) -> tensor<2x128x128xf16>
// CHECK-NEXT:   %[[V3:.*]] = mhlo.convert %[[V2]] : (tensor<2x128x128xf16>) -> tensor<2x128x128xf32>
// CHECK-NEXT:   return %[[V3]] : tensor<2x128x128xf32>


func.func @main2(%arg0: tensor<4x?xf32>, %arg1: tensor<4x?xi64>) -> tensor<2x128x128xf32> {
  %0 = call @test_func(%arg0, %arg0, %arg1) : (tensor<4x?xf32>, tensor<4x?xf32>, tensor<4x?xi64>) -> tensor<2x128x128xf32>
  return %0 : tensor<2x128x128xf32>
}
// CHECK-LABEL:  func.func @main2
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x?xf32>, %[[ARG1:.*]]: tensor<4x?xi64>) -> tensor<2x128x128xf32> {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.convert %[[ARG0]] : (tensor<4x?xf32>) -> tensor<4x?xf16>
// CHECK-NEXT:   %[[V1:.*]] = call @test_func(%[[V0]], %[[V0]], %[[ARG1]]) : (tensor<4x?xf16>, tensor<4x?xf16>, tensor<4x?xi64>) -> tensor<2x128x128xf16>
// CHECK-NEXT:   %[[V2:.*]] = mhlo.convert %[[V1]] : (tensor<2x128x128xf16>) -> tensor<2x128x128xf32>
// CHECK-NEXT:   return %[[V2]] : tensor<2x128x128xf32>
