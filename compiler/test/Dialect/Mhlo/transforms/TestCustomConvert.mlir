// RUN: byteir-opt %s -test-custom-convert --canonicalize | FileCheck %s

func.func private @test_func(%arg0: tensor<4x?xi64>, %arg1: tensor<4x?xi64>, %arg2: tensor<4x?xi64>) -> tensor<2x128x128xi64> attributes {__byteir_unit_test__}
// CHECK-LABEL: func.func private @test_func
// CHECK-SAME: (tensor<4x?xi16>, tensor<4x?xi64>, tensor<4x?xi64>) -> tensor<2x128x128xi16>

func.func @main1(%arg0: tensor<4x?xi64>, %arg1: tensor<4x?xi64>, %arg2: tensor<4x?xi64>) -> tensor<2x128x128xi64> {
  %0 = call @test_func(%arg0, %arg1, %arg2) : (tensor<4x?xi64>, tensor<4x?xi64>, tensor<4x?xi64>) -> tensor<2x128x128xi64>
  return %0 : tensor<2x128x128xi64>
}
// CHECK-LABEL:  func.func @main1
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x?xi64>, %[[ARG1:.*]]: tensor<4x?xi64>, %[[ARG2:.*]]: tensor<4x?xi64>) -> tensor<2x128x128xi64> {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.convert %[[ARG0]] : (tensor<4x?xi64>) -> tensor<4x?xi16>
// CHECK-NEXT:   %[[V1:.*]] = call @test_func(%[[V0]], %[[ARG1]], %[[ARG2]]) : (tensor<4x?xi16>, tensor<4x?xi64>, tensor<4x?xi64>) -> tensor<2x128x128xi16>
// CHECK-NEXT:   %[[V2:.*]] = mhlo.convert %[[V1]] : (tensor<2x128x128xi16>) -> tensor<2x128x128xi64>
// CHECK-NEXT:   return %[[V2]] : tensor<2x128x128xi64>

func.func @main2(%arg0: tensor<4x?xi64>, %arg1: tensor<4x?xi64>) -> tensor<2x128x128xi64> {
  %0 = call @test_func(%arg0, %arg0, %arg1) : (tensor<4x?xi64>, tensor<4x?xi64>, tensor<4x?xi64>) -> tensor<2x128x128xi64>
  return %0 : tensor<2x128x128xi64>
}
// CHECK-LABEL:  func.func @main2
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x?xi64>, %[[ARG1:.*]]: tensor<4x?xi64>) -> tensor<2x128x128xi64> {
// CHECK-NEXT:   %[[V0:.*]] = mhlo.convert %[[ARG0]] : (tensor<4x?xi64>) -> tensor<4x?xi16>
// CHECK-NEXT:   %[[V1:.*]] = call @test_func(%[[V0]], %[[ARG0]], %[[ARG1]]) : (tensor<4x?xi16>, tensor<4x?xi64>, tensor<4x?xi64>) -> tensor<2x128x128xi16>
// CHECK-NEXT:   %[[V2:.*]] = mhlo.convert %[[V1]] : (tensor<2x128x128xi16>) -> tensor<2x128x128xi64>
// CHECK-NEXT:   return %[[V2]] : tensor<2x128x128xi64>
