// RUN: onnx-frontend-opt -check-non-lowered %s -split-input-file | FileCheck %s

func.func @test_onnx_non_lowered(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  // expected-error @+1 {{onnx.NoValue: ONNX op is not lowered}}
  %0 = "onnx.NoValue"() : () -> none
  return %arg0 : tensor<1x2xf32>
}

func.func @test_onnx_lowered(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = mhlo.constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>
  %1 = mhlo.add %arg0, %0 : tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
  // CHECK-LABEL: func.func @test_onnx_lowered
}