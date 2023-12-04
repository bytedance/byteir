// RUN: onnx-frontend-opt -check-non-lowered %s -split-input-file -verify-diagnostics

func.func @test_onnx_non_lowered(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  // expected-warning @+2 {{onnx.NoValue: ONNX op is not lowered}}
  // expected-error @-2 {{Please lower all ONNX ops}}
  %0 = "onnx.NoValue"() : () -> none
  return %arg0 : tensor<1x2xf32>
}
