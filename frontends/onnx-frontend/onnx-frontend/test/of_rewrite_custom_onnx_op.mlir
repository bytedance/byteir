// RUN: onnx-frontend-opt -rewrite-custom-onnx-ops -of-canonicalize -constprop-onnx -of-canonicalize %s -split-input-file | FileCheck %s

func.func @test_quantize16(%arg0: tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xui16> {
  %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<"0x1F1F1F1F"> : tensor<f32>} : () -> tensor<f32>
  %1 = "onnx.Constant"() {onnx_node_name = "Constant_1", value = dense<"0x0000"> : tensor<ui16>} : () -> tensor<ui16>
  %2 = "onnx.Custom"(%arg0, %0, %1) {domain_name = "", function_name = "quantize", inputs_for_infer = [0], onnx_node_name = "QuantLinear", shape_infer_pattern = "SameAs"} : (tensor<1x128x1x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x1x1xui16>
  return %2 : tensor<1x128x1x1xui16>
  // CHECK-LABEL: @test_quantize16(
  // CHECK-SAME:   [[ARG0:%.*]]: tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xui16> {
  // CHECK-DAG:    [[CONST0:%.*]] = onnx.Constant dense<3.36953024E-20> : tensor<f32>
  // CHECK-DAG:    [[CONST1:%.*]] = onnx.Constant dense<0> : tensor<ui16>
  // CHECK-NEXT:   [[CUSTOM:%.*]] = stablehlo.custom_call @byteir.quantize([[ARG0]], [[CONST0]], [[CONST1]]) {byteir_attrs = {}} : (tensor<1x128x1x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x1x1xui16>
  // CHECK-NEXT:   return [[CUSTOM]] : tensor<1x128x1x1xui16>
}

// -----

func.func @test_dequantize16(%arg0: tensor<1x128x1x1xui16>) -> tensor<1x128x1x1xf32> {
  %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<"0x1F1F1F1F"> : tensor<f32>} : () -> tensor<f32>
  %1 = "onnx.Constant"() {onnx_node_name = "Constant_1", value = dense<"0x0000"> : tensor<ui16>} : () -> tensor<ui16>
  %2 = "onnx.Custom"(%arg0, %0, %1) {domain_name = "", function_name = "dequantize", inputs_for_infer = [0], onnx_node_name = "DeQuantLinear", shape_infer_pattern = "SameAs"} : (tensor<1x128x1x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x1x1xf32>
  return %2 : tensor<1x128x1x1xf32>
  // CHECK-LABEL: @test_dequantize16(
  // CHECK-SAME:   [[ARG0:%.*]]: tensor<1x128x1x1xui16>) -> tensor<1x128x1x1xf32> {
  // CHECK-DAG:    [[CONST0:%.*]] = onnx.Constant dense<3.36953024E-20> : tensor<f32>
  // CHECK-DAG:    [[CONST1:%.*]] = onnx.Constant dense<0> : tensor<ui16>
  // CHECK-NEXT:   [[CUSTOM:%.*]] = stablehlo.custom_call @byteir.dequantize([[ARG0]], [[CONST0]], [[CONST1]]) {byteir_attrs = {}} : (tensor<1x128x1x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x1x1xf32>
  // CHECK-NEXT:   return [[CUSTOM]] : tensor<1x128x1x1xf32>
}

func.func @test_quantize16_typed(%arg0: tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xi16> {
  %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<"0x1F1F1F1F"> : tensor<f32>} : () -> tensor<f32>
  %1 = "onnx.Constant"() {onnx_node_name = "Constant_1", value = dense<"0x0000"> : tensor<i16>} : () -> tensor<i16>
  %2 = "onnx.Custom"(%arg0, %0, %1) {domain_name = "", function_name = "quantize", inputs_for_infer = [0], onnx_node_name = "QuantLinear", shape_infer_pattern = "SameAs"} : (tensor<1x128x1x1xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x1x1xi16>
  return %2 : tensor<1x128x1x1xi16>
  // CHECK-LABEL: @test_quantize16_typed(
  // CHECK-SAME:   [[ARG0:%.*]]: tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xi16> {
  // CHECK-DAG:    [[CONST0:%.*]] = onnx.Constant dense<3.36953024E-20> : tensor<f32>
  // CHECK-DAG:    [[CONST1:%.*]] = onnx.Constant dense<0> : tensor<i16>
  // CHECK-NEXT:   [[CUSTOM:%.*]] = stablehlo.custom_call @byteir.quantize([[ARG0]], [[CONST0]], [[CONST1]]) {byteir_attrs = {}} : (tensor<1x128x1x1xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x1x1xi16>
  // CHECK-NEXT:   return [[CUSTOM]] : tensor<1x128x1x1xi16>
}

// -----

func.func @test_dequantize16_typed(%arg0: tensor<1x128x1x1xi16>) -> tensor<1x128x1x1xf32> {
  %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<"0x1F1F1F1F"> : tensor<f32>} : () -> tensor<f32>
  %1 = "onnx.Constant"() {onnx_node_name = "Constant_1", value = dense<"0x0000"> : tensor<i16>} : () -> tensor<i16>
  %2 = "onnx.Custom"(%arg0, %0, %1) {domain_name = "", function_name = "dequantize", inputs_for_infer = [0], onnx_node_name = "DeQuantLinear", shape_infer_pattern = "SameAs"} : (tensor<1x128x1x1xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x1x1xf32>
  return %2 : tensor<1x128x1x1xf32>
  // CHECK-LABEL: @test_dequantize16_typed(
  // CHECK-SAME:   [[ARG0:%.*]]: tensor<1x128x1x1xi16>) -> tensor<1x128x1x1xf32> {
  // CHECK-DAG:    [[CONST0:%.*]] = onnx.Constant dense<3.36953024E-20> : tensor<f32>
  // CHECK-DAG:    [[CONST1:%.*]] = onnx.Constant dense<0> : tensor<i16>
  // CHECK-NEXT:   [[CUSTOM:%.*]] = stablehlo.custom_call @byteir.dequantize([[ARG0]], [[CONST0]], [[CONST1]]) {byteir_attrs = {}} : (tensor<1x128x1x1xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x1x1xf32>
  // CHECK-NEXT:   return [[CUSTOM]] : tensor<1x128x1x1xf32>
}
