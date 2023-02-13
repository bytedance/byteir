// RUN: byteir-opt %s | FileCheck %s

func.func @extension_type(%arg0: tensor<3x?xf32, #ace.tensor_encoding<is_dynamic = [0, 1]>>) -> tensor<3x?xf32, #ace.tensor_encoding<is_dynamic = [0, 1]>> {
  return %arg0 : tensor<3x?xf32, #ace.tensor_encoding<is_dynamic = [0, 1]>>
}
// CHECK: tensor<3x?xf32, #ace.tensor_encoding<is_dynamic = [0, 1]>>