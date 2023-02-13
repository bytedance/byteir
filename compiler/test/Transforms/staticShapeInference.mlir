// RUN: byteir-opt %s -static-shape-infer | FileCheck %s

// CHECK-LABEL: @InferShapedTypeOpInterface
func.func @InferShapedTypeOpInterface(%arg0 : tensor<8x4xi32>, %arg1 : tensor<8x4xi32>) -> tensor<?x4xi1> {
// CHECK-NEXT: %0 = mhlo.compare LT, %arg0, %arg1 : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<8x4xi1>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<?x4xi1>
// CHECK-NEXT: return %0 : tensor<8x4xi1>
  return %0 : tensor<?x4xi1>
}