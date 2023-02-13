// RUN: byteir-opt %s -linalg-ext-bufferize | FileCheck %s

func.func @scan_1d(%0: tensor<128xi32>) -> tensor<128xi32> {
  %c0 = tensor.empty() : tensor<i32>
  %1 = tensor.empty() : tensor<128xi32>
  %2:2 = linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : tensor<128xi32>) outs(%1, %c0 : tensor<128xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  } -> tensor<128xi32>, tensor<i32>
  return %2#0 : tensor<128xi32>
}
//CHECK-LABEL: func.func @scan_1d
//CHECK: linalg_ext.scan
//CHECK-SAME: memref<128xi32>

func.func @scan_2d(%0: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %c0 = tensor.empty() : tensor<32xi32>
  %1 = tensor.empty() : tensor<16x32xi32>
  %2:2 = linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : tensor<16x32xi32>) outs(%1, %c0 : tensor<16x32xi32>, tensor<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  } -> tensor<16x32xi32>, tensor<32xi32>
  return %2#0 : tensor<16x32xi32>
}
//CHECK-LABEL: func.func @scan_2d
//CHECK: linalg_ext.scan
//CHECK-SAME: memref<16x32xi32>


func.func @softmax(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<64xf32>
  %2 = tensor.empty() : tensor<64xf32>
  %3 = tensor.empty() : tensor<64xf32>
  %4:4 = linalg_ext.softmax 
    dimension(0) 
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %1, %2, %3 : tensor<1024x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) : tensor<1024x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
  return %4#0 : tensor<1024x64xf32>
}
//CHECK-LABEL: func.func @softmax
//CHECK: linalg_ext.softmax
//CHECK-SAME: memref<1024x64xf32>

func.func @diag(%arg0: tensor<1024xf32>) -> (tensor<1024x1024xf32>) {
  %0 = tensor.empty() : tensor<1024x1024xf32>
  %1 = linalg_ext.diag 
    ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024x1024xf32>) : tensor<1024x1024xf32>
  return %1 : tensor<1024x1024xf32>
}
//CHECK-LABEL: func.func @diag
//CHECK: linalg_ext.diag
//CHECK-SAME: memref<1024xf32>

func.func @batch_matmul(%arg0: tensor<128x16x1024x32xf32>, %arg1: tensor<128x16x32x512xf32>) -> (tensor<128x16x1024x512xf32>) {
  %0 = tensor.empty() : tensor<128x16x1024x512xf32>
  %1 = linalg_ext.batch_matmul ins(%arg0, %arg1 : tensor<128x16x1024x32xf32>, tensor<128x16x32x512xf32>) outs(%0 : tensor<128x16x1024x512xf32>) layout = "nn"
  return %1 : tensor<128x16x1024x512xf32>
}
//CHECK-LABEL: func.func @batch_matmul
//CHECK: linalg_ext.batch_matmul
//CHECK-SAME: memref<128x16x1024x32xf32>

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @alias_in_generic(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32): 
      %4 = linalg_ext.alias(%arg3 : f32) : f32    
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
//CHECK-LABEL: func.func @alias_in_generic
//CHECK: linalg.generic 
//CHECK-SAME: memref<?x?xf32>
//CHECK: linalg_ext.alias
