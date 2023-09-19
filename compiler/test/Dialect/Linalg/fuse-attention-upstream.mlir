// RUN: byteir-opt %s --transform-dialect-interpreter --canonicalize-ext --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @dot_attention
func.func @dot_attention(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x32xf32>
  %2 = tensor.empty() : tensor<1024xf32>
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         linalg.matmul
// CHECK:         linalg.matmul
// CHECK:         linalg.reduce
// CHECK:         linalg.generic
// CHECK:         linalg.matmul
// CHECK:         linalg.matmul
// CHECK:         linalg.reduce
// CHECK:         linalg.generic
// CHECK:         linalg.reduce
// CHECK:         linalg.generic
// CHECK:         linalg.matmul
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %7 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x32xf32>, tensor<32x512xf32>) outs(%4 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %reduced = linalg.reduce ins(%7 : tensor<1024x512xf32>) outs(%3 : tensor<1024xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %11 = arith.maxf %in, %init : f32
      linalg.yield %11 : f32
    }
  %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %reduced : tensor<1024x512xf32>, tensor<1024xf32>) outs(%0 : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %11 = arith.subf %in, %in_2 : f32
    %12 = math.exp %11 : f32
    linalg.yield %12 : f32
  } -> tensor<1024x512xf32>
  %reduced_1 = linalg.reduce ins(%8 : tensor<1024x512xf32>) outs(%6 : tensor<1024xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %11 = arith.addf %in, %init : f32
      linalg.yield %11 : f32
    }
  %9 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %reduced_1 : tensor<1024x512xf32>, tensor<1024xf32>) outs(%0 : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %11 = arith.divf %in, %in_2 : f32
    linalg.yield %11 : f32
  } -> tensor<1024x512xf32>
  %10 = linalg.matmul {__root__} ins(%9, %arg2 : tensor<1024x512xf32>, tensor<512x32xf32>) outs(%5 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  return %10 : tensor<1024x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [4, 0, 8], tile_interchange = [2, 1, 0]} :
    (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  transform.structured.tile_loop_hint %1 : !transform.any_op
}
