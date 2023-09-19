// RUN: byteir-opt %s -canonicalize-ext | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fold_trivial_linalg_generic(%arg0: tensor<125x32xf32>) -> tensor<125x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<125x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<125x32xf32>) -> tensor<125x32xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<125x32xf32>) outs(%1 : tensor<125x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<125x32xf32>
  return %2 : tensor<125x32xf32>
}
// CHECK-LABEL: func.func @fold_trivial_linalg_generic
// CHECK-NEXT: return

