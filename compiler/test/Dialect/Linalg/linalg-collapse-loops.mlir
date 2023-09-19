// RUN: byteir-opt %s --linalg-collapse-loops --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @tensor_collapse(%arg0 : tensor<1x256xf16>) -> tensor<1x256xf16> {
  %empty = tensor.empty() : tensor<1x256xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x256xf16>) outs(%empty : tensor<1x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.cmpf une, %in, %in : f16
    %1 = arith.uitofp %0 : i1 to f16
    linalg.yield %1 : f16
  } -> tensor<1x256xf16>
  return %0 : tensor<1x256xf16>
}
// CHECK-LABEL: tensor_collapse
//     CHECK-SAME: (%[[ARG:.*]]: tensor<1x256xf16>)
//   CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x256xf16>
//   CHECK-DAG: %[[COLLAPSE_ARG:.*]] = tensor.collapse_shape %[[ARG]]
//   CHECK-DAG: %[[COLLAPSE_EMPTY:.*]] = tensor.collapse_shape %[[EMPTY]]
//   CHECK: linalg.generic
//     CHECK-SAME: ins(%[[COLLAPSE_ARG]] : tensor<256xf16>)
//     CHECK-SAME: outs(%[[COLLAPSE_EMPTY]] : tensor<256xf16>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @memref_collapse(%arg0: memref<1x256xf16>) -> memref<1x256xf16> {
  %alloc = memref.alloc() : memref<1x256xf16>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x256xf16>) outs(%alloc : memref<1x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.cmpf une, %in, %in : f16
    %1 = arith.uitofp %0 : i1 to f16
    linalg.yield %1 : f16
  }
  return %alloc : memref<1x256xf16>
}
// CHECK-LABEL: memref_collapse
//     CHECK-SAME: (%[[ARG:.*]]: memref<1x256xf16>)
//   CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x256xf16>
//   CHECK-DAG: %[[COLLAPSE_ARG:.*]] = memref.collapse_shape %[[ARG]]
//   CHECK-DAG: %[[COLLAPSE_ALLOC:.*]] = memref.collapse_shape %[[ALLOC]]
//   CHECK: linalg.generic
//     CHECK-SAME: ins(%[[COLLAPSE_ARG]] : memref<256xf16>)
//     CHECK-SAME: outs(%[[COLLAPSE_ALLOC]] : memref<256xf16>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @memref_non_identity(%arg0: memref<1x328xf16>) -> (memref<1x128xf16>, memref<1x128xf16>) {
  %subview = memref.subview %arg0[0, 0] [1, 128] [1, 1] : memref<1x328xf16> to memref<1x128xf16, strided<[328, 1]>>
  %alloc = memref.alloc() : memref<1x128xf16>
  memref.copy %subview, %alloc : memref<1x128xf16, strided<[328, 1]>> to memref<1x128xf16>
  %alloc_0 = memref.alloc() : memref<1x128xf16>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<1x128xf16, strided<[328, 1]>>) outs(%alloc_0 : memref<1x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.cmpf une, %in, %in : f16
    %1 = arith.uitofp %0 : i1 to f16
    linalg.yield %1 : f16
  }
  return %alloc, %alloc_0 : memref<1x128xf16>, memref<1x128xf16>
}
// CHECK-LABEL: memref_non_identity
//   CHECK-NO: memref.collapse_shape
