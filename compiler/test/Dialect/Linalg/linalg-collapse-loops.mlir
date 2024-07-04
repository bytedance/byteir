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

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
module {
  func.func @tensor_collapse_with_dynamic_shape(%arg0: tensor<?x32x64x?x16x?xf16>, %arg1: tensor<32x64x?xf16>, %arg2: tensor<?x16x?xf16>) -> tensor<?x32x64x?x16x?xf16> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x32x64x?x16x?xf16>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<?x32x64x?x16x?xf16>
    %dim_1 = tensor.dim %arg0, %c5 : tensor<?x32x64x?x16x?xf16>
    %0 = tensor.empty(%dim, %dim_0, %dim_1) : tensor<?x32x64x?x16x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x32x64x?x16x?xf16>, tensor<32x64x?xf16>, tensor<?x16x?xf16>) outs(%0 : tensor<?x32x64x?x16x?xf16>) {
    ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
      %2 = arith.addf %in_2, %in_3 : f16
      %3 = arith.cmpf une, %in, %in : f16
      %4 = arith.uitofp %3 : i1 to f16
      %5 = arith.addf %4, %2 : f16
      linalg.yield %5 : f16
    } -> tensor<?x32x64x?x16x?xf16>
    return %1 : tensor<?x32x64x?x16x?xf16>
  }
}

// CHECK-LABEL: func.func @tensor_collapse_with_dynamic_shape
// CHECK-DAG: %[[EMPTY:.*]] = tensor.empty
// CHECK-DAG: %[[COLLAPSED0:.*]] = tensor.collapse_shape %arg0 {{\[}}[0], [1, 2], [3], [4, 5]] : tensor<?x32x64x?x16x?xf16> into tensor<?x2048x?x?xf16>
// CHECK-DAG: %[[COLLAPSED1:.*]] = tensor.collapse_shape %arg1 {{\[}}[0, 1], [2]] : tensor<32x64x?xf16> into tensor<2048x?xf16>
// CHECK-DAG: %[[COLLAPSED2:.*]] = tensor.collapse_shape %arg2 {{\[}}[0], [1, 2]] : tensor<?x16x?xf16> into tensor<?x?xf16>
// CHECK-DAG: %[[COLLAPSED3:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[}}[0], [1, 2], [3], [4, 5]] : tensor<?x32x64x?x16x?xf16> into tensor<?x2048x?x?xf16>
// CHECK-DAG: %expanded = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2], [3], [4, 5]]
// CHECK-DAG: return %expanded : tensor<?x32x64x?x16x?xf16>
