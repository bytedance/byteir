// RUN: byteir-opt %s -canonicalize-ext --split-input-file | FileCheck %s

func.func @coalecsed_for_extract_from_shape_cast(%arg0 : vector<4x8xf32>) -> vector<4xf32> {
  %0 = vector.shape_cast %arg0 : vector<4x8xf32> to vector<32xf32>
  %1 = vector.shape_cast %0 : vector<32xf32> to vector<1x8x1x4xf32>
  %2 = vector.extract %1[0, 4, 0] : vector<1x8x1x4xf32>
  return %2 : vector<4xf32>
}
// CHECK-LABEL: func.func @coalecsed_for_extract_from_shape_cast
//   CHECK-SAME: (%[[ARG:.*]]: vector<4x8xf32>)
//   CHECK-NEXT: %[[T0:.*]] = vector.shape_cast %[[ARG]] : vector<4x8xf32> to vector<32xf32>
//   CHECK-NEXT: %[[T1:.*]] = vector.shape_cast %[[T0]] : vector<32xf32> to vector<8x4xf32>
//   CHECK-NEXT: %[[T2:.*]] = vector.extract %[[T1]][4] : vector<8x4xf32>
//   CHECK-NEXT: return %[[T2]]
