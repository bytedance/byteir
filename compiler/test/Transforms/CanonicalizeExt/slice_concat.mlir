// RUN: byteir-opt %s -canonicalize-ext="blind-fold=true" | FileCheck %s

func.func @fold_concat_of_continuous_slices(%arg0: tensor<4x11xf32>) -> tensor<4x11xf32> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 7]> : tensor<2xi64>, start_indices = dense<[0, 5]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x2xf32>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 11]> : tensor<2xi64>, start_indices = dense<[0, 7]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x4xf32>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x5xf32>
  %3 = "mhlo.concatenate"(%2, %0, %1) {dimension = 1 : i64} : (tensor<4x5xf32>, tensor<4x2xf32>, tensor<4x4xf32>) -> tensor<4x11xf32>
  return %3 : tensor<4x11xf32>
}
// CHECK-LABEL: func.func @fold_concat_of_continuous_slices
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x11xf32>)
// CHECK-NEXT: return %[[ARG0]] : tensor<4x11xf32>

func.func @not_fold_concat_of_slice(%655: tensor<1x112x56x128xf16>) -> tensor<1x56x112x128xf16> {
  %656 = "mhlo.slice"(%655) {limit_indices = dense<[1, 59, 56, 128]> : tensor<4xi64>, start_indices = dense<[0, 3, 0, 0]> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<1x112x56x128xf16>) -> tensor<1x56x56x128xf16>
  %657 = "mhlo.concatenate"(%656, %656) {dimension = 2 : i64} : (tensor<1x56x56x128xf16>, tensor<1x56x56x128xf16>) -> tensor<1x56x112x128xf16>
  func.return %657 : tensor<1x56x112x128xf16>
}
// CHECK-LEBEL: func.func @not_fold_concat_of_slice
// CHECK:  "mhlo.slice"
// CHECK:  "mhlo.concatenate"

func.func @not_fold_concat_of_continuous_slices(%arg0: tensor<512x120x8xf16>) -> (tensor<512x30x16xf16>, tensor<512x30x16xf16>)  {
  %0 = mhlo.logistic %arg0 : (tensor<512x120x8xf16>) -> tensor<512x120x8xf16>
  %1 = "mhlo.slice"(%0) {limit_indices = dense<[512, 30, 8]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<512x120x8xf16>) -> tensor<512x30x8xf16> loc(fused["Slice:", "vec_slice_765"])
  %2 = "mhlo.slice"(%0) {limit_indices = dense<[512, 60, 8]> : tensor<3xi64>, start_indices = dense<[0, 30, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<512x120x8xf16>) -> tensor<512x30x8xf16> loc(fused["Slice:", "vec_slice_765"])
  %3 = "mhlo.slice"(%0) {limit_indices = dense<[512, 90, 8]> : tensor<3xi64>, start_indices = dense<[0, 60, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<512x120x8xf16>) -> tensor<512x30x8xf16> loc(fused["Slice:", "vec_slice_765"])
  %4 = "mhlo.slice"(%0) {limit_indices = dense<[512, 120, 8]> : tensor<3xi64>, start_indices = dense<[0, 90, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<512x120x8xf16>) -> tensor<512x30x8xf16> loc(fused["Slice:", "vec_slice_765"])
  %5 = "mhlo.concatenate"(%1, %2) {dimension = 2 : i64} : (tensor<512x30x8xf16>, tensor<512x30x8xf16>) -> tensor<512x30x16xf16> loc(fused["ConcatV2:", "concat_262"])
  %6 = "mhlo.concatenate"(%3, %4) {dimension = 2 : i64} : (tensor<512x30x8xf16>, tensor<512x30x8xf16>) -> tensor<512x30x16xf16> loc(fused["ConcatV2:", "concat_264"])
  return %5, %6 : tensor<512x30x16xf16>, tensor<512x30x16xf16>
}
// CHECK-LABEL: func.func @not_fold_concat_of_continuous_slices
// CHECK:  mhlo.logistic
// CHECK:  "mhlo.slice"
// CHECK:  "mhlo.slice"
// CHECK:  "mhlo.slice"
// CHECK:  "mhlo.slice"
// CHECK:  "mhlo.concatenate"
// CHECK:  "mhlo.concatenate"

// case1
func.func @fold_concat_with_continuous_slices_1(%arg0: tensor<128x3xf16>) -> tensor<128x3xf16> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 1]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %3 = "mhlo.concatenate"(%0, %1, %2) {dimension = 1 : i64} : (tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x1xf16>) -> tensor<128x3xf16>
  return %3 : tensor<128x3xf16>
}
// CHECK-LABEL: func.func @fold_concat_with_continuous_slices_1(%arg0: tensor<128x3xf16>) -> tensor<128x3xf16> {
// CHECK-NEXT:    return %arg0 : tensor<128x3xf16>
// CHECK-NEXT:  }

// case2
func.func @fold_concat_with_continuous_slices_2(%arg0: tensor<128x4xf16>) -> tensor<128x3xf16> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 1]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %3 = "mhlo.concatenate"(%0, %1, %2) {dimension = 1 : i64} : (tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x1xf16>) -> tensor<128x3xf16>
  return %3 : tensor<128x3xf16>
}
// CHECK-LABEL: func.func @fold_concat_with_continuous_slices_2(%arg0: tensor<128x4xf16>) -> tensor<128x3xf16> {
// CHECK-NEXT:    %[[A_0:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x3xf16>
// CHECK-NEXT:    return %[[A_0]] : tensor<128x3xf16>
// CHECK-NEXT:  }

// case3
func.func @fold_concat_with_continuous_slices_3(%arg0: tensor<128x3xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 1]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %3 = "mhlo.concatenate"(%arg1, %arg2, %0, %1, %2) {dimension = 1 : i64} : (tensor<128x2xf16>, tensor<128x2xf16>, tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x1xf16>) -> tensor<128x7xf16>
  return %3 : tensor<128x7xf16>
}
// CHECK-LABEL: func.func @fold_concat_with_continuous_slices_3(%arg0: tensor<128x3xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
// CHECK-NEXT:    %[[A_0:.*]] = "mhlo.concatenate"(%arg1, %arg2, %arg0) {dimension = 1 : i64} : (tensor<128x2xf16>, tensor<128x2xf16>, tensor<128x3xf16>) -> tensor<128x7xf16>
// CHECK-NEXT:    return %[[A_0]] : tensor<128x7xf16>
// CHECK-NEXT:  }

// case4
func.func @fold_concat_with_continuous_slices_4(%arg0: tensor<128x3xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 1]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %3 = "mhlo.concatenate"(%arg1, %0, %1, %2, %arg2) {dimension = 1 : i64} : (tensor<128x2xf16>, tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x2xf16>) -> tensor<128x7xf16>
  return %3 : tensor<128x7xf16>
}
// CHECK-LABEL: func.func @fold_concat_with_continuous_slices_4(%arg0: tensor<128x3xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
// CHECK-NEXT:    %[[A_0:.*]] = "mhlo.concatenate"(%arg1, %arg0, %arg2) {dimension = 1 : i64} : (tensor<128x2xf16>, tensor<128x3xf16>, tensor<128x2xf16>) -> tensor<128x7xf16>
// CHECK-NEXT:    return %[[A_0]] : tensor<128x7xf16>
// CHECK-NEXT:  }

// case5
func.func @fold_concat_with_continuous_slices_5(%arg0: tensor<128x3xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 1]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x3xf16>) -> tensor<128x1xf16>
  %3 = "mhlo.concatenate"(%0, %1, %2, %arg1, %arg2) {dimension = 1 : i64} : (tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x2xf16>, tensor<128x2xf16>) -> tensor<128x7xf16>
  return %3 : tensor<128x7xf16>
}
// CHECK-LABEL: func.func @fold_concat_with_continuous_slices_5(%arg0: tensor<128x3xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
// CHECK-NEXT:    %[[A_0:.*]] = "mhlo.concatenate"(%arg0, %arg1, %arg2) {dimension = 1 : i64} : (tensor<128x3xf16>, tensor<128x2xf16>, tensor<128x2xf16>) -> tensor<128x7xf16>
// CHECK-NEXT:    return %[[A_0]] : tensor<128x7xf16>
// CHECK-NEXT:  }

// case6
func.func @fold_concat_with_continuous_slices_6(%arg0: tensor<128x4xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 1]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %3 = "mhlo.concatenate"(%0, %1, %2, %arg1, %arg2) {dimension = 1 : i64} : (tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x2xf16>, tensor<128x2xf16>) -> tensor<128x7xf16>
  return %3 : tensor<128x7xf16>
}

// CHECK-LABEL: func.func @fold_concat_with_continuous_slices_6(%arg0: tensor<128x4xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
// CHECK-NEXT:    %[[A_0:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x3xf16>
// CHECK-NEXT:    %[[A_1:.*]] = "mhlo.concatenate"(%[[A_0]], %arg1, %arg2) {dimension = 1 : i64} : (tensor<128x3xf16>, tensor<128x2xf16>, tensor<128x2xf16>) -> tensor<128x7xf16>
// CHECK-NEXT:    return %[[A_1]] : tensor<128x7xf16>
// CHECK-NEXT:  }

// case7
func.func @fold_concat_with_continuous_slices_7(%arg0: tensor<128x4xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 1]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x1xf16>
  %3 = "mhlo.concatenate"(%arg1, %0, %1, %2, %arg2) {dimension = 1 : i64} : (tensor<128x2xf16>, tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x1xf16>, tensor<128x2xf16>) -> tensor<128x7xf16>
  return %3 : tensor<128x7xf16>
}

// CHECK-LABEL: func.func @fold_concat_with_continuous_slices_7(%arg0: tensor<128x4xf16>, %arg1: tensor<128x2xf16>, %arg2: tensor<128x2xf16>) -> tensor<128x7xf16> {
// CHECK-NEXT:    %[[A_0:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x4xf16>) -> tensor<128x3xf16>
// CHECK-NEXT:    %[[A_1:.*]] = "mhlo.concatenate"(%arg1, %0, %arg2) {dimension = 1 : i64} : (tensor<128x2xf16>, tensor<128x3xf16>, tensor<128x2xf16>) -> tensor<128x7xf16>
// CHECK-NEXT:    return %[[A_1]] : tensor<128x7xf16>
// CHECK-NEXT:  }

func.func @slice_reshape_concat_case0(%arg0: tensor<512x1x12xf16>, %arg1: tensor<512x48xf16>) -> (tensor<512x5x12xf16>) {
  %0 = "mhlo.slice"(%arg1) {limit_indices = dense<[512, 12]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x48xf16>) -> tensor<512x12xf16>
  %1 = "mhlo.slice"(%arg1) {limit_indices = dense<[512, 24]> : tensor<2xi64>, start_indices = dense<[0, 12]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x48xf16>) -> tensor<512x12xf16>
  %2 = "mhlo.slice"(%arg1) {limit_indices = dense<[512, 36]> : tensor<2xi64>, start_indices = dense<[0, 24]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x48xf16>) -> tensor<512x12xf16>
  %3 = "mhlo.slice"(%arg1) {limit_indices = dense<[512, 48]> : tensor<2xi64>, start_indices = dense<[0, 36]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x48xf16>) -> tensor<512x12xf16>
  %4 = mhlo.reshape %0 : (tensor<512x12xf16>) -> tensor<512x1x12xf16>
  %5 = mhlo.reshape %1 : (tensor<512x12xf16>) -> tensor<512x1x12xf16>
  %6 = mhlo.reshape %2 : (tensor<512x12xf16>) -> tensor<512x1x12xf16>
  %7 = mhlo.reshape %3 : (tensor<512x12xf16>) -> tensor<512x1x12xf16>
  %8 = "mhlo.concatenate"(%arg0, %4, %5, %6, %7) {dimension = 1 : i64} : (tensor<512x1x12xf16>, tensor<512x1x12xf16>, tensor<512x1x12xf16>, tensor<512x1x12xf16>, tensor<512x1x12xf16>) -> tensor<512x5x12xf16>
  return %8 : tensor<512x5x12xf16>
}
// CHECK-LABEL: slice_reshape_concat_case0
// CHECK-SAME: %[[ARG0:[^:[:space:]]+]]
// CHECK-SAME: %[[ARG1:[^:[:space:]]+]]
// CHECK-NOT: mhlo.slice
// CHECK-NEXT: %[[VAL_0:.*]] = mhlo.reshape %[[ARG1]]
// CHECK-NEXT: "mhlo.concatenate"(%[[ARG0]], %[[VAL_0]])

func.func @slice_reshape_concat_case1(%arg0: tensor<128x64xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x12x16xf32>{
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 16]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x64xf32>) -> tensor<128x16xf32>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 32]> : tensor<2xi64>, start_indices = dense<[0, 16]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x64xf32>) -> tensor<128x16xf32>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 48]> : tensor<2xi64>, start_indices = dense<[0, 32]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x64xf32>) -> tensor<128x16xf32>
  %3 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 64]> : tensor<2xi64>, start_indices = dense<[0, 48]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x64xf32>) -> tensor<128x16xf32>
  %4 = "mhlo.slice"(%arg1) {limit_indices = dense<[128, 16]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x16xf32>
  %5 = "mhlo.slice"(%arg1) {limit_indices = dense<[128, 32]> : tensor<2xi64>, start_indices = dense<[0, 16]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x16xf32>
  %6 = "mhlo.slice"(%arg1) {limit_indices = dense<[128, 48]> : tensor<2xi64>, start_indices = dense<[0, 32]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x16xf32>
  %7 = "mhlo.slice"(%arg1) {limit_indices = dense<[128, 64]> : tensor<2xi64>, start_indices = dense<[0, 48]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x16xf32>
  %8 = "mhlo.slice"(%arg1) {limit_indices = dense<[128, 80]> : tensor<2xi64>, start_indices = dense<[0, 64]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x16xf32>
  %9 = "mhlo.slice"(%arg1) {limit_indices = dense<[128, 96]> : tensor<2xi64>, start_indices = dense<[0, 80]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x16xf32>
  %10 = "mhlo.slice"(%arg1) {limit_indices = dense<[128, 112]> : tensor<2xi64>, start_indices = dense<[0, 96]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x16xf32>
  %11 = "mhlo.slice"(%arg1) {limit_indices = dense<128> : tensor<2xi64>, start_indices = dense<[0, 112]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x16xf32>
  %12 = mhlo.reshape %4 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %13 = mhlo.reshape %5 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %14 = mhlo.reshape %6 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %15 = mhlo.reshape %7 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %16 = mhlo.reshape %8 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %17 = mhlo.reshape %9 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %18 = mhlo.reshape %10 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %19 = mhlo.reshape %11 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %20 = mhlo.reshape %0 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %21 = mhlo.reshape %1 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %22 = mhlo.reshape %2 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %23 = mhlo.reshape %3 : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
  %24 = "mhlo.concatenate"(%12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23) {dimension = 1 : i64} : (tensor<128x1x16xf32>, tensor<128x1x16xf32>, tensor<128x1x16xf32>, tensor<128x1x16xf32>, tensor<128x1x16xf32>, tensor<128x1x16xf32>,tensor<128x1x16xf32>, tensor<128x1x16xf32>, tensor<128x1x16xf32>, tensor<128x1x16xf32>, tensor<128x1x16xf32>, tensor<128x1x16xf32>) -> tensor<128x12x16xf32>
  return %24 : tensor<128x12x16xf32>
}
// CHECK-LABEL: slice_reshape_concat_case1
// CHECK-SAME: %[[ARG0:[^:[:space:]]+]]
// CHECK-SAME: %[[ARG1:[^:[:space:]]+]]
// CHECK-DAG: %[[VAL_0:.*]] = mhlo.reshape %[[ARG0]]
// CHECK-DAG: %[[VAL_1:.*]] = mhlo.reshape %[[ARG1]]
// CHECK-NEXT: "mhlo.concatenate"(%[[VAL_1]], %[[VAL_0]])

func.func @reshape_concat_case2(%arg0: tensor<512x1x12xf16>, %arg1: tensor<512x48xf16>) -> (tensor<512x5x12xf16>) {
  %483 = "mhlo.slice"(%arg1) {limit_indices = dense<[512, 12]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x48xf16>) -> tensor<512x12xf16>
  %484 = "mhlo.slice"(%arg1) {limit_indices = dense<[512, 24]> : tensor<2xi64>, start_indices = dense<[0, 12]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x48xf16>) -> tensor<512x12xf16>
  %485 = "mhlo.slice"(%arg1) {limit_indices = dense<[512, 42]> : tensor<2xi64>, start_indices = dense<[0, 30]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x48xf16>) -> tensor<512x12xf16>
  %486 = "mhlo.slice"(%arg1) {limit_indices = dense<[512, 48]> : tensor<2xi64>, start_indices = dense<[0, 36]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x48xf16>) -> tensor<512x12xf16>
  %1465 = mhlo.reshape %483 : (tensor<512x12xf16>) -> tensor<512x1x12xf16>
  %1466 = mhlo.reshape %484 : (tensor<512x12xf16>) -> tensor<512x1x12xf16>
  %1467 = mhlo.reshape %485 : (tensor<512x12xf16>) -> tensor<512x1x12xf16>
  %1468 = mhlo.reshape %486 : (tensor<512x12xf16>) -> tensor<512x1x12xf16>
  %1469 = "mhlo.concatenate"(%arg0, %1465, %1466, %1467, %1468) {dimension = 1 : i64} : (tensor<512x1x12xf16>, tensor<512x1x12xf16>, tensor<512x1x12xf16>, tensor<512x1x12xf16>, tensor<512x1x12xf16>) -> tensor<512x5x12xf16>
  return %1469 : tensor<512x5x12xf16>
}
// CHECK-LABEL: reshape_concat_case2
// CHECK: mhlo.slice
// CHECK: mhlo.reshape
// CHECK: mhlo.concatenate
