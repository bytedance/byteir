// RUN: byteir-opt %s -hlo-move-down -slice-move-down-and-merge | FileCheck %s

func.func @slice_move_down_unary(%arg0 : tensor<1x64xf16>, %arg1: tensor<1x64xf16>) -> (tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16>) {
    %0 = "mhlo.add" (%arg0, %arg1) : (tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
    %1 = "mhlo.slice"(%0) { limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } : (tensor<1x64xf16>) -> tensor<1x1xf16>
    %2 = "mhlo.slice"(%0) { limit_indices = dense<[1, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } : (tensor<1x64xf16>) -> tensor<1x1xf16>
    %3 = "mhlo.slice"(%0) { limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } : (tensor<1x64xf16>) -> tensor<1x1xf16>
    %4 = "mhlo.negate"(%1) : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %5 = "mhlo.negate"(%2) : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %6 = "mhlo.negate"(%3) : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %7 = "mhlo.exponential"(%4) : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %8 = "mhlo.exponential"(%5) : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %9 = "mhlo.exponential"(%6) : (tensor<1x1xf16>) -> tensor<1x1xf16>
    return %7, %8, %9 : tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16>
}

// CHECK-LABEL: func.func @slice_move_down_unary
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.negate
// CHECK-NEXT: mhlo.exponential
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: return

func.func @slice_move_down_convert(%arg0 : tensor<1x64xf32>) -> (tensor<1x1xi32>, tensor<1x1xi32>) {
    %0 = "mhlo.slice"(%arg0) { limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } : (tensor<1x64xf32>) -> tensor<1x1xf32>
    %1 = "mhlo.slice"(%arg0) { limit_indices = dense<[1, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } : (tensor<1x64xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.convert"(%0) : (tensor<1x1xf32>) -> tensor<1x1xi32>
    %3 = "mhlo.convert"(%1) : (tensor<1x1xf32>) -> tensor<1x1xi32>
    return %2, %3 : tensor<1x1xi32>, tensor<1x1xi32>
}

// CHECK-LABEL: func.func @slice_move_down_convert
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
// CHECK-NEXT: %[[V0:.*]] = mhlo.convert %[[ARG]] : (tensor<1x64xf32>) -> tensor<1x64xi32>
// CHECK-NEXT: %[[V1:.*]] = "mhlo.slice"(%[[V0]])
// CHECK-SAME: (tensor<1x64xi32>) -> tensor<1x1xi32>
// CHECK-NEXT: %[[V2:.*]] = "mhlo.slice"(%[[V0]])
// CHECK-SAME: (tensor<1x64xi32>) -> tensor<1x1xi32>
// CHECK-NEXT: return %[[V1]], %[[V2]]

func.func @slice_move_down_binary(%arg0 : tensor<1x64xf16>, %arg1: tensor<1x64xf16>, %arg2: tensor<1x1xf16>) -> (tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16>) {
    %0 = "mhlo.add" (%arg0, %arg1) : (tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
    %1 = "mhlo.slice"(%0) { limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } : (tensor<1x64xf16>) -> tensor<1x1xf16>
    %2 = "mhlo.slice"(%0) { limit_indices = dense<[1, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } : (tensor<1x64xf16>) -> tensor<1x1xf16>
    %3 = "mhlo.slice"(%0) { limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } : (tensor<1x64xf16>) -> tensor<1x1xf16>
    %4 = "mhlo.add"(%1, %arg2) : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
    %5 = "mhlo.add"(%2, %arg2) : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
    %6 = "mhlo.add"(%3, %arg2) : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
    return %4, %5, %6 : tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16>
}

// CHECK-LABEL: func.func @slice_move_down_binary
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: return

func.func @slice_movedown_complex(%3254: tensor<512x10xf16>, %arg259: tensor<512xf16>) -> (tensor<512xf16>, tensor<512xf16>, tensor<512xf16>) {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<512x1xf16>
  %286 = mhlo.reshape %arg259 : (tensor<512xf16>) -> tensor<512x1xf16>
  %3255 = "mhlo.slice"(%3254) {limit_indices = dense<[512, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x10xf16>) -> tensor<512x1xf16>
  %3256 = "mhlo.slice"(%3254) {limit_indices = dense<[512, 10]> : tensor<2xi64>, start_indices = dense<[0, 9]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x10xf16>) -> tensor<512x1xf16>
  %3257 = mhlo.multiply %3255, %286 : tensor<512x1xf16>
  %3258 = mhlo.add %3257, %0 : tensor<512x1xf16>
  %3259 = mhlo.divide %0, %3258 : tensor<512x1xf16>
  %3260 = mhlo.reshape %3259 : (tensor<512x1xf16>) -> tensor<512xf16>
  %3261 = mhlo.logistic %3260 : tensor<512xf16>
  %3266 = mhlo.multiply %3256, %286 : tensor<512x1xf16>
  %3267 = mhlo.add %3266, %0 : tensor<512x1xf16>
  %3268 = mhlo.divide %0, %3267 : tensor<512x1xf16>
  %3269 = mhlo.logistic %3268 : tensor<512x1xf16>
  %3270 = mhlo.reshape %3269 : (tensor<512x1xf16>) -> tensor<512xf16>
  %3275 = "mhlo.slice"(%3254) {limit_indices = dense<[512, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x10xf16>) -> tensor<512x1xf16>
  %3277 = mhlo.multiply %3275, %286 : tensor<512x1xf16>
  %3278 = mhlo.add %3277, %0 : tensor<512x1xf16>
  %3279 = mhlo.divide %0, %3278 : tensor<512x1xf16>
  %3280 = mhlo.logistic %3279 : tensor<512x1xf16> loc(fused["Sigmoid:", "Sigmoid_32"])
  %3281 = mhlo.reshape %3280 : (tensor<512x1xf16>) -> tensor<512xf16>
  return %3261, %3270, %3281 : tensor<512xf16>, tensor<512xf16>, tensor<512xf16>
}

// CHECK-LABEL: func.func @slice_movedown_complex
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.divide
// CHECK-NEXT: mhlo.logistic
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return
