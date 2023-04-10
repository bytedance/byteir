// RUN: byteir-opt %s -hlo-move-up | FileCheck %s
// RUN: byteir-opt %s -hlo-move-up="multi-input" | FileCheck %s --check-prefix MULTIINPUT

func.func @transpose_move_up_unary(%arg0: tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.abs"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x20x32xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_up_unary
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func.func @reshape_move_up_convert(%arg0: tensor<1x32xi32>) -> tensor<1x1x32xf32> {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x32xi32>) -> tensor<1x32xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<1x32xf32>) -> tensor<1x1x32xf32>
    return %1 : tensor<1x1x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_up_convert
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.convert
// CHECK-NEXT: return

func.func @transpose_unary_side_user(%arg0: tensor<31x20x32xf32>) -> (tensor<20x31x32xf32>, tensor<31x20x32xf32>) {
    %0 = "mhlo.abs"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x20x32xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.abs"(%0) : (tensor<31x20x32xf32>) -> tensor<31x20x32xf32>
    return %1, %2 : tensor<20x31x32xf32>, tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @transpose_unary_side_user
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func.func @transpose_move_up_unary_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %2 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_up_unary_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func.func @transpose_move_up_unary_many_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.sine"(%1) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = "mhlo.sine"(%2) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %4 = "mhlo.sine"(%3) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %5 = "mhlo.transpose"(%4) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %5 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_up_unary_many_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: return

func.func @transpose_binary_splat_const(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<31x20x32xf32>
    %1 = mhlo.add %arg0, %0 : tensor<31x20x32xf32>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %2 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_binary_splat_const
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: return

func.func @transpose_move_up_binary(%arg0 : tensor<31x20x32xf32>, %arg1 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<31x20x32xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_up_binary
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: return

// MULTIINPUT-LABEL: func.func @transpose_move_up_binary
// MULTIINPUT-NEXT: mhlo.transpose
// MULTIINPUT-NEXT: mhlo.transpose
// MULTIINPUT-NEXT: mhlo.add
// MULTIINPUT-NEXT: return

func.func @transpose_move_up_binary_cancel(%arg0 : tensor<20x31x32xf32>, %arg1 : tensor<20x31x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    %1 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    %2 = mhlo.add %0, %1 : tensor<31x20x32xf32>
    %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_up_binary_cancel
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: return

// MULTIINPUT-LABEL: func.func @transpose_move_up_binary_cancel
// MULTIINPUT-NEXT: mhlo.add
// MULTIINPUT-NEXT: return

func.func @transpose_move_up_binary_same(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = mhlo.add %arg0, %arg0 : tensor<31x20x32xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_up_binary_same
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: return

// MULTIINPUT-LABEL: func.func @transpose_move_up_binary_same
// MULTIINPUT-NEXT: mhlo.transpose
// MULTIINPUT-NEXT: mhlo.add
// MULTIINPUT-NEXT: return

func.func @reshape_move_up_unary(%arg0: tensor<31x20x32xf32>) -> tensor<31x640xf32> {
    %0 = "mhlo.abs"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x20x32xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    return %1 : tensor<31x640xf32>
}
// CHECK-LABEL: func.func @reshape_move_up_unary
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func.func @reshape_unary_side_user(%arg0: tensor<31x20x32xf32>) -> (tensor<31x640xf32>, tensor<31x20x32xf32>) {
    %0 = "mhlo.abs"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x20x32xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %2 = "mhlo.abs"(%0) : (tensor<31x20x32xf32>) -> tensor<31x20x32xf32>
    return %1, %2 : tensor<31x640xf32>, tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @reshape_unary_side_user
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func.func @reshape_move_up_unary_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = "mhlo.abs"(%0) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<31x640xf32>) -> tensor<31x20x32xf32>
    return %2 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_up_unary_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func.func @reshape_move_up_unary_many_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.sine"(%1) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = "mhlo.sine"(%2) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %4 = "mhlo.sine"(%3) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %5 = "mhlo.reshape"(%4) : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %5 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_up_unary_many_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: return

func.func @reshape_binary_splat_const(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<31x20x32xf32>
    %1 = mhlo.add %arg0, %0 : tensor<31x20x32xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %2 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @reshape_binary_splat_const
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: return

func.func @reshape_move_up_binary(%arg0 : tensor<31x20x32xf32>, %arg1 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<31x20x32xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_up_binary
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

// MULTIINPUT-LABEL: func.func @reshape_move_up_binary
// MULTIINPUT-NEXT: mhlo.reshape
// MULTIINPUT-NEXT: mhlo.reshape
// MULTIINPUT-NEXT: mhlo.add
// MULTIINPUT-NEXT: return

func.func @reshape_move_up_binary_cancel(%arg0 : tensor<20x31x32xf32>, %arg1 : tensor<20x31x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    %1 = "mhlo.reshape"(%arg1) : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    %2 = mhlo.add %0, %1 : tensor<31x20x32xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_up_binary_cancel
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

// MULTIINPUT-LABEL: func.func @reshape_move_up_binary_cancel
// MULTIINPUT-NEXT: mhlo.add
// MULTIINPUT-NEXT: return

func.func @reshape_move_up_binary_same(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = mhlo.add %arg0, %arg0 : tensor<31x20x32xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_up_binary_same
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: return

// MULTIINPUT-LABEL: func.func @reshape_move_up_binary_same
// MULTIINPUT-NEXT: mhlo.reshape
// MULTIINPUT-NEXT: mhlo.add
// MULTIINPUT-NEXT: return

