// RUN: byteir-opt %s -hlo-move-down | FileCheck %s
// RUN: byteir-opt %s -hlo-move-down="multi-user" | FileCheck %s --check-prefix MULTIUSER
// RUN: byteir-opt %s -hlo-move-down="all-multi-user" | FileCheck %s --check-prefix AllMULTIUSER

func.func @transpose_move_down_unary(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_down_unary
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: return

func.func @transpose_binary_same(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = mhlo.add %0, %0 : tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_binary_same
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: return

func.func @transpose_move_down_binary_splat_const(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<20x31x32xf32>
    %1 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %2 = mhlo.add %1, %0 : tensor<20x31x32xf32>
    return %2 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_down_binary_splat_const
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: return

func.func @transpose_move_down_unary_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %2 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_down_unary_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func.func @transpose_move_down_unary_many_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.sine"(%1) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = "mhlo.sine"(%2) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %4 = "mhlo.sine"(%3) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %5 = "mhlo.transpose"(%4) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %5 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_down_unary_many_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: return


func.func @transpose_move_down_two_unary(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.sine"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = mhlo.add %1, %2 : tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_down_two_unary
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: return

// MULTIUSER-LABEL: func.func @transpose_move_down_two_unary
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ABS}: mhlo.transpose
// MULTIUSER-DAG{SINE}: mhlo.sine
// MULTIUSER-NEXT{SINE}: mhlo.transpose
// MULTIUSER: mhlo.add
// MULTIUSER-NEXT: return

// AllMULTIUSER-LABEL: func.func @transpose_move_down_two_unary
// AllMULTIUSER-DAG{ABS}: mhlo.abs
// AllMULTIUSER-NEXT{ABS}: mhlo.transpose
// AllMULTIUSER-DAG{SINE}: mhlo.sine
// AllMULTIUSER-NEXT{SINE}: mhlo.transpose
// AllMULTIUSER: mhlo.add
// AllMULTIUSER-NEXT: return

func.func @transpose_move_down_1_unary_1_invalid(%arg0 : tensor<31x20x32xf32>, %arg1 : tensor<20x31x32xf32>)-> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = mhlo.add %0, %arg1 : tensor<20x31x32xf32>
    %3 = mhlo.multiply %1, %2 : tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @transpose_move_down_1_unary_1_invalid
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

// MULTIUSER-LABEL: func.func @transpose_move_down_1_unary_1_invalid
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ABS}: mhlo.transpose
// MULTIUSER-DAG{ADD}: mhlo.transpose
// MULTIUSER-NEXT{ADD}: mhlo.add
// MULTIUSER: mhlo.multiply
// MULTIUSER-NEXT: return

// AllMULTIUSER-LABEL: func.func @transpose_move_down_1_unary_1_invalid
// AllMULTIUSER-NEXT: mhlo.transpose
// AllMULTIUSER-NEXT: mhlo.abs
// AllMULTIUSER-NEXT: mhlo.add
// AllMULTIUSER-NEXT: mhlo.multiply
// AllMULTIUSER-NEXT: return

func.func @reshape_move_down_unary(%arg0 : tensor<31x20x32xf32>) -> tensor<31x640xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = "mhlo.abs"(%0) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    return %1 : tensor<31x640xf32>
}
// CHECK-LABEL: func.func @reshape_move_down_unary
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @reshape_move_down_convert(%arg0: tensor<1x32xi32>) -> tensor<1x32x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x32xi32>) -> tensor<1x1x32xi32>
    %1 = "mhlo.convert"(%0) : (tensor<1x1x32xi32>) -> tensor<1x1x32xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x32xf32>) -> tensor<1x32x32xf32>
    return %2 : tensor<1x32x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_down_convert
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
// CHECK-NEXT: %[[V0:.*]] = mhlo.convert %[[ARG]] : (tensor<1x32xi32>) -> tensor<1x32xf32>
// CHECK-NEXT: %[[V1:.*]] = "mhlo.broadcast_in_dim"(%[[V0]])
// CHECK-SAME: broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>
// CHECK-SAME: (tensor<1x32xf32>) -> tensor<1x32x32xf32>
// CHECK-NEXT: return

func.func @reshape_binary_same(%arg0 : tensor<31x20x32xf32>) -> tensor<31x640xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = mhlo.add %0, %0 : tensor<31x640xf32>
    return %1 : tensor<31x640xf32>
}
// CHECK-LABEL: func.func @reshape_binary_same
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @reshape_move_down_binary_splat_const(%arg0 : tensor<31x20x32xf32>) -> tensor<31x640xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<31x640xf32>
    %1 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %2 = mhlo.add %1, %0 : tensor<31x640xf32>
    return %2 : tensor<31x640xf32>
}
// CHECK-LABEL: func.func @reshape_move_down_binary_splat_const
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @reshape_move_down_binary_with_arg(%arg0 : tensor<31x20x32xf32>, %arg1 : tensor<31x640xf32>) -> tensor<31x640xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = mhlo.add %0, %arg1 : tensor<31x640xf32>
    return %1 : tensor<31x640xf32>
}
// CHECK-LABEL: func.func @reshape_move_down_binary_with_arg
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @reshape_move_down_unary_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = "mhlo.abs"(%0) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<31x640xf32>) -> tensor<31x20x32xf32>
    return %2 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_down_unary_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func.func @reshape_move_down_unary_many_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = "mhlo.abs"(%0) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %2 = "mhlo.sine"(%1) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %3 = "mhlo.sine"(%2) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %4 = "mhlo.sine"(%3) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %5 = "mhlo.reshape"(%4) : (tensor<31x640xf32>) -> tensor<31x20x32xf32>
    return %5 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_down_unary_many_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: return

func.func @reshape_move_down_two_unary(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.sine"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = mhlo.add %1, %2 : tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_down_two_unary
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: return

// MULTIUSER-LABEL: func.func @reshape_move_down_two_unary
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ABS}: mhlo.reshape
// MULTIUSER-DAG{SINE}: mhlo.sine
// MULTIUSER-NEXT{SINE}: mhlo.reshape
// MULTIUSER: mhlo.add
// MULTIUSER-NEXT: return

// AllMULTIUSER-LABEL: func.func @reshape_move_down_two_unary
// AllMULTIUSER-DAG{ABS}: mhlo.abs
// AllMULTIUSER-NEXT{ABS}: mhlo.reshape
// AllMULTIUSER-DAG{SINE}: mhlo.sine
// AllMULTIUSER-NEXT{SINE}: mhlo.reshape
// AllMULTIUSER: mhlo.add
// AllMULTIUSER-NEXT: return

func.func @reshape_move_down_1_unary_1_invalid(%arg0 : tensor<31x20x32xf32>, %arg1 : tensor<20x31x32xf32>)-> tensor<20x31x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.abs"(%arg1) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = mhlo.add %0, %2 : tensor<20x31x32xf32>
    %4 = mhlo.multiply %1, %3 : tensor<20x31x32xf32>
    return %4 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func.func @reshape_move_down_1_unary_1_invalid
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

// MULTIUSER-LABEL: func.func @reshape_move_down_1_unary_1_invalid
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ABS}: mhlo.reshape
// MULTIUSER-DAG{ADD}: mhlo.reshape
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ADD}: mhlo.add
// MULTIUSER: mhlo.multiply
// MULTIUSER-NEXT: return

// AllMULTIUSER-LABEL: func.func @reshape_move_down_1_unary_1_invalid
// AllMULTIUSER-NEXT: mhlo.reshape
// AllMULTIUSER-NEXT: mhlo.abs
// AllMULTIUSER-NEXT: mhlo.abs
// AllMULTIUSER-NEXT: mhlo.add
// AllMULTIUSER-NEXT: mhlo.multiply
// AllMULTIUSER-NEXT: return

func.func @broadcast_move_down_unary(%arg0 : tensor<32xf32>) -> tensor<4x32xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    return %1 : tensor<4x32xf32>
}
// CHECK-LABEL: func.func @broadcast_move_down_unary
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_move_down_convert(%arg0 : tensor<32xi32>) -> tensor<4x32xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xi32>) -> tensor<4x32xi32>
    %1 = "mhlo.convert"(%0) : (tensor<4x32xi32>) -> tensor<4x32xf32>
    %2 = "mhlo.abs"(%1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    return %2 : tensor<4x32xf32>
}
// CHECK-LABEL: func.func @broadcast_move_down_convert
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]
// CHECK-NEXT: %[[V0:.*]] = mhlo.convert %[[ARG]] : (tensor<32xi32>) -> tensor<32xf32>
// CHECK-NEXT: %[[V1:.*]] = mhlo.abs %[[V0]] : tensor<32xf32>
// CHECK-NEXT: %[[V2:.*]] = "mhlo.broadcast_in_dim"(%[[V1]])
// CHECK-SAME: (tensor<32xf32>) -> tensor<4x32xf32>
// CHECK-NEXT: return

func.func @broadcast_binary_same(%arg0 : tensor<32xf32>) -> tensor<4x32xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %1 = mhlo.add %0, %0 : tensor<4x32xf32>
    return %1 : tensor<4x32xf32>
}
// CHECK-LABEL: func.func @broadcast_binary_same
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_move_down_binary_with_dtype_alter(%arg0 : tensor<32xf32>, %arg1 : tensor<32xf32>) -> tensor<4x32xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xi1>
    return %2 : tensor<4x32xi1>
}
// CHECK-LABEL: func.func @broadcast_move_down_binary_with_dtype_alter
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: tensor<32xf32>, %[[ARG1:[a-zA-Z0-9]+]]: tensor<32xf32>)
// CHECK-NEXT: %[[V0:.*]] = mhlo.compare  EQ, %[[ARG0]], %[[ARG1]]
// CHECK-SAME: (tensor<32xf32>, tensor<32xf32>) -> tensor<32xi1>
// CHECK-NEXT: %[[V1:.*]] = "mhlo.broadcast_in_dim"(%[[V0]])
// CHECK-SAME: (tensor<32xi1>) -> tensor<4x32xi1>
// CHECK-NEXT: return

 func.func @broadcast_move_down_binary_splat_const(%arg0 : tensor<32xf32>) -> tensor<4x32xf32> {
     %0 = mhlo.constant dense<1.000000e+00> : tensor<4x32xf32>
     %1 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
     %2 = mhlo.add %1, %0 : tensor<4x32xf32>
     return %2 : tensor<4x32xf32>
 }
 // CHECK-LABEL: func.func @broadcast_move_down_binary_splat_const
 // CHECK-NEXT: mhlo.constant
 // CHECK-NEXT: mhlo.add
 // CHECK-NEXT: mhlo.broadcast
 // CHECK-NEXT: return

func.func @broadcast_move_down_unary_and_merge(%arg0 : tensor<32xf32>) -> tensor<8x4x32xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x32xf32>) -> tensor<8x4x32xf32>
    return %2 : tensor<8x4x32xf32>
}
// CHECK-LABEL: func.func @broadcast_move_down_unary_and_merge
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_move_down_unary_many_and_merge(%arg0 : tensor<32xf32>) -> tensor<8x4x32xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    %2 = "mhlo.sine"(%1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    %3 = "mhlo.sine"(%2) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    %4 = "mhlo.sine"(%3) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x32xf32>) -> tensor<8x4x32xf32>
    return %5 : tensor<8x4x32xf32>
}
// CHECK-LABEL: func.func @broadcast_move_down_unary_many_and_merge
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_parallel_to_binary(%arg0 : tensor<32xf32>, %arg1 : tensor<32xf32>) -> tensor<4x32xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %2 = mhlo.add %0, %1 : tensor<4x32xf32>
    return %2 : tensor<4x32xf32>
}

// CHECK-LABEL: func.func @broadcast_parallel_to_binary
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_not_move_down_binary(%arg0: tensor<2x16x1x768xf32>, %arg1: tensor<2x1x256x768xf32>) -> tensor<2x16x256x768xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x16x1x768xf32>) -> tensor<2x16x256x768xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x1x256x768xf32>) -> tensor<2x16x256x768xf32>
    %2 = "mhlo.multiply"(%0, %1) : (tensor<2x16x256x768xf32>, tensor<2x16x256x768xf32>) -> tensor<2x16x256x768xf32>
    return %2 : tensor<2x16x256x768xf32>
}
// CHECK-LABEL: func.func @broadcast_not_move_down_binary
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @broadcast_parallel_to_binary_with_unary(%arg0 : tensor<32xf32>, %arg1 : tensor<32xf32>) -> tensor<4x32xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    %2 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<4x32xf32>
    %3 = "mhlo.abs"(%2) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    %4 = mhlo.add %1, %3 : tensor<4x32xf32>
    return %4 : tensor<4x32xf32>
}
// CHECK-LABEL: func.func @broadcast_parallel_to_binary_with_unary
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_reshape(%arg0 : tensor<1x64xf16>) -> tensor<1024x64xf16> {
    %0 = "mhlo.broadcast_in_dim" (%arg0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x64xf16>) -> tensor<1x1024x64xf16>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1024x64xf16>) -> tensor<1024x64xf16>
    return %1 : tensor<1024x64xf16>
}

// CHECK-LABEL: func.func @broadcast_reshape
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_reshape_dot(%arg0 : tensor<1x64xf16>, %arg1 : tensor<64x176xf16>) -> tensor<1024x176xf16> {
    %0 = "mhlo.broadcast_in_dim" (%arg0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x64xf16>) -> tensor<1x1024x64xf16>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1024x64xf16>) -> tensor<1024x64xf16>
    %2 = "mhlo.dot"(%1, %arg1) : (tensor<1024x64xf16>, tensor<64x176xf16>) -> tensor<1024x176xf16>
    return %2 : tensor<1024x176xf16>
}

// CHECK-LABEL: func.func @broadcast_reshape_dot
// CHECK-NEXT: mhlo.dot
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_reshape_dot_with_concat(%arg0 : tensor<1x64xf16>, %arg1: tensor<1024x176xf16>, %arg2 : tensor<64x176xf16>) -> (tensor<1024x240xf16>, tensor<1024x176xf16>) {
    %0 = "mhlo.broadcast_in_dim" (%arg0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x64xf16>) -> tensor<1x1024x64xf16>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1024x64xf16>) -> tensor<1024x64xf16>
    %2 = "mhlo.concatenate"(%arg1, %1) {dimension = 1 : i64} : (tensor<1024x176xf16>, tensor<1024x64xf16>) -> tensor<1024x240xf16>
    %3 = "mhlo.dot"(%1, %arg2) : (tensor<1024x64xf16>, tensor<64x176xf16>) -> tensor<1024x176xf16>
    return %2, %3 : tensor<1024x240xf16>, tensor<1024x176xf16> 
}

// CHECK-LABEL: func.func @broadcast_reshape_dot_with_concat
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.concatenate
// CHECK-NEXT: mhlo.dot
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @broadcast_reshape_dot_with_concat_and_add(%arg0 : tensor<1x64xf16>, %arg1: tensor<1024x176xf16>, %arg2 : tensor<64x176xf16>, %arg3 : tensor<176xf16>) -> (tensor<1024x240xf16>, tensor<1024x176xf16>) {
    %0 = "mhlo.broadcast_in_dim" (%arg0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x64xf16>) -> tensor<1x1024x64xf16>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1024x64xf16>) -> tensor<1024x64xf16>
    %2 = "mhlo.concatenate"(%arg1, %1) {dimension = 1 : i64} : (tensor<1024x176xf16>, tensor<1024x64xf16>) -> tensor<1024x240xf16>
    %3 = "mhlo.dot"(%1, %arg2) : (tensor<1024x64xf16>, tensor<64x176xf16>) -> tensor<1024x176xf16>
    %4 = "mhlo.broadcast_in_dim" (%arg3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<176xf16>) -> tensor<1024x176xf16>
    %5 = mhlo.add %3, %4 : tensor<1024x176xf16>
    return %2, %5 : tensor<1024x240xf16>, tensor<1024x176xf16> 
}

// CHECK-LABEL: func.func @broadcast_reshape_dot_with_concat
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.concatenate
// CHECK-NEXT: mhlo.dot
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

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
// CHECK-NEXT: return %[[V2]], %[[V1]]

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
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: mhlo.slice
// CHECK-NEXT: return
