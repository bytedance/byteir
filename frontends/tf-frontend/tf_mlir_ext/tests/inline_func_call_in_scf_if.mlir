// RUN: tf-ext-opt -inline-func-call-in-scf-if %s | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func private @SwitchMergeToIf_True_0(%arg0: tensor<1x100x64xf16>) -> tensor<1x1x64xf16> {
    %cst = "tf.Const"() <{value = dense<1> : tensor<3xi32>}> : () -> tensor<3xi32>
    %cst_0 = "tf.Const"() <{value = dense<[0, 1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %cst_1 = "tf.Const"() <{value = dense<0> : tensor<3xi32>}> : () -> tensor<3xi32>
    %0 = "tf.StridedSlice"(%arg0, %cst_1, %cst_0, %cst) <{begin_mask = 5 : i64, ellipsis_mask = 0 : i64, end_mask = 5 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> {device = ""} : (tensor<1x100x64xf16>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x1x64xf16>
    return %0 : tensor<1x1x64xf16>
  }
  func.func private @SwitchMergeToIf_False_0(%arg0: tensor<1x100x64xf16>) -> tensor<1x100x64xf16> {
    return %arg0 : tensor<1x100x64xf16>
  }
  func.func @inline_func_call_in_scf_if(%arg0: i1, %arg1: tensor<1x100x64xf16>) -> tensor<1x?x64xf16> {
    %2 = scf.if %arg0 -> (tensor<1x?x64xf16>) {
      %0 = func.call @SwitchMergeToIf_True_0(%arg1) : (tensor<1x100x64xf16>) -> tensor<1x1x64xf16>
      %1 = "tf.Cast"(%0) <{Truncate = false}> : (tensor<1x1x64xf16>) -> tensor<1x?x64xf16>
      scf.yield %1 : tensor<1x?x64xf16>
    } else {
      %0 = func.call @SwitchMergeToIf_False_0(%arg1) : (tensor<1x100x64xf16>) -> tensor<1x100x64xf16>
      %1 = "tf.Cast"(%0) <{Truncate = false}> : (tensor<1x100x64xf16>) -> tensor<1x?x64xf16>
      scf.yield %1 : tensor<1x?x64xf16>
    }
    return %2 : tensor<1x?x64xf16> 
  }
}

// CHECK:  func.func @inline_func_call_in_scf_if(%arg0: i1, %arg1: tensor<1x100x64xf16>) -> tensor<1x?x64xf16> {
// CHECK-NEXT:  %0 = scf.if %arg0 -> (tensor<1x?x64xf16>) {
// CHECK-NEXT:    %cst = "tf.Const"() <{value = dense<1> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT:    %cst_0 = "tf.Const"() <{value = dense<[0, 1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT:    %cst_1 = "tf.Const"() <{value = dense<0> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT:    %1 = "tf.StridedSlice"(%arg1, %cst_1, %cst_0, %cst) <{begin_mask = 5 : i64, ellipsis_mask = 0 : i64, end_mask = 5 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> {device = ""} : (tensor<1x100x64xf16>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x1x64xf16>
// CHECK-NEXT:    %2 = "tf.Cast"(%1) <{Truncate = false}> : (tensor<1x1x64xf16>) -> tensor<1x?x64xf16>
// CHECK-NEXT:    scf.yield %2 : tensor<1x?x64xf16>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %1 = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<1x100x64xf16>) -> tensor<1x?x64xf16>
// CHECK-NEXT:    scf.yield %1 : tensor<1x?x64xf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %0 : tensor<1x?x64xf16>
