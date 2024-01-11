// RUN: tf-ext-opt %s -process-dynamic-stitch-as-static -canonicalize -tf-shape-inference -o %t
// RUN: FileCheck %s < %t
// RUN: python3 numerical_test.py %s %t --config process_dynamic_stitch_as_static

func.func @dynamic_stitch_with_multiple_data(%arg0: tensor<512x368xf16>, %arg1: tensor<512x368xf16>, %arg2: tensor<512x368xf16>, %arg3: tensor<512xi32>) -> tensor<?x736xf16> attributes {tf.entry_func.function = {control_outputs = "", inputs = "queries,keys,mask,par", outputs = "Output"}} {
  %cst = "tf.Const"() {value = dense<[512, 368]> : tensor<2xi32>} : () -> tensor<2xi32>
  %cst_1 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_3 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_4 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %cst_5 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_6 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
	%0:2 = "tf.DynamicPartition"(%arg0, %arg3) {device = ""} : (tensor<512x368xf16>, tensor<512xi32>) -> (tensor<?x368xf16>, tensor<?x368xf16>)
	%1:2 = "tf.DynamicPartition"(%arg1, %arg3) {device = ""} : (tensor<512x368xf16>, tensor<512xi32>) -> (tensor<?x368xf16>, tensor<?x368xf16>)
	%2:2 = "tf.DynamicPartition"(%arg2, %arg3) {device = ""} : (tensor<512x368xf16>, tensor<512xi32>) -> (tensor<?x368xf16>, tensor<?x368xf16>)
	%3 = "tf.ConcatV2"(%0#0, %1#0, %cst_4) {device = ""} : (tensor<?x368xf16>, tensor<?x368xf16>, tensor<i32>) -> tensor<?x736xf16>
	%4 = "tf.ConcatV2"(%1#1, %2#1, %cst_4) {device = ""} : (tensor<?x368xf16>, tensor<?x368xf16>, tensor<i32>) -> tensor<?x736xf16>
  %5 = "tf.StridedSlice"(%cst, %cst_1, %cst_2, %cst_3) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %6 = "tf.Range"(%cst_5, %5, %cst_6) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<512xi32>
	%7:2 = "tf.DynamicPartition"(%6, %arg3) {device = ""} : (tensor<512xi32>, tensor<512xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %8 = "tf.DynamicStitch"(%7#0, %7#1, %3, %4) {device = ""} : (tensor<?xi32>, tensor<?xi32>, tensor<?x736xf16>, tensor<?x736xf16>) -> tensor<?x736xf16>
	return %8 : tensor<?x736xf16>
}

// CHECK-LABEL:  func.func @dynamic_stitch_with_multiple_data
// CHECK-DAG:    %[[CST:.+]] = "tf.Const"() <{value = dense<1> : tensor<512xi32>}> : () -> tensor<512xi32>
// CHECK-DAG:    %[[CST0:.+]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT:    %0 = "tf.ConcatV2"(%arg0, %arg1, %[[CST0]]) {device = ""} : (tensor<512x368xf16>, tensor<512x368xf16>, tensor<i32>) -> tensor<?x736xf16>
// CHECK-NEXT:    %1 = "tf.ConcatV2"(%arg1, %arg2, %[[CST0]]) {device = ""} : (tensor<512x368xf16>, tensor<512x368xf16>, tensor<i32>) -> tensor<?x736xf16>
// CHECK-NEXT:    %2 = "tf.Equal"(%arg3, %cst) <{incompatible_shape_error = true}> : (tensor<512xi32>, tensor<512xi32>) -> tensor<512xi1>
// CHECK-NEXT:    %3 = "tf.Select"(%2, %1, %0) : (tensor<512xi1>, tensor<?x736xf16>, tensor<?x736xf16>) -> tensor<?x736xf16>
// CHECK-NEXT:    return %3 : tensor<?x736xf16>

func.func @dynamic_stitch_with_zero_fill(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4x10xi1>, %arg3: tensor<4xi32>) -> tensor<?x10xf32> attributes {tf.entry_func.function = {control_outputs = "", inputs = "queries,keys,mask,par", outputs = "Output"}} {
  %cst = "tf.Const"() {value = dense<[4, 5]> : tensor<2xi32>} : () -> tensor<2xi32>
  %cst_0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %0:2 = "tf.DynamicPartition"(%arg1, %arg3) {T = f32, device = "", num_partitions = 2 : i64} : (tensor<4x5xf32>, tensor<4xi32>) -> (tensor<?x5xf32>, tensor<?x5xf32>)
  %1:2 = "tf.DynamicPartition"(%arg2, %arg3) {T = i1, device = "", num_partitions = 2 : i64} : (tensor<4x10xi1>, tensor<4xi32>) -> (tensor<?x10xi1>, tensor<?x10xi1>)
  %2:2 = "tf.DynamicPartition"(%arg0, %arg3) {T = f32, device = "", num_partitions = 2 : i64} : (tensor<4x5xf32>, tensor<4xi32>) -> (tensor<?x5xf32>, tensor<?x5xf32>)
  %3 = "tf.Shape"(%2#1) {device = ""} : (tensor<?x5xf32>) -> tensor<2xi32>
  %4 = "tf.ConcatV2"(%2#1, %0#1, %cst_0) {device = ""} : (tensor<?x5xf32>, tensor<?x5xf32>, tensor<i32>) -> tensor<?x10xf32>
  %5 = "tf.ZerosLike"(%4) {device = ""} : (tensor<?x10xf32>) -> tensor<?x10xf32>
  %6 = "tf.SelectV2"(%1#1, %4, %5) {device = ""} : (tensor<?x10xi1>, tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
  %7 = "tf.Shape"(%6) {device = ""} : (tensor<?x10xf32>) -> tensor<2xi32>
  %cst_1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_3 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_4 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_5 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %8 = "tf.StridedSlice"(%cst, %cst_3, %cst_4, %cst_5) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %9 = "tf.Range"(%cst_2, %8, %cst_1) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4xi32>
  %10:2 = "tf.DynamicPartition"(%9, %arg3) {T = i32, device = "", num_partitions = 2 : i64} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %cst_6 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_7 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_8 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %11 = "tf.StridedSlice"(%3, %cst_6, %cst_7, %cst_8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %12 = "tf.Sub"(%8, %11) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %cst_9 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_10 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_11 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %13 = "tf.StridedSlice"(%7, %cst_9, %cst_10, %cst_11) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %14 = "tf.Pack"(%12, %13) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %cst_12 = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %15 = "tf.Fill"(%14, %cst_12) {device = ""} : (tensor<2xi32>, tensor<f32>) -> tensor<?x?xf32>
  %16 = "tf.DynamicStitch"(%10#0, %10#1, %15, %6) {device = ""} : (tensor<?xi32>, tensor<?xi32>, tensor<?x?xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
  return %16 : tensor<?x10xf32>
}

// CHECK-LABEL:  func.func @dynamic_stitch_with_zero_fill
// CHECK-DAG:    %[[CST:.+]] = "tf.Const"() <{value = dense<1> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:    %[[CST0:.+]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT:    %0 = "tf.ConcatV2"(%arg0, %arg1, %[[CST0]]) {device = ""} : (tensor<4x5xf32>, tensor<4x5xf32>, tensor<i32>) -> tensor<?x10xf32>
// CHECK-NEXT:    %1 = "tf.ZerosLike"(%0) {device = ""} : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:    %2 = "tf.SelectV2"(%arg2, %0, %1) {device = ""} : (tensor<4x10xi1>, tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:    %3 = "tf.ZerosLike"(%2) : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:    %4 = "tf.Equal"(%arg3, %cst) <{incompatible_shape_error = true}> : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
// CHECK-NEXT:    %5 = "tf.Select"(%4, %2, %3) : (tensor<4xi1>, tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:    return %5 : tensor<?x10xf32>
