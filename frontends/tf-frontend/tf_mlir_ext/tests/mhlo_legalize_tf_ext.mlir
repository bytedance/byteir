// RUN: tf-ext-opt -mhlo-legalize-tf-ext %s | FileCheck %s
func.func @dynamic_rank_StridedSlice(%arg0: tensor<*xf16>) -> tensor<*xf16> {
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.Const"() {value = dense<[0, 12]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<*xf16>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<*xf16>
  return %3 : tensor<*xf16> 
}
// CHECK-LABEL: dynamic_rank_StridedSlice
// CHECK: "tf.StridedSlice"

func.func @dynamic_stride_indice_StridedSlice(%arg0: tensor<6x8xf16>, %arg1: tensor<2xi32>) -> tensor<?x?xf16> {
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.Const"() {value = dense<[0, 12]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.StridedSlice"(%arg0, %0, %1, %arg1) { begin_mask = 3 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<6x8xf16>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf16>
  return %2 : tensor<?x?xf16> 
}
// CHECK-LABEL: dynamic_stride_indice_StridedSlice
// CHECK: "tf.StridedSlice"

func.func @begin_end_mask_StridedSlice(%arg0: tensor<20x30x40xf16>) -> tensor<18x12x40xf16> {
  %0 = "tf.Const"() {value = dense<[2, 0, 0]>: tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tf.Const"() {value = dense<[0, 12, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 6 : i64, ellipsis_mask = 0 : i64, end_mask = 5 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<20x30x40xf16>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<18x12x40xf16>
  return %3 : tensor<18x12x40xf16> 
}
// CHECK-LABEL: begin_end_mask_StridedSlice
// CHECK: mhlo.slice

func.func @shrink_newaxis_mask_StridedSlice(%arg0: tensor<20x30x40xf16>) -> tensor<20x40x1xf16> {
  %0 = "tf.Const"() {value = dense<[0, 8, 0, 9]>: tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Const"() {value = dense<[0, 0, 0, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 5 : i64, ellipsis_mask = 0 : i64, end_mask = 5 : i64, new_axis_mask = 8 : i64, shrink_axis_mask = 2 : i64} : (tensor<20x30x40xf16>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<20x40x1xf16>
  return %3 : tensor<20x40x1xf16> 
}
// CHECK-LABEL: shrink_newaxis_mask_StridedSlice
// CHECK: mhlo.slice
// CHECK: mhlo.reshape

func.func @ellipsis_mask_StridedSlice(%arg0: tensor<10x20x30x40x50x60xf16>) -> tensor<2x1x30x40x10x60xf16> {
  %0 = "tf.Const"() {value = dense<[2, 6, 100, 99, -20, 100]>: tensor<6xi32>} : () -> tensor<6xi32>
  %1 = "tf.Const"() {value = dense<[11, 8, 99, 100,-10, 99]> : tensor<6xi32>} : () -> tensor<6xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<6xi32>} : () -> tensor<6xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 32 : i64, ellipsis_mask = 8 : i64, end_mask = 32 : i64, new_axis_mask = 4 : i64, shrink_axis_mask = 1 : i64} : (tensor<10x20x30x40x50x60xf16>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>) -> tensor<2x1x30x40x10x60xf16>
  return %3 : tensor<2x1x30x40x10x60xf16> 
}
// CHECK-LABEL: ellipsis_mask_StridedSlice
// CHECK: mhlo.slice
// CHECK: mhlo.reshape

func.func @negative_stride_indice_StridedSlice(%arg0: tensor<10x20x30x40x50x60xf16>) -> tensor<2x1x30x40x5x60xf16> {
  %0 = "tf.Const"() {value = dense<[2, 6, 100, 99, -10, 100]>: tensor<6xi32>} : () -> tensor<6xi32>
  %1 = "tf.Const"() {value = dense<[11, 8, 99, 100,-20, 99]> : tensor<6xi32>} : () -> tensor<6xi32>
  %2 = "tf.Const"() {value = dense<[1, 1, 1, 1, -2, 1]> : tensor<6xi32>} : () -> tensor<6xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 32 : i64, ellipsis_mask = 8 : i64, end_mask = 32 : i64, new_axis_mask = 4 : i64, shrink_axis_mask = 1 : i64} : (tensor<10x20x30x40x50x60xf16>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>) -> tensor<2x1x30x40x5x60xf16>
  return %3 : tensor<2x1x30x40x5x60xf16> 
}
// CHECK-LABEL: negative_stride_indice_StridedSlice
// CHECK: mhlo.reverse
// CHECK: mhlo.slice
// CHECK: mhlo.reshape

func.func @dynamic_shape_to_sataic_shapeStridedSlice(%arg0: tensor<?x?x40xf16>) -> tensor<11x2x40xf16> {
  %0 = "tf.Const"() {value = dense<[2, 6, 100]>: tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tf.Const"() {value = dense<[11, 8, 99]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tf.Const"() {value = dense<[1, 1, -1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 5 : i64, ellipsis_mask = 0 : i64, end_mask = 4 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?x?x40xf16>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<11x2x40xf16>
  return %3 : tensor<11x2x40xf16> 
}
// CHECK-LABEL: dynamic_shape_to_sataic_shapeStridedSlice
// CHECK: mhlo.reverse
// CHECK: mhlo.slice

func.func @dynamic_begin_static_size_StridedSlice(%arg0: tensor<20x30x40xf16>, %arg1: tensor<3xi32>) -> tensor<8x30xf16> {
  %0 = "tf.Const"() {value = dense<[11, 8, 99]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tf.Const"() {value = dense<[-1, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tf.StridedSlice"(%arg0, %arg1, %0, %1) { begin_mask = 1 : i64, ellipsis_mask = 2 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 4 : i64} : (tensor<20x30x40xf16>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<8x30xf16>
  return %2 : tensor<8x30xf16> 
}
// CHECK-LABEL: dynamic_begin_static_size_StridedSlice
// CHECK: mhlo.reverse
// CHECK: mhlo.dynamic_slice
// CHECK: mhlo.reshape

func.func @static_StridedSlice(%arg0: tensor<1x26xf16>) -> tensor<1x12xf16> {
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.Const"() {value = dense<[0, 12]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 3 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1x26xf16>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x12xf16>
  return %3 : tensor<1x12xf16> 
}
// CHECK-LABEL: static_StridedSlice
// CHECK: mhlo.slice

func.func @dynamic_input_shape_StridedSlice(%arg0: tensor<?x26xf16>) -> tensor<?x12xf16> {
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.Const"() {value = dense<[0, 12]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 3 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?x26xf16>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x12xf16>
  return %3 : tensor<?x12xf16> 
}
// CHECK-LABEL: dynamic_input_shape_StridedSlice
// CHECK: mhlo.real_dynamic_slice
// CHECK: mhlo.dynamic_reshape

func.func @dynamic_begin_end_StridedSlice(%arg0: tensor<10x20x30x40xf16>, %arg1: tensor<5xi32>, %arg2: tensor<5xi32>) -> tensor<?x30x?x1xf16> {
  %0 = "tf.Const"() {value = dense<[-2, -1, -1, -3, 1]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = "tf.StridedSlice"(%arg0, %arg1, %arg2, %0) { begin_mask = 0 : i64, ellipsis_mask = 4 : i64, end_mask = 0 : i64, new_axis_mask = 16 : i64, shrink_axis_mask = 2 : i64} : (tensor<10x20x30x40xf16>, tensor<5xi32>, tensor<5xi32>, tensor<5xi32>) -> tensor<?x30x?x1xf16>
  return %1 : tensor<?x30x?x1xf16> 
}
// CHECK-LABEL: dynamic_begin_end_StridedSlice
// CHECK: mhlo.reverse
// CHECK: mhlo.real_dynamic_slice
// CHECK: mhlo.dynamic_reshape

func.func @static_batch_matmul_v2(%arg0: tensor<32x246x16xf16>, %arg1: tensor<32x16x16xf16>) -> tensor<32x246x16xf16> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false, device = ""} : (tensor<32x246x16xf16>, tensor<32x16x16xf16>) -> tensor<32x246x16xf16>
  return %0 : tensor<32x246x16xf16> 
}
// CHECK-LABEL: @static_batch_matmul_v2
// CHECK-NEXT: %0 = "mhlo.dot_general"(%arg0, %arg1) <{dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}> : (tensor<32x246x16xf16>, tensor<32x16x16xf16>) -> tensor<32x246x16xf16>
// CHECK-NEXT: return %0 : tensor<32x246x16xf16>

func.func @dynamic_batch_matmul_v2(%arg0: tensor<?x246x16xf16>, %arg1: tensor<?x16x16xf16>) -> tensor<?x246x16xf16> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false, device = ""} : (tensor<?x246x16xf16>, tensor<?x16x16xf16>) -> tensor<?x246x16xf16>
  return %0 : tensor<?x246x16xf16>
}
// CHECK-LABEL: @dynamic_batch_matmul_v2
// CHECK: mhlo.dot_general
// CHECK: return %0 : tensor<?x246x16xf16>

func.func @round_fp32(%arg0: tensor<6xf32>) -> tensor<6xf32> {
  %0 = "tf.Round"(%arg0) : (tensor<6xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}
// CHECK-LABEL: @round_fp32
// CHECK-NEXT: %0 = mhlo.round_nearest_even %arg0 : tensor<6xf32>
// CHECK-NEXT: return %0 : tensor<6xf32>

func.func @round_int32(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  %0 = "tf.Abs"(%arg0) : (tensor<6xi32>) -> tensor<6xi32>
  %1 = "tf.Round"(%0) : (tensor<6xi32>) -> tensor<6xi32>
  return %1 : tensor<6xi32>
}
// CHECK-LABEL: @round_int32
// CHECK-NEXT: %0 = "tf.Abs"(%arg0) : (tensor<6xi32>) -> tensor<6xi32>
// CHECK-NEXT: return %0 : tensor<6xi32>

func.func @tile_right_dynamic(%arg0: tensor<1x64xf16>, %arg1: tensor<2xi32>) -> tensor<1x?xf16> {
  %0 = "tf.Tile"(%arg0, %arg1) {device = ""} : (tensor<1x64xf16>, tensor<2xi32>) -> tensor<1x?xf16>
  return %0 : tensor<1x?xf16>
}
// CHECK-LABEL: func.func @tile_right_dynamic(%arg0: tensor<1x64xf16>, %arg1: tensor<2xi32>) -> tensor<1x?xf16> {
// CHECK-DGA:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c64 = arith.constant 64 : index
// CHECK-LABEL:   %extracted = tensor.extract %arg1[%c0] : tensor<2xi32>
// CHECK-LABEL:   %0 = arith.index_cast %extracted : i32 to index
// CHECK-LABEL:   %extracted_0 = tensor.extract %arg1[%c1] : tensor<2xi32>
// CHECK-LABEL:   %1 = arith.index_cast %extracted_0 : i32 to index
// CHECK-LABEL:   %2 = arith.muli %1, %c64 : index
// CHECK-LABEL:   %from_elements = tensor.from_elements %0, %c1, %1, %c64 : tensor<4xindex>
// CHECK-LABEL:   %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %from_elements) <{broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>}> : (tensor<1x64xf16>, tensor<4xindex>) -> tensor<?x1x?x64xf16>
// CHECK-LABEL:   %4 = mhlo.dynamic_reshape %3, %from_elements_1 : (tensor<?x1x?x64xf16>, tensor<2xindex>) -> tensor<1x?xf16>
// CHECK-LABEL:   return %4 : tensor<1x?xf16>

func.func @reshape_case0(%arg0: tensor<?x24xf16>) -> tensor<?x24x1xf16> {
  %cst = "tf.Const"() <{value = dense<[-1, 24, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
  %0 = "tf.Reshape"(%arg0, %cst) : (tensor<?x24xf16>, tensor<3xi64>) -> tensor<?x24x1xf16>
  return %0 : tensor<?x24x1xf16>
}
// CHECK-LABEL: func.func @reshape_case0
// CHECK-DGA:     %c1 = shape.const_size 1
// CHECK-DGA:     %c24 = shape.const_size 24
// CHECK-LABEL:   %0 = shape.shape_of %arg0 : tensor<?x24xf16> -> tensor<2xindex>
// CHECK-LABEL:   %1 = shape.num_elements %0 : tensor<2xindex> -> index
// CHECK-LABEL:   %2 = shape.index_to_size %1
// CHECK-LABEL:   %3 = shape.div %2, %c24 : !shape.size, !shape.size -> !shape.size
// CHECK-LABEL:   %4 = shape.from_extents %3, %c24, %c1 : !shape.size, !shape.size, !shape.size
// CHECK-LABEL:   %5 = shape.to_extent_tensor %4 : !shape.shape -> tensor<3xindex>
// CHECK-LABEL:   %6 = mhlo.dynamic_reshape %arg0, %5 : (tensor<?x24xf16>, tensor<3xindex>) -> tensor<?x24x1xf16>
// CHECK-LABEL:   return %6 : tensor<?x24x1xf16>
