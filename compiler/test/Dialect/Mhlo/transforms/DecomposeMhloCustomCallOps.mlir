// RUN: byteir-opt %s --decompose-mhlo-custom-call-ops --canonicalize | FileCheck %s

func.func @byteir.addn(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.custom_call @byteir.addn(%arg0, %arg1, %arg2) {byteir_attrs = {}} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @byteir.addn
// CHECK-NOT:  byteir.addn
// CHECK:      mhlo.add
// CHECK:      mhlo.add
// CHECK:      return

func.func @byteir.arg_max$return_1(%arg0: tensor<3x4xf32>) -> tensor<3xi64> {
  %0 = mhlo.custom_call @byteir.arg_max(%arg0) {byteir_attrs = {axis = 1 : i64, keep_dims = false, select_last_index = false}} : (tensor<3x4xf32>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}
// CHECK-LABEL: func.func @byteir.arg_max$return_1
// CHECK-NOT:  byteir.arg_max
// CHECK-DAG:  mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:  mhlo.constant dense<0> : tensor<i64>
// CHECK:      mhlo.iota
// CHECK:      mhlo.broadcast_in_dim
// CHECK:      mhlo.reduce
// CHECK:        mhlo.compare GE
// CHECK:        mhlo.select
// CHECK:        mhlo.compare
// CHECK:        mhlo.minimum
// CHECK:        mhlo.select
// CHECK:        mhlo.select
// CHECK:        mhlo.return
// CHECK:      return

func.func @byteir.arg_max$return_2(%arg0: tensor<3x4xf32>) -> (tensor<3xf32>, tensor<3xi64>) {
  %0:2 = mhlo.custom_call @byteir.arg_max(%arg0) {byteir_attrs = {axis = 1 : i64, keep_dims = false, select_last_index = false}} : (tensor<3x4xf32>) -> (tensor<3xf32>, tensor<3xi64>)
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xi64>
}
// CHECK-LABEL: func.func @byteir.arg_max$return_2
// CHECK-NOT:  byteir.arg_max
// CHECK-DAG:  mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:  mhlo.constant dense<0> : tensor<i64>
// CHECK:      mhlo.iota
// CHECK:      mhlo.broadcast_in_dim
// CHECK:      mhlo.reduce
// CHECK:        mhlo.compare GE
// CHECK:        mhlo.select
// CHECK:        mhlo.compare
// CHECK:        mhlo.minimum
// CHECK:        mhlo.select
// CHECK:        mhlo.select
// CHECK:        mhlo.return
// CHECK:      return

func.func @byteir.arg_min$return_1(%arg0: tensor<3x4xf32>) -> tensor<3xi64> {
  %0 = mhlo.custom_call @byteir.arg_min(%arg0) {byteir_attrs = {axis = 1 : i64, keep_dims = false, select_last_index = false}} : (tensor<3x4xf32>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}
// CHECK-LABEL: func.func @byteir.arg_min$return_1
// CHECK-NOT:  byteir.arg_min
// CHECK-DAG:  mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK-DAG:  mhlo.constant dense<0> : tensor<i64>
// CHECK:      mhlo.iota
// CHECK:      mhlo.broadcast_in_dim
// CHECK:      mhlo.reduce
// CHECK:        mhlo.compare LE
// CHECK:        mhlo.select
// CHECK:        mhlo.compare
// CHECK:        mhlo.minimum
// CHECK:        mhlo.select
// CHECK:        mhlo.select
// CHECK:        mhlo.return
// CHECK:      return

func.func @byteir.arg_min$return_2(%arg0: tensor<3x4xf32>) -> (tensor<3xf32>, tensor<3xi64>) {
  %0:2 = mhlo.custom_call @byteir.arg_min(%arg0) {byteir_attrs = {axis = 1 : i64, keep_dims = false, select_last_index = false}} : (tensor<3x4xf32>) -> (tensor<3xf32>, tensor<3xi64>)
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xi64>
}
// CHECK-LABEL: func.func @byteir.arg_min$return_2
// CHECK-NOT:  byteir.arg_min
// CHECK-DAG:  mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK-DAG:  mhlo.constant dense<0> : tensor<i64>
// CHECK:      mhlo.iota
// CHECK:      mhlo.broadcast_in_dim
// CHECK:      mhlo.reduce
// CHECK:        mhlo.compare LE
// CHECK:        mhlo.select
// CHECK:        mhlo.compare
// CHECK:        mhlo.minimum
// CHECK:        mhlo.select
// CHECK:        mhlo.select
// CHECK:        mhlo.return
// CHECK:      return
