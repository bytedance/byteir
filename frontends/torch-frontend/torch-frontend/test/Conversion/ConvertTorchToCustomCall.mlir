// RUN: torch-frontend-opt %s -convert-torch-to-custom-call --canonicalize-ext | FileCheck %s

func.func @torch.aten.native_layer_norm(%arg0: !torch.vtensor<[3,7,4,5],f32>) -> !torch.vtensor<[3,7,4,5],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %2 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %2, %1, %0, %float1.000000e-05 : !torch.vtensor<[3,7,4,5],f32>, !torch.list<int>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],f32>, !torch.float -> !torch.vtensor<[3,7,4,5],f32>, !torch.vtensor<[3,7,1,1],f32>, !torch.vtensor<[3,7,1,1],f32>
  return %result0 : !torch.vtensor<[3,7,4,5],f32>
}
// CHECK-LABEL: func.func @torch.aten.native_layer_norm
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK: byteir_attrs = {axis = [2, 3], epsilon = 1.000000e-05 : f64}
// CHECK-NOT: torch.aten.native_layer_norm

func.func @torch.aten.layer_norm(%arg0: !torch.vtensor<[3,7,4,5],f32>) -> !torch.vtensor<[3,7,4,5],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %false = torch.constant.bool false
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %2 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.layer_norm %arg0, %2, %1, %0, %float1.000000e-05, %false : !torch.vtensor<[3,7,4,5],f32>, !torch.list<int>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,7,4,5],f32>
  return %result : !torch.vtensor<[3,7,4,5],f32>
}
// CHECK-LABEL: func.func @torch.aten.layer_norm
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK: byteir_attrs = {axis = [2, 3], epsilon = 1.000000e-05 : f64}
// CHECK-NOT: torch.aten.layer_norm

func.func @torch.aten.layer_norm_v2(%arg0: !torch.vtensor<[3,7,4,5],f32>) -> !torch.vtensor<[3,7,4,5],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %false = torch.constant.bool false
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %2 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.layer_norm %arg0, %2, %1, %0, %float1.000000e-05, %false {eps_outside_sqrt = true} : !torch.vtensor<[3,7,4,5],f32>, !torch.list<int>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,7,4,5],f32>
  return %result : !torch.vtensor<[3,7,4,5],f32>
}
// CHECK-LABEL: func.func @torch.aten.layer_norm
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK: byteir_attrs = {axis = [2, 3], eps_outside_sqrt = true, epsilon = 1.000000e-05 : f64}
// CHECK-NOT: torch.aten.layer_norm

func.func @torch.aten.softmax.int(%t: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %dtype = torch.constant.none
  %dim = torch.constant.int 1
  %ret = torch.aten.softmax.int %t, %dim, %dtype: !torch.vtensor<[2,3],f32>, !torch.int, !torch.none -> !torch.vtensor<[2,3],f32>
  return %ret : !torch.vtensor<[2,3],f32>
}
// CHECK-LABEL: func.func @torch.aten.softmax.int
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.softmax
// CHECK: byteir_attrs = {axis = 1 : i64}
// CHECK-NOT: torch.aten.softmax.int

func.func @torch.aten._softmax(%t: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %half_to_float = torch.constant.bool false
  %dim = torch.constant.int 1
  %ret = torch.aten._softmax %t, %dim, %half_to_float: !torch.vtensor<[2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,3],f32>
  return %ret : !torch.vtensor<[2,3],f32>
}
// CHECK-LABEL: func.func @torch.aten._softmax
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.softmax
// CHECK: byteir_attrs = {axis = 1 : i64}
// CHECK-NOT: torch.aten._softmax

func.func @torch.aten._log_softmax(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten._log_softmax %arg0, %int0, %false : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}
// CHECK-LABEL: func.func @torch.aten._log_softmax
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.log_softmax
// CHECK: byteir_attrs = {axis = 0 : i64}
// CHECK-NOT: torch.aten._log_softmaxcd

func.func @torch.aten.argmax(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int0, %false : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,?],si64>
  return %0 : !torch.vtensor<[?,?],si64>
}
// CHECK-LABEL: func.func @torch.aten.argmax
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.arg_max
// CHECK: byteir_attrs = {axis = 0 : i64, keep_dims = false, select_last_index = false}
// CHECK-NOT: torch.aten.argmax

func.func @torch.aten.max.dim(%arg0: !torch.vtensor<[32,64,21128],f32>) -> !torch.vtensor<[32,64],f32> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %int-1, %false : !torch.vtensor<[32,64,21128],f32>, !torch.int, !torch.bool -> !torch.vtensor<[32,64],f32>, !torch.vtensor<[32,64],si64>
  return %values : !torch.vtensor<[32,64],f32>
}
// CHECK-LABEL: func.func @torch.aten.max.dim
// CHECK: mhlo.reduce
// CHECK: mhlo.max
// CHECK-NOT: torch.aten.max.dim

func.func @torch.aten.max.dim.1(%arg0: !torch.vtensor<[32,64,21128],f32>) -> (!torch.vtensor<[32,64],f32>, !torch.vtensor<[32,64],si64>) {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %int-1, %false : !torch.vtensor<[32,64,21128],f32>, !torch.int, !torch.bool -> !torch.vtensor<[32,64],f32>, !torch.vtensor<[32,64],si64>
  return %values, %indices : !torch.vtensor<[32,64],f32>, !torch.vtensor<[32,64],si64>
}
// CHECK-LABEL: func.func @torch.aten.max.dim.1
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.arg_max
// CHECK: byteir_attrs = {axis = 2 : i64, keep_dims = false, select_last_index = false}
// CHECK-NOT: torch.aten.max.dim

func.func @torch.aten.one_hot(%arg0: !torch.vtensor<[2,3],si64>) -> !torch.vtensor<[2,3,4],si64> {
  %int4 = torch.constant.int 4
  %0 = torch.aten.one_hot %arg0, %int4 : !torch.vtensor<[2,3],si64>, !torch.int -> !torch.vtensor<[2,3,4],si64>
  return %0 : !torch.vtensor<[2,3,4],si64>
}
// CHECK-LABEL: func.func @torch.aten.one_hot
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.one_hot
// CHECK: byteir_attrs = {axis = 2 : i64, depth = 4 : i64, off_value = 0 : i64, on_value = 1 : i64}
// CHECK-NOT: torch.aten.one_hot

func.func @torch.aten.topk(%arg0: !torch.vtensor<[3,10],f32>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>) {
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %true = torch.constant.bool true
  %values, %indices = torch.aten.topk %arg0, %int3, %int1, %true, %true : !torch.vtensor<[3,10],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
  return %values, %indices : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
}
// CHECK-LABEL: func.func @torch.aten.topk
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.top_k
// CHECK: byteir_attrs = {axis = [1], k = 3 : i64, sorted = true}
// CHECH-NOT: torch.aten.topk

func.func @torch.custom.dynamic_partition(%arg0: !torch.vtensor<[10,5],f32>, %arg1: !torch.vtensor<[10],si64>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) {
  %value0, %value1 = "torch.custom_op"(%arg0, %arg1) {custom_op_attrs = {num_partitions = 2 : i64}, custom_op_name = "dynamic_partition"} : (!torch.vtensor<[10,5],f32>, !torch.vtensor<[10],si64>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>)
  return %value0, %value1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>
}
// CHECK-LABEL: func.func @torch.custom.dynamic_partition
// CHECK: mhlo.custom_call
// CHECK-SAME: @tf.DynamicPartition
// CHECK: byteir_attrs = {num_partitions = 2 : i64}
// CHECH-NOT: torch.custom_op

func.func @torch.custom.dynamic_stitch(%arg0: !torch.vtensor<[?],si64>, %arg1: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,?],f32>) {
  %0 = "torch.custom_op"(%arg0, %arg1) {custom_op_attrs = {}, custom_op_name = "dynamic_stitch"} : (!torch.vtensor<[?],si64>, !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
// CHECK-LABEL: func.func @torch.custom.dynamic_stitch
// CHECK: mhlo.custom_call
// CHECK-SAME: @tf.DynamicStitch
// CHECK: byteir_attrs = {}
// CHECH-NOT: torch.custom_op

func.func @torch.custom.dynamic_mask_stitch(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],si64>) -> (!torch.vtensor<[?,?],f32>) {
  %0 = "torch.custom_op"(%arg0, %arg1) {custom_op_attrs = {}, custom_op_name = "dynamic_mask_stitch"} : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?],si64>) -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
// CHECK-LABEL: func.func @torch.custom.dynamic_mask_stitch
// CHECK: mhlo.custom_call
// CHECK-SAME: @tf.DynamicMaskStitch
// CHECK: byteir_attrs = {}
// CHECH-NOT: torch.custom_op
