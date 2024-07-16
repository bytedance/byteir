// RUN: torch-frontend-opt %s -convert-torch-to-custom-call="valid-custom-call-ops=byteir.layer_norm,byteir.softmax,byteir.log_softmax,byteir.nll_loss_forward,byteir.nll_loss_backward,byteir.gelu,byteir.arg_max,byteir.arg_min,byteir.one_hot,byteir.topk,byteir.non_zero" --canonicalize-ext | FileCheck %s
// RUN: torch-frontend-opt %s -convert-torch-to-custom-call --canonicalize-ext | FileCheck %s --check-prefix NONE
// RUN: torch-frontend-opt %s -convert-torch-to-custom-call="valid-custom-call-ops=math.asin" --canonicalize-ext | FileCheck %s --check-prefix MATH

func.func @torch.aten.asin(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.asin %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
// MATH-LABEL: func.func @torch.aten.asin
// MATH: stablehlo.custom_call
// MATH-SAME: @math.asin
// MATH: byteir_attrs = {}
// MATH-NOT: torch.aten.asin

func.func @torch.aten.gelu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %str = torch.constant.str "tanh"
  %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[?,?],f32>, !torch.str -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
// CHECK-LABEL: func.func @torch.aten.gelu
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.gelu
// CHECK: byteir_attrs = {approximate = "tanh"}
// CHECK-NOT: torch.aten.gelu

// NONE-LABEL: func.func @torch.aten.gelu
// NONE-NOT: stablehlo.custom_call
// NONE: torch.aten.gelu

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
// CHECK: stablehlo.custom_call
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
// CHECK: stablehlo.custom_call
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
// CHECK: stablehlo.custom_call
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
// CHECK: stablehlo.custom_call
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
// CHECK: stablehlo.custom_call
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
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.log_softmax
// CHECK: byteir_attrs = {axis = 0 : i64}
// CHECK-NOT: torch.aten._log_softmaxcd

func.func @torch.aten.max.dim(%arg0: !torch.vtensor<[32,64,21128],f32>) -> !torch.vtensor<[32,64],f32> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %int-1, %false : !torch.vtensor<[32,64,21128],f32>, !torch.int, !torch.bool -> !torch.vtensor<[32,64],f32>, !torch.vtensor<[32,64],si64>
  return %values : !torch.vtensor<[32,64],f32>
}
// CHECK-LABEL: func.func @torch.aten.max.dim
// CHECK: torch.aten.max.dim

func.func @torch.aten.max.dim.1(%arg0: !torch.vtensor<[32,64,21128],f32>) -> (!torch.vtensor<[32,64],f32>, !torch.vtensor<[32,64],si64>) {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %int-1, %false : !torch.vtensor<[32,64,21128],f32>, !torch.int, !torch.bool -> !torch.vtensor<[32,64],f32>, !torch.vtensor<[32,64],si64>
  return %values, %indices : !torch.vtensor<[32,64],f32>, !torch.vtensor<[32,64],si64>
}
// CHECK-LABEL: func.func @torch.aten.max.dim.1
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.arg_max
// CHECK: byteir_attrs = {axis = 2 : i64, keep_dims = false, select_last_index = false}
// CHECK-NOT: torch.aten.max.dim

func.func @torch.aten.one_hot(%arg0: !torch.vtensor<[2,3],si64>) -> !torch.vtensor<[2,3,4],si64> {
  %int4 = torch.constant.int 4
  %0 = torch.aten.one_hot %arg0, %int4 : !torch.vtensor<[2,3],si64>, !torch.int -> !torch.vtensor<[2,3,4],si64>
  return %0 : !torch.vtensor<[2,3,4],si64>
}
// CHECK-LABEL: func.func @torch.aten.one_hot
// CHECK: stablehlo.custom_call
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
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.top_k
// CHECK: byteir_attrs = {axis = [1], k = 3 : i64, sorted = true}
// CHECH-NOT: torch.aten.topk

func.func @torch.custom.dynamic_partition(%arg0: !torch.vtensor<[10,5],f32>, %arg1: !torch.vtensor<[10],si64>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) {
  %value0, %value1 = "torch.custom_op"(%arg0, %arg1) {custom_op_attrs = {num_partitions = 2 : i64}, custom_op_name = "dynamic_partition"} : (!torch.vtensor<[10,5],f32>, !torch.vtensor<[10],si64>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>)
  return %value0, %value1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>
}
// CHECK-LABEL: func.func @torch.custom.dynamic_partition
// CHECK: stablehlo.custom_call
// CHECK-SAME: @tf.DynamicPartition
// CHECK: byteir_attrs = {num_partitions = 2 : i64}
// CHECH-NOT: torch.custom_op

func.func @torch.custom.dynamic_stitch(%arg0: !torch.vtensor<[?],si64>, %arg1: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,?],f32>) {
  %0 = "torch.custom_op"(%arg0, %arg1) {custom_op_attrs = {}, custom_op_name = "dynamic_stitch"} : (!torch.vtensor<[?],si64>, !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
// CHECK-LABEL: func.func @torch.custom.dynamic_stitch
// CHECK: stablehlo.custom_call
// CHECK-SAME: @tf.DynamicStitch
// CHECK: byteir_attrs = {}
// CHECH-NOT: torch.custom_op

func.func @torch.custom.dynamic_mask_stitch(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],si64>) -> (!torch.vtensor<[?,?],f32>) {
  %0 = "torch.custom_op"(%arg0, %arg1) {custom_op_attrs = {}, custom_op_name = "dynamic_mask_stitch"} : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?],si64>) -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
// CHECK-LABEL: func.func @torch.custom.dynamic_mask_stitch
// CHECK: stablehlo.custom_call
// CHECK-SAME: @tf.DynamicMaskStitch
// CHECK: byteir_attrs = {}
// CHECH-NOT: torch.custom_op

func.func @torch.aten.nll_loss_forward(%arg0: !torch.vtensor<[8192,50257],f32>, %arg1: !torch.vtensor<[8192],si64>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) {
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %none = torch.constant.none
  %output, %total_weight = torch.aten.nll_loss_forward %arg0, %arg1, %none, %int1, %int-1 : !torch.vtensor<[8192,50257],f32>, !torch.vtensor<[8192],si64>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
  return %output, %total_weight : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
}
// CHECK-LABEL: func.func @torch.aten.nll_loss_forward
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.nll_loss_forward
// CHECK: byteir_attrs = {ignore_index = -1 : i64, reduction = 1 : i64}
// CHECH-NOT: torch.aten.nll_loss_forward

func.func @torch.aten.nll_loss_backward(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[8192,50257],f32>, %arg2: !torch.vtensor<[8192],si64>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[8192,50257],f32>) {
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %none = torch.constant.none
  %0 = torch.aten.nll_loss_backward %arg0, %arg1, %arg2, %none, %int1, %int-1, %arg3 : !torch.vtensor<[],f32>, !torch.vtensor<[8192,50257],f32>, !torch.vtensor<[8192],si64>, !torch.none, !torch.int, !torch.int, !torch.vtensor<[],f32> -> !torch.vtensor<[8192,50257],f32>
  return %0 : !torch.vtensor<[8192,50257],f32>
}
// CHECK-LABEL: func.func @torch.aten.nll_loss_backward
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.nll_loss_backward
// CHECK: byteir_attrs = {ignore_index = -1 : i64, reduction = 1 : i64}
// CHECH-NOT: torch.aten.nll_loss_backward

func.func @torch.byteir.flash_attn_fwd(%arg0: !torch.vtensor<[2,12,256,128],f32>, %arg1: !torch.vtensor<[2,12,256,128],f32>, %arg2: !torch.vtensor<[2,12,256,128],f32>) -> (!torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,256,12],f32>, !torch.vtensor<[2],si64>) {
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %false = torch.constant.bool false
  %0:8 = torch.operator "byteir.flash_attn_fwd"(%arg0, %arg1, %arg2, %float1.000000e-01, %float1.000000e00, %false, %false) : (!torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.float, !torch.float, !torch.bool, !torch.bool) -> (!torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,256,12],f32>, !torch.vtensor<[2,256,12,12],f32>, !torch.vtensor<[2],si64>)
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#7 : !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,12,256,128],f32>, !torch.vtensor<[2,256,12],f32>, !torch.vtensor<[2],si64>
}
// CHECK-LABEL: func.func @torch.byteir.flash_attn_fwd
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.flash_attn_fwd
// CHECK: byteir_attrs = {causal = false, dropout_p = 1.000000e-01 : f64, return_softmax = false, softmax_scale = 1.000000e+00 : f64}
// CHECH-NOT: torch.operator

func.func @torch.byteir.flash_attn_bwd(%arg0: !torch.vtensor<[2,256,12,128],f16>, %arg1: !torch.vtensor<[2,256,12,128],f16>, %arg2: !torch.vtensor<[2,256,12,128],f16>, %arg3: !torch.vtensor<[2,256,12,128],f16>, %arg4: !torch.vtensor<[2,256,12,128],f16>, %arg5: !torch.vtensor<[2,12,256],f32>, %arg6: !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,12,256],f32>, !torch.vtensor<[2,12,256,128],f32>) {
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %true = torch.constant.bool true
  %0:5 = torch.operator "byteir.flash_attn_bwd"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %float1.000000e-01, %float1.000000e00, %true, %arg6) : (!torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,12,256],f32>, !torch.float, !torch.float, !torch.bool, !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,12,256],f32>, !torch.vtensor<[2,12,256,128],f32>)
  return %0#0, %0#1, %0#2, %0#3, %0#4: !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,12,256],f32>, !torch.vtensor<[2,12,256,128],f32>
}
// CHECK-LABEL: func.func @torch.byteir.flash_attn_bwd
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.flash_attn_bwd
// CHECK: byteir_attrs = {causal = true, dropout_p = 1.000000e-01 : f64, softmax_scale = 1.000000e+00 : f64}
// CHECH-NOT: torch.operator

func.func @torch.aten.nonzero(%arg0: !torch.vtensor<[5],si64>) -> !torch.vtensor<[?,1],si64> {
  %0 = torch.aten.nonzero %arg0 : !torch.vtensor<[5],si64> -> !torch.vtensor<[?,1],si64>
  return %0 : !torch.vtensor<[?,1],si64>
}
// CHECK-LABEL: func.func @torch.aten.nonzero
// CHECK: stablehlo.custom_call
// CHECK-SAME: @byteir.non_zero
// CHECK: byteir_attrs = {}
// CHECH-NOT: torch.aten.nonzero
