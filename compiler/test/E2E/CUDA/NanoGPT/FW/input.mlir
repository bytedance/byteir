// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @forward
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<65x384xf32>, %arg1: tensor<256x384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<1152x384xf32>, %arg5: tensor<1152xf32>, %arg6: tensor<384x384xf32>, %arg7: tensor<384xf32>, %arg8: tensor<384xf32>, %arg9: tensor<384xf32>, %arg10: tensor<1536x384xf32>, %arg11: tensor<1536xf32>, %arg12: tensor<384x1536xf32>, %arg13: tensor<384xf32>, %arg14: tensor<384xf32>, %arg15: tensor<384xf32>, %arg16: tensor<1152x384xf32>, %arg17: tensor<1152xf32>, %arg18: tensor<384x384xf32>, %arg19: tensor<384xf32>, %arg20: tensor<384xf32>, %arg21: tensor<384xf32>, %arg22: tensor<1536x384xf32>, %arg23: tensor<1536xf32>, %arg24: tensor<384x1536xf32>, %arg25: tensor<384xf32>, %arg26: tensor<384xf32>, %arg27: tensor<384xf32>, %arg28: tensor<1152x384xf32>, %arg29: tensor<1152xf32>, %arg30: tensor<384x384xf32>, %arg31: tensor<384xf32>, %arg32: tensor<384xf32>, %arg33: tensor<384xf32>, %arg34: tensor<1536x384xf32>, %arg35: tensor<1536xf32>, %arg36: tensor<384x1536xf32>, %arg37: tensor<384xf32>, %arg38: tensor<384xf32>, %arg39: tensor<384xf32>, %arg40: tensor<1152x384xf32>, %arg41: tensor<1152xf32>, %arg42: tensor<384x384xf32>, %arg43: tensor<384xf32>, %arg44: tensor<384xf32>, %arg45: tensor<384xf32>, %arg46: tensor<1536x384xf32>, %arg47: tensor<1536xf32>, %arg48: tensor<384x1536xf32>, %arg49: tensor<384xf32>, %arg50: tensor<384xf32>, %arg51: tensor<384xf32>, %arg52: tensor<1152x384xf32>, %arg53: tensor<1152xf32>, %arg54: tensor<384x384xf32>, %arg55: tensor<384xf32>, %arg56: tensor<384xf32>, %arg57: tensor<384xf32>, %arg58: tensor<1536x384xf32>, %arg59: tensor<1536xf32>, %arg60: tensor<384x1536xf32>, %arg61: tensor<384xf32>, %arg62: tensor<384xf32>, %arg63: tensor<384xf32>, %arg64: tensor<1152x384xf32>, %arg65: tensor<1152xf32>, %arg66: tensor<384x384xf32>, %arg67: tensor<384xf32>, %arg68: tensor<384xf32>, %arg69: tensor<384xf32>, %arg70: tensor<1536x384xf32>, %arg71: tensor<1536xf32>, %arg72: tensor<384x1536xf32>, %arg73: tensor<384xf32>, %arg74: tensor<384xf32>, %arg75: tensor<384xf32>, %arg76: tensor<65x384xf32>, %arg77: tensor<1x1x256x256xf32>, %arg78: tensor<1x1x256x256xf32>, %arg79: tensor<1x1x256x256xf32>, %arg80: tensor<1x1x256x256xf32>, %arg81: tensor<1x1x256x256xf32>, %arg82: tensor<1x1x256x256xf32>, %arg83: tensor<64x256xi64>, %arg84: tensor<64x256xi64>):
    %0 = "mhlo.constant"() {value = dense<8.000000e-01> : tensor<64x256x384xf32>} : () -> tensor<64x256x384xf32>
    %1 = "mhlo.constant"() {value = dense<8.000000e-01> : tensor<64x256x384xf64>} : () -> tensor<64x256x384xf64>
    %2 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %3 = "mhlo.constant"() {value = dense<0.797884583> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %4 = "mhlo.constant"() {value = dense<4.471500e-02> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %5 = "mhlo.constant"() {value = dense<3.000000e+00> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %6 = "mhlo.constant"() {value = dense<5.000000e-01> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %7 = "mhlo.constant"() {value = dense<8.000000e-01> : tensor<64x6x256x256xf32>} : () -> tensor<64x6x256x256xf32>
    %8 = "mhlo.constant"() {value = dense<8.000000e-01> : tensor<64x6x256x256xf64>} : () -> tensor<64x6x256x256xf64>
    %9 = "mhlo.constant"() {value = dense<0xFF800000> : tensor<64x6x256x256xf32>} : () -> tensor<64x6x256x256xf32>
    %10 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<1x1x256x256xf32>} : () -> tensor<1x1x256x256xf32>
    %11 = "mhlo.constant"() {value = dense<1.250000e-01> : tensor<64x6x256x256xf32>} : () -> tensor<64x6x256x256xf32>
    %12 = "arith.constant"() {value = dense<[64, 256, 384]> : tensor<3xi64>} : () -> tensor<3xi64>
    %13 = "arith.constant"() {value = dense<[64, 6, 256, 256]> : tensor<4xi64>} : () -> tensor<4xi64>
    %14 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<f64>} : () -> tensor<f64>
    %15 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f64>} : () -> tensor<f64>
    %16 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<256xi64>
    %17 = "mhlo.reshape"(%16) : (tensor<256xi64>) -> tensor<1x256xi64>
    %18 = "mhlo.gather"(%arg0, %arg83) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 384]> : tensor<2xi64>} : (tensor<65x384xf32>, tensor<64x256xi64>) -> tensor<64x256x384xf32>
    %19 = "mhlo.gather"(%arg1, %17) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 384]> : tensor<2xi64>} : (tensor<256x384xf32>, tensor<1x256xi64>) -> tensor<1x256x384xf32>
    %20 = "mhlo.broadcast_in_dim"(%19) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x256x384xf32>) -> tensor<64x256x384xf32>
    %21 = "mhlo.add"(%18, %20) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %22 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %23 = "mhlo.compare"(%22, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %24 = "mhlo.convert"(%23) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %25 = "mhlo.divide"(%24, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %26 = "mhlo.multiply"(%21, %25) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %27:3 = "mhlo.custom_call"(%26, %arg2, %arg3) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %28 = "mhlo.transpose"(%arg4) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1152x384xf32>) -> tensor<384x1152xf32>
    %29 = "mhlo.reshape"(%27#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %30 = "mhlo.dot"(%29, %28) : (tensor<16384x384xf32>, tensor<384x1152xf32>) -> tensor<16384x1152xf32>
    %31 = "mhlo.broadcast_in_dim"(%arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1152xf32>) -> tensor<16384x1152xf32>
    %32 = "mhlo.add"(%31, %30) : (tensor<16384x1152xf32>, tensor<16384x1152xf32>) -> tensor<16384x1152xf32>
    %33 = "mhlo.reshape"(%32) : (tensor<16384x1152xf32>) -> tensor<64x256x1152xf32>
    %34 = "mhlo.slice"(%33) {limit_indices = dense<[64, 256, 384]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %35 = "mhlo.slice"(%33) {limit_indices = dense<[64, 256, 768]> : tensor<3xi64>, start_indices = dense<[0, 0, 384]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %36 = "mhlo.slice"(%33) {limit_indices = dense<[64, 256, 1152]> : tensor<3xi64>, start_indices = dense<[0, 0, 768]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %37 = "mhlo.reshape"(%35) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %38 = "mhlo.reshape"(%34) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %39 = "mhlo.transpose"(%38) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %40 = "mhlo.reshape"(%36) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %41 = "mhlo.transpose"(%40) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %42 = "mhlo.transpose"(%37) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x64x256xf32>
    %43 = "mhlo.reshape"(%39) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %44 = "mhlo.reshape"(%42) : (tensor<64x6x64x256xf32>) -> tensor<384x64x256xf32>
    %45 = "mhlo.dot_general"(%43, %44) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %46 = "mhlo.reshape"(%45) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %47 = "mhlo.multiply"(%46, %11) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %48 = "mhlo.compare"(%arg77, %10) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1>
    %49 = "mhlo.broadcast_in_dim"(%48) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %50 = "mhlo.select"(%49, %9, %47) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %51 = "mhlo.custom_call"(%50) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %52 = "mhlo.rng"(%15, %14, %13) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<4xi64>) -> tensor<64x6x256x256xf64>
    %53 = "mhlo.compare"(%52, %8) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x6x256x256xf64>, tensor<64x6x256x256xf64>) -> tensor<64x6x256x256xi1>
    %54 = "mhlo.convert"(%53) : (tensor<64x6x256x256xi1>) -> tensor<64x6x256x256xf32>
    %55 = "mhlo.divide"(%54, %7) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %56 = "mhlo.multiply"(%51, %55) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %57 = "mhlo.reshape"(%56) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %58 = "mhlo.reshape"(%41) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %59 = "mhlo.dot_general"(%57, %58) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %60 = "mhlo.reshape"(%59) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %61 = "mhlo.transpose"(%60) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %62 = "mhlo.transpose"(%arg6) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %63 = "mhlo.reshape"(%61) : (tensor<64x256x6x64xf32>) -> tensor<16384x384xf32>
    %64 = "mhlo.dot"(%63, %62) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %65 = "mhlo.broadcast_in_dim"(%arg7) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %66 = "mhlo.add"(%65, %64) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %67 = "mhlo.reshape"(%66) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %68 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %69 = "mhlo.compare"(%68, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %70 = "mhlo.convert"(%69) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %71 = "mhlo.divide"(%70, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %72 = "mhlo.multiply"(%67, %71) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %73 = "mhlo.add"(%26, %72) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %74:3 = "mhlo.custom_call"(%73, %arg8, %arg9) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %75 = "mhlo.transpose"(%arg10) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %76 = "mhlo.reshape"(%74#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %77 = "mhlo.dot"(%76, %75) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %78 = "mhlo.broadcast_in_dim"(%arg11) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1536xf32>) -> tensor<16384x1536xf32>
    %79 = "mhlo.add"(%78, %77) : (tensor<16384x1536xf32>, tensor<16384x1536xf32>) -> tensor<16384x1536xf32>
    %80 = "mhlo.reshape"(%79) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %81 = "mhlo.multiply"(%80, %6) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %82 = "mhlo.power"(%80, %5) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %83 = "mhlo.multiply"(%82, %4) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %84 = "mhlo.add"(%80, %83) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %85 = "mhlo.multiply"(%84, %3) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %86 = "mhlo.tanh"(%85) : (tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %87 = "mhlo.add"(%86, %2) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %88 = "mhlo.multiply"(%81, %87) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %89 = "mhlo.transpose"(%arg12) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %90 = "mhlo.reshape"(%88) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %91 = "mhlo.dot"(%90, %89) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %92 = "mhlo.broadcast_in_dim"(%arg13) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %93 = "mhlo.add"(%92, %91) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %94 = "mhlo.reshape"(%93) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %95 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %96 = "mhlo.compare"(%95, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %97 = "mhlo.convert"(%96) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %98 = "mhlo.divide"(%97, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %99 = "mhlo.multiply"(%94, %98) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %100 = "mhlo.add"(%73, %99) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %101:3 = "mhlo.custom_call"(%100, %arg14, %arg15) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %102 = "mhlo.transpose"(%arg16) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1152x384xf32>) -> tensor<384x1152xf32>
    %103 = "mhlo.reshape"(%101#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %104 = "mhlo.dot"(%103, %102) : (tensor<16384x384xf32>, tensor<384x1152xf32>) -> tensor<16384x1152xf32>
    %105 = "mhlo.broadcast_in_dim"(%arg17) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1152xf32>) -> tensor<16384x1152xf32>
    %106 = "mhlo.add"(%105, %104) : (tensor<16384x1152xf32>, tensor<16384x1152xf32>) -> tensor<16384x1152xf32>
    %107 = "mhlo.reshape"(%106) : (tensor<16384x1152xf32>) -> tensor<64x256x1152xf32>
    %108 = "mhlo.slice"(%107) {limit_indices = dense<[64, 256, 384]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %109 = "mhlo.slice"(%107) {limit_indices = dense<[64, 256, 768]> : tensor<3xi64>, start_indices = dense<[0, 0, 384]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %110 = "mhlo.slice"(%107) {limit_indices = dense<[64, 256, 1152]> : tensor<3xi64>, start_indices = dense<[0, 0, 768]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %111 = "mhlo.reshape"(%109) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %112 = "mhlo.reshape"(%108) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %113 = "mhlo.transpose"(%112) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %114 = "mhlo.reshape"(%110) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %115 = "mhlo.transpose"(%114) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %116 = "mhlo.transpose"(%111) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x64x256xf32>
    %117 = "mhlo.reshape"(%113) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %118 = "mhlo.reshape"(%116) : (tensor<64x6x64x256xf32>) -> tensor<384x64x256xf32>
    %119 = "mhlo.dot_general"(%117, %118) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %120 = "mhlo.reshape"(%119) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %121 = "mhlo.multiply"(%120, %11) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %122 = "mhlo.compare"(%arg78, %10) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1>
    %123 = "mhlo.broadcast_in_dim"(%122) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %124 = "mhlo.select"(%123, %9, %121) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %125 = "mhlo.custom_call"(%124) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %126 = "mhlo.rng"(%15, %14, %13) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<4xi64>) -> tensor<64x6x256x256xf64>
    %127 = "mhlo.compare"(%126, %8) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x6x256x256xf64>, tensor<64x6x256x256xf64>) -> tensor<64x6x256x256xi1>
    %128 = "mhlo.convert"(%127) : (tensor<64x6x256x256xi1>) -> tensor<64x6x256x256xf32>
    %129 = "mhlo.divide"(%128, %7) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %130 = "mhlo.multiply"(%125, %129) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %131 = "mhlo.reshape"(%130) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %132 = "mhlo.reshape"(%115) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %133 = "mhlo.dot_general"(%131, %132) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %134 = "mhlo.reshape"(%133) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %135 = "mhlo.transpose"(%134) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %136 = "mhlo.transpose"(%arg18) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %137 = "mhlo.reshape"(%135) : (tensor<64x256x6x64xf32>) -> tensor<16384x384xf32>
    %138 = "mhlo.dot"(%137, %136) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %139 = "mhlo.broadcast_in_dim"(%arg19) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %140 = "mhlo.add"(%139, %138) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %141 = "mhlo.reshape"(%140) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %142 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %143 = "mhlo.compare"(%142, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %144 = "mhlo.convert"(%143) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %145 = "mhlo.divide"(%144, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %146 = "mhlo.multiply"(%141, %145) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %147 = "mhlo.add"(%100, %146) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %148:3 = "mhlo.custom_call"(%147, %arg20, %arg21) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %149 = "mhlo.transpose"(%arg22) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %150 = "mhlo.reshape"(%148#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %151 = "mhlo.dot"(%150, %149) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %152 = "mhlo.broadcast_in_dim"(%arg23) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1536xf32>) -> tensor<16384x1536xf32>
    %153 = "mhlo.add"(%152, %151) : (tensor<16384x1536xf32>, tensor<16384x1536xf32>) -> tensor<16384x1536xf32>
    %154 = "mhlo.reshape"(%153) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %155 = "mhlo.multiply"(%154, %6) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %156 = "mhlo.power"(%154, %5) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %157 = "mhlo.multiply"(%156, %4) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %158 = "mhlo.add"(%154, %157) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %159 = "mhlo.multiply"(%158, %3) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %160 = "mhlo.tanh"(%159) : (tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %161 = "mhlo.add"(%160, %2) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %162 = "mhlo.multiply"(%155, %161) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %163 = "mhlo.transpose"(%arg24) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %164 = "mhlo.reshape"(%162) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %165 = "mhlo.dot"(%164, %163) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %166 = "mhlo.broadcast_in_dim"(%arg25) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %167 = "mhlo.add"(%166, %165) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %168 = "mhlo.reshape"(%167) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %169 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %170 = "mhlo.compare"(%169, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %171 = "mhlo.convert"(%170) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %172 = "mhlo.divide"(%171, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %173 = "mhlo.multiply"(%168, %172) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %174 = "mhlo.add"(%147, %173) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %175:3 = "mhlo.custom_call"(%174, %arg26, %arg27) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %176 = "mhlo.transpose"(%arg28) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1152x384xf32>) -> tensor<384x1152xf32>
    %177 = "mhlo.reshape"(%175#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %178 = "mhlo.dot"(%177, %176) : (tensor<16384x384xf32>, tensor<384x1152xf32>) -> tensor<16384x1152xf32>
    %179 = "mhlo.broadcast_in_dim"(%arg29) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1152xf32>) -> tensor<16384x1152xf32>
    %180 = "mhlo.add"(%179, %178) : (tensor<16384x1152xf32>, tensor<16384x1152xf32>) -> tensor<16384x1152xf32>
    %181 = "mhlo.reshape"(%180) : (tensor<16384x1152xf32>) -> tensor<64x256x1152xf32>
    %182 = "mhlo.slice"(%181) {limit_indices = dense<[64, 256, 384]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %183 = "mhlo.slice"(%181) {limit_indices = dense<[64, 256, 768]> : tensor<3xi64>, start_indices = dense<[0, 0, 384]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %184 = "mhlo.slice"(%181) {limit_indices = dense<[64, 256, 1152]> : tensor<3xi64>, start_indices = dense<[0, 0, 768]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %185 = "mhlo.reshape"(%183) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %186 = "mhlo.reshape"(%182) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %187 = "mhlo.transpose"(%186) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %188 = "mhlo.reshape"(%184) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %189 = "mhlo.transpose"(%188) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %190 = "mhlo.transpose"(%185) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x64x256xf32>
    %191 = "mhlo.reshape"(%187) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %192 = "mhlo.reshape"(%190) : (tensor<64x6x64x256xf32>) -> tensor<384x64x256xf32>
    %193 = "mhlo.dot_general"(%191, %192) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %194 = "mhlo.reshape"(%193) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %195 = "mhlo.multiply"(%194, %11) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %196 = "mhlo.compare"(%arg79, %10) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1>
    %197 = "mhlo.broadcast_in_dim"(%196) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %198 = "mhlo.select"(%197, %9, %195) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %199 = "mhlo.custom_call"(%198) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %200 = "mhlo.rng"(%15, %14, %13) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<4xi64>) -> tensor<64x6x256x256xf64>
    %201 = "mhlo.compare"(%200, %8) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x6x256x256xf64>, tensor<64x6x256x256xf64>) -> tensor<64x6x256x256xi1>
    %202 = "mhlo.convert"(%201) : (tensor<64x6x256x256xi1>) -> tensor<64x6x256x256xf32>
    %203 = "mhlo.divide"(%202, %7) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %204 = "mhlo.multiply"(%199, %203) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %205 = "mhlo.reshape"(%204) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %206 = "mhlo.reshape"(%189) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %207 = "mhlo.dot_general"(%205, %206) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %208 = "mhlo.reshape"(%207) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %209 = "mhlo.transpose"(%208) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %210 = "mhlo.transpose"(%arg30) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %211 = "mhlo.reshape"(%209) : (tensor<64x256x6x64xf32>) -> tensor<16384x384xf32>
    %212 = "mhlo.dot"(%211, %210) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %213 = "mhlo.broadcast_in_dim"(%arg31) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %214 = "mhlo.add"(%213, %212) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %215 = "mhlo.reshape"(%214) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %216 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %217 = "mhlo.compare"(%216, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %218 = "mhlo.convert"(%217) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %219 = "mhlo.divide"(%218, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %220 = "mhlo.multiply"(%215, %219) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %221 = "mhlo.add"(%174, %220) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %222:3 = "mhlo.custom_call"(%221, %arg32, %arg33) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %223 = "mhlo.transpose"(%arg34) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %224 = "mhlo.reshape"(%222#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %225 = "mhlo.dot"(%224, %223) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %226 = "mhlo.broadcast_in_dim"(%arg35) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1536xf32>) -> tensor<16384x1536xf32>
    %227 = "mhlo.add"(%226, %225) : (tensor<16384x1536xf32>, tensor<16384x1536xf32>) -> tensor<16384x1536xf32>
    %228 = "mhlo.reshape"(%227) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %229 = "mhlo.multiply"(%228, %6) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %230 = "mhlo.power"(%228, %5) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %231 = "mhlo.multiply"(%230, %4) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %232 = "mhlo.add"(%228, %231) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %233 = "mhlo.multiply"(%232, %3) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %234 = "mhlo.tanh"(%233) : (tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %235 = "mhlo.add"(%234, %2) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %236 = "mhlo.multiply"(%229, %235) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %237 = "mhlo.transpose"(%arg36) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %238 = "mhlo.reshape"(%236) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %239 = "mhlo.dot"(%238, %237) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %240 = "mhlo.broadcast_in_dim"(%arg37) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %241 = "mhlo.add"(%240, %239) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %242 = "mhlo.reshape"(%241) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %243 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %244 = "mhlo.compare"(%243, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %245 = "mhlo.convert"(%244) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %246 = "mhlo.divide"(%245, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %247 = "mhlo.multiply"(%242, %246) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %248 = "mhlo.add"(%221, %247) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %249:3 = "mhlo.custom_call"(%248, %arg38, %arg39) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %250 = "mhlo.transpose"(%arg40) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1152x384xf32>) -> tensor<384x1152xf32>
    %251 = "mhlo.reshape"(%249#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %252 = "mhlo.dot"(%251, %250) : (tensor<16384x384xf32>, tensor<384x1152xf32>) -> tensor<16384x1152xf32>
    %253 = "mhlo.broadcast_in_dim"(%arg41) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1152xf32>) -> tensor<16384x1152xf32>
    %254 = "mhlo.add"(%253, %252) : (tensor<16384x1152xf32>, tensor<16384x1152xf32>) -> tensor<16384x1152xf32>
    %255 = "mhlo.reshape"(%254) : (tensor<16384x1152xf32>) -> tensor<64x256x1152xf32>
    %256 = "mhlo.slice"(%255) {limit_indices = dense<[64, 256, 384]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %257 = "mhlo.slice"(%255) {limit_indices = dense<[64, 256, 768]> : tensor<3xi64>, start_indices = dense<[0, 0, 384]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %258 = "mhlo.slice"(%255) {limit_indices = dense<[64, 256, 1152]> : tensor<3xi64>, start_indices = dense<[0, 0, 768]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %259 = "mhlo.reshape"(%257) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %260 = "mhlo.reshape"(%256) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %261 = "mhlo.transpose"(%260) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %262 = "mhlo.reshape"(%258) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %263 = "mhlo.transpose"(%262) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %264 = "mhlo.transpose"(%259) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x64x256xf32>
    %265 = "mhlo.reshape"(%261) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %266 = "mhlo.reshape"(%264) : (tensor<64x6x64x256xf32>) -> tensor<384x64x256xf32>
    %267 = "mhlo.dot_general"(%265, %266) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %268 = "mhlo.reshape"(%267) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %269 = "mhlo.multiply"(%268, %11) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %270 = "mhlo.compare"(%arg80, %10) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1>
    %271 = "mhlo.broadcast_in_dim"(%270) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %272 = "mhlo.select"(%271, %9, %269) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %273 = "mhlo.custom_call"(%272) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %274 = "mhlo.rng"(%15, %14, %13) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<4xi64>) -> tensor<64x6x256x256xf64>
    %275 = "mhlo.compare"(%274, %8) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x6x256x256xf64>, tensor<64x6x256x256xf64>) -> tensor<64x6x256x256xi1>
    %276 = "mhlo.convert"(%275) : (tensor<64x6x256x256xi1>) -> tensor<64x6x256x256xf32>
    %277 = "mhlo.divide"(%276, %7) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %278 = "mhlo.multiply"(%273, %277) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %279 = "mhlo.reshape"(%278) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %280 = "mhlo.reshape"(%263) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %281 = "mhlo.dot_general"(%279, %280) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %282 = "mhlo.reshape"(%281) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %283 = "mhlo.transpose"(%282) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %284 = "mhlo.transpose"(%arg42) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %285 = "mhlo.reshape"(%283) : (tensor<64x256x6x64xf32>) -> tensor<16384x384xf32>
    %286 = "mhlo.dot"(%285, %284) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %287 = "mhlo.broadcast_in_dim"(%arg43) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %288 = "mhlo.add"(%287, %286) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %289 = "mhlo.reshape"(%288) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %290 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %291 = "mhlo.compare"(%290, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %292 = "mhlo.convert"(%291) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %293 = "mhlo.divide"(%292, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %294 = "mhlo.multiply"(%289, %293) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %295 = "mhlo.add"(%248, %294) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %296:3 = "mhlo.custom_call"(%295, %arg44, %arg45) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %297 = "mhlo.transpose"(%arg46) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %298 = "mhlo.reshape"(%296#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %299 = "mhlo.dot"(%298, %297) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %300 = "mhlo.broadcast_in_dim"(%arg47) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1536xf32>) -> tensor<16384x1536xf32>
    %301 = "mhlo.add"(%300, %299) : (tensor<16384x1536xf32>, tensor<16384x1536xf32>) -> tensor<16384x1536xf32>
    %302 = "mhlo.reshape"(%301) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %303 = "mhlo.multiply"(%302, %6) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %304 = "mhlo.power"(%302, %5) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %305 = "mhlo.multiply"(%304, %4) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %306 = "mhlo.add"(%302, %305) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %307 = "mhlo.multiply"(%306, %3) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %308 = "mhlo.tanh"(%307) : (tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %309 = "mhlo.add"(%308, %2) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %310 = "mhlo.multiply"(%303, %309) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %311 = "mhlo.transpose"(%arg48) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %312 = "mhlo.reshape"(%310) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %313 = "mhlo.dot"(%312, %311) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %314 = "mhlo.broadcast_in_dim"(%arg49) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %315 = "mhlo.add"(%314, %313) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %316 = "mhlo.reshape"(%315) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %317 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %318 = "mhlo.compare"(%317, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %319 = "mhlo.convert"(%318) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %320 = "mhlo.divide"(%319, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %321 = "mhlo.multiply"(%316, %320) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %322 = "mhlo.add"(%295, %321) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %323:3 = "mhlo.custom_call"(%322, %arg50, %arg51) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %324 = "mhlo.transpose"(%arg52) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1152x384xf32>) -> tensor<384x1152xf32>
    %325 = "mhlo.reshape"(%323#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %326 = "mhlo.dot"(%325, %324) : (tensor<16384x384xf32>, tensor<384x1152xf32>) -> tensor<16384x1152xf32>
    %327 = "mhlo.broadcast_in_dim"(%arg53) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1152xf32>) -> tensor<16384x1152xf32>
    %328 = "mhlo.add"(%327, %326) : (tensor<16384x1152xf32>, tensor<16384x1152xf32>) -> tensor<16384x1152xf32>
    %329 = "mhlo.reshape"(%328) : (tensor<16384x1152xf32>) -> tensor<64x256x1152xf32>
    %330 = "mhlo.slice"(%329) {limit_indices = dense<[64, 256, 384]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %331 = "mhlo.slice"(%329) {limit_indices = dense<[64, 256, 768]> : tensor<3xi64>, start_indices = dense<[0, 0, 384]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %332 = "mhlo.slice"(%329) {limit_indices = dense<[64, 256, 1152]> : tensor<3xi64>, start_indices = dense<[0, 0, 768]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %333 = "mhlo.reshape"(%331) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %334 = "mhlo.reshape"(%330) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %335 = "mhlo.transpose"(%334) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %336 = "mhlo.reshape"(%332) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %337 = "mhlo.transpose"(%336) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %338 = "mhlo.transpose"(%333) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x64x256xf32>
    %339 = "mhlo.reshape"(%335) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %340 = "mhlo.reshape"(%338) : (tensor<64x6x64x256xf32>) -> tensor<384x64x256xf32>
    %341 = "mhlo.dot_general"(%339, %340) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %342 = "mhlo.reshape"(%341) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %343 = "mhlo.multiply"(%342, %11) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %344 = "mhlo.compare"(%arg81, %10) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1>
    %345 = "mhlo.broadcast_in_dim"(%344) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %346 = "mhlo.select"(%345, %9, %343) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %347 = "mhlo.custom_call"(%346) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %348 = "mhlo.rng"(%15, %14, %13) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<4xi64>) -> tensor<64x6x256x256xf64>
    %349 = "mhlo.compare"(%348, %8) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x6x256x256xf64>, tensor<64x6x256x256xf64>) -> tensor<64x6x256x256xi1>
    %350 = "mhlo.convert"(%349) : (tensor<64x6x256x256xi1>) -> tensor<64x6x256x256xf32>
    %351 = "mhlo.divide"(%350, %7) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %352 = "mhlo.multiply"(%347, %351) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %353 = "mhlo.reshape"(%352) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %354 = "mhlo.reshape"(%337) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %355 = "mhlo.dot_general"(%353, %354) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %356 = "mhlo.reshape"(%355) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %357 = "mhlo.transpose"(%356) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %358 = "mhlo.transpose"(%arg54) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %359 = "mhlo.reshape"(%357) : (tensor<64x256x6x64xf32>) -> tensor<16384x384xf32>
    %360 = "mhlo.dot"(%359, %358) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %361 = "mhlo.broadcast_in_dim"(%arg55) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %362 = "mhlo.add"(%361, %360) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %363 = "mhlo.reshape"(%362) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %364 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %365 = "mhlo.compare"(%364, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %366 = "mhlo.convert"(%365) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %367 = "mhlo.divide"(%366, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %368 = "mhlo.multiply"(%363, %367) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %369 = "mhlo.add"(%322, %368) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %370:3 = "mhlo.custom_call"(%369, %arg56, %arg57) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %371 = "mhlo.transpose"(%arg58) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %372 = "mhlo.reshape"(%370#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %373 = "mhlo.dot"(%372, %371) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %374 = "mhlo.broadcast_in_dim"(%arg59) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1536xf32>) -> tensor<16384x1536xf32>
    %375 = "mhlo.add"(%374, %373) : (tensor<16384x1536xf32>, tensor<16384x1536xf32>) -> tensor<16384x1536xf32>
    %376 = "mhlo.reshape"(%375) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %377 = "mhlo.multiply"(%376, %6) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %378 = "mhlo.power"(%376, %5) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %379 = "mhlo.multiply"(%378, %4) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %380 = "mhlo.add"(%376, %379) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %381 = "mhlo.multiply"(%380, %3) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %382 = "mhlo.tanh"(%381) : (tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %383 = "mhlo.add"(%382, %2) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %384 = "mhlo.multiply"(%377, %383) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %385 = "mhlo.transpose"(%arg60) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %386 = "mhlo.reshape"(%384) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %387 = "mhlo.dot"(%386, %385) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %388 = "mhlo.broadcast_in_dim"(%arg61) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %389 = "mhlo.add"(%388, %387) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %390 = "mhlo.reshape"(%389) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %391 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %392 = "mhlo.compare"(%391, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %393 = "mhlo.convert"(%392) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %394 = "mhlo.divide"(%393, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %395 = "mhlo.multiply"(%390, %394) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %396 = "mhlo.add"(%369, %395) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %397:3 = "mhlo.custom_call"(%396, %arg62, %arg63) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %398 = "mhlo.transpose"(%arg64) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1152x384xf32>) -> tensor<384x1152xf32>
    %399 = "mhlo.reshape"(%397#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %400 = "mhlo.dot"(%399, %398) : (tensor<16384x384xf32>, tensor<384x1152xf32>) -> tensor<16384x1152xf32>
    %401 = "mhlo.broadcast_in_dim"(%arg65) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1152xf32>) -> tensor<16384x1152xf32>
    %402 = "mhlo.add"(%401, %400) : (tensor<16384x1152xf32>, tensor<16384x1152xf32>) -> tensor<16384x1152xf32>
    %403 = "mhlo.reshape"(%402) : (tensor<16384x1152xf32>) -> tensor<64x256x1152xf32>
    %404 = "mhlo.slice"(%403) {limit_indices = dense<[64, 256, 384]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %405 = "mhlo.slice"(%403) {limit_indices = dense<[64, 256, 768]> : tensor<3xi64>, start_indices = dense<[0, 0, 384]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %406 = "mhlo.slice"(%403) {limit_indices = dense<[64, 256, 1152]> : tensor<3xi64>, start_indices = dense<[0, 0, 768]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<64x256x1152xf32>) -> tensor<64x256x384xf32>
    %407 = "mhlo.reshape"(%405) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %408 = "mhlo.reshape"(%404) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %409 = "mhlo.transpose"(%408) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %410 = "mhlo.reshape"(%406) : (tensor<64x256x384xf32>) -> tensor<64x256x6x64xf32>
    %411 = "mhlo.transpose"(%410) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %412 = "mhlo.transpose"(%407) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x64x256xf32>
    %413 = "mhlo.reshape"(%409) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %414 = "mhlo.reshape"(%412) : (tensor<64x6x64x256xf32>) -> tensor<384x64x256xf32>
    %415 = "mhlo.dot_general"(%413, %414) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %416 = "mhlo.reshape"(%415) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %417 = "mhlo.multiply"(%416, %11) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %418 = "mhlo.compare"(%arg82, %10) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1>
    %419 = "mhlo.broadcast_in_dim"(%418) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %420 = "mhlo.select"(%419, %9, %417) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %421 = "mhlo.custom_call"(%420) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %422 = "mhlo.rng"(%15, %14, %13) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<4xi64>) -> tensor<64x6x256x256xf64>
    %423 = "mhlo.compare"(%422, %8) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x6x256x256xf64>, tensor<64x6x256x256xf64>) -> tensor<64x6x256x256xi1>
    %424 = "mhlo.convert"(%423) : (tensor<64x6x256x256xi1>) -> tensor<64x6x256x256xf32>
    %425 = "mhlo.divide"(%424, %7) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %426 = "mhlo.multiply"(%421, %425) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %427 = "mhlo.reshape"(%426) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %428 = "mhlo.reshape"(%411) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %429 = "mhlo.dot_general"(%427, %428) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %430 = "mhlo.reshape"(%429) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %431 = "mhlo.transpose"(%430) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %432 = "mhlo.transpose"(%arg66) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %433 = "mhlo.reshape"(%431) : (tensor<64x256x6x64xf32>) -> tensor<16384x384xf32>
    %434 = "mhlo.dot"(%433, %432) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %435 = "mhlo.broadcast_in_dim"(%arg67) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %436 = "mhlo.add"(%435, %434) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %437 = "mhlo.reshape"(%436) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %438 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %439 = "mhlo.compare"(%438, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %440 = "mhlo.convert"(%439) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %441 = "mhlo.divide"(%440, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %442 = "mhlo.multiply"(%437, %441) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %443 = "mhlo.add"(%396, %442) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %444:3 = "mhlo.custom_call"(%443, %arg68, %arg69) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %445 = "mhlo.transpose"(%arg70) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %446 = "mhlo.reshape"(%444#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %447 = "mhlo.dot"(%446, %445) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %448 = "mhlo.broadcast_in_dim"(%arg71) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1536xf32>) -> tensor<16384x1536xf32>
    %449 = "mhlo.add"(%448, %447) : (tensor<16384x1536xf32>, tensor<16384x1536xf32>) -> tensor<16384x1536xf32>
    %450 = "mhlo.reshape"(%449) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %451 = "mhlo.multiply"(%450, %6) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %452 = "mhlo.power"(%450, %5) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %453 = "mhlo.multiply"(%452, %4) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %454 = "mhlo.add"(%450, %453) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %455 = "mhlo.multiply"(%454, %3) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %456 = "mhlo.tanh"(%455) : (tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %457 = "mhlo.add"(%456, %2) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %458 = "mhlo.multiply"(%451, %457) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %459 = "mhlo.transpose"(%arg72) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %460 = "mhlo.reshape"(%458) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %461 = "mhlo.dot"(%460, %459) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %462 = "mhlo.broadcast_in_dim"(%arg73) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %463 = "mhlo.add"(%462, %461) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    %464 = "mhlo.reshape"(%463) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %465 = "mhlo.rng"(%15, %14, %12) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<3xi64>) -> tensor<64x256x384xf64>
    %466 = "mhlo.compare"(%465, %1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<64x256x384xf64>, tensor<64x256x384xf64>) -> tensor<64x256x384xi1>
    %467 = "mhlo.convert"(%466) : (tensor<64x256x384xi1>) -> tensor<64x256x384xf32>
    %468 = "mhlo.divide"(%467, %0) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %469 = "mhlo.multiply"(%464, %468) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %470 = "mhlo.add"(%443, %469) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %471:3 = "mhlo.custom_call"(%470, %arg74, %arg75) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<64x256x384xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>)
    %472 = "mhlo.transpose"(%arg76) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<65x384xf32>) -> tensor<384x65xf32>
    %473 = "mhlo.reshape"(%471#0) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %474 = "mhlo.dot"(%473, %472) : (tensor<16384x384xf32>, tensor<384x65xf32>) -> tensor<16384x65xf32>
    %475 = "mhlo.reshape"(%474) : (tensor<16384x65xf32>) -> tensor<64x256x65xf32>
    %476 = "mhlo.reshape"(%arg84) : (tensor<64x256xi64>) -> tensor<16384xi64>
    %477 = "mhlo.custom_call"(%474) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 1 : i64}, call_target_name = "byteir.log_softmax", called_computations = [], has_side_effect = false} : (tensor<16384x65xf32>) -> tensor<16384x65xf32>
    %478:2 = "mhlo.custom_call"(%477, %476) {api_version = 1 : i32, backend_config = "", byteir_attrs = {ignore_index = -1 : i64, reduction = 1 : i64}, call_target_name = "byteir.nll_loss_forward", called_computations = [], has_side_effect = false} : (tensor<16384x65xf32>, tensor<16384xi64>) -> (tensor<f32>, tensor<f32>)
    "func.return"(%475, %478#0, %474, %196, %55, %211, %444#2, %arg14, %arg56, %471#1, %441, %148#2, %137, %210, %457, %98, %296#1, %145, %396, %297, %160, %203, %63, %418, %arg38, %87, %62, %177, %192, %237, %323#1, %339, %303, %351, %arg20, %arg62, %arg83, %86, %100, %476, %248, %358, %73, %136, %205, %456, %155, %246, %266, %284, %222#1, %29, %224, %27#2, %249#2, %arg44, %161, %413, %76, %74#1, %80, %89, %58, %323#2, %399, %174, %347, %223, %250, %273, %arg68, %arg26, %270, %421, %44, %340, %51, %344, %234, %277, %101#1, %425, %43, %206, %445, %arg50, %235, %382, %383, %446, %150, %433, %81, %427, %90, %221, %428, %471#2, %arg32, %arg74, %102, %57, %238, %414, %451, %432, %148#1, %394, %48, %103, %154, %125, %101#2, %191, %199, %249#1, %309, %311, %443, %131, %25, %229, %74#2, %353, %122, %149, %398, %312, %472, %308, %473, %354, %397#2, %444#1, %477, %285, %172, %251, %376, %372, %219, %397#1, %370#2, %164, %28, %arg2, %302, %386, %385, %377, %147, %322, %293, %371, %298, %359, %132, %320, %27#1, %71, %280, %370#1, %26, %117, %279, %75, %arg8, %460, %129, %176, %369, %459, %295, %265, %325, %468, %17, %175#1, %296#2, %367, %228, %470, %478#1, %324, %450, %163, %175#2, %118, %222#2) : (tensor<64x256x65xf32>, tensor<f32>, tensor<16384x65xf32>, tensor<1x1x256x256xi1>, tensor<64x6x256x256xf32>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<384x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<384x1536xf32>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<16384x384xf32>, tensor<1x1x256x256xi1>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<384x384xf32>, tensor<16384x384xf32>, tensor<384x64x256xf32>, tensor<1536x384xf32>, tensor<64x256x1xf32>, tensor<384x256x64xf32>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x256xi64>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<16384xi64>, tensor<64x256x384xf32>, tensor<384x384xf32>, tensor<64x256x384xf32>, tensor<384x384xf32>, tensor<384x256x256xf32>, tensor<64x256x1536xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<384x64x256xf32>, tensor<384x384xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<384x256x64xf32>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1536xf32>, tensor<1536x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<64x256x384xf32>, tensor<64x6x256x256xf32>, tensor<384x1536xf32>, tensor<384x1152xf32>, tensor<64x6x256x256xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1x1x256x256xi1>, tensor<64x6x256x256xf32>, tensor<384x64x256xf32>, tensor<384x64x256xf32>, tensor<64x6x256x256xf32>, tensor<1x1x256x256xi1>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1xf32>, tensor<64x6x256x256xf32>, tensor<384x256x64xf32>, tensor<384x256x64xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<64x256x1536xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<64x256x1536xf32>, tensor<384x256x256xf32>, tensor<16384x1536xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1152xf32>, tensor<384x256x256xf32>, tensor<16384x1536xf32>, tensor<384x64x256xf32>, tensor<64x256x1536xf32>, tensor<384x384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<1x1x256x256xi1>, tensor<16384x384xf32>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1xf32>, tensor<384x256x64xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1xf32>, tensor<64x256x1536xf32>, tensor<1536x384xf32>, tensor<64x256x384xf32>, tensor<384x256x256xf32>, tensor<64x256x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x1xf32>, tensor<384x256x256xf32>, tensor<1x1x256x256xi1>, tensor<384x1536xf32>, tensor<384x1152xf32>, tensor<16384x1536xf32>, tensor<384x65xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<16384x65xf32>, tensor<16384x384xf32>, tensor<64x256x384xf32>, tensor<16384x384xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<16384x1536xf32>, tensor<384x1152xf32>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<16384x1536xf32>, tensor<1536x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<384x1536xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<384x256x64xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<384x256x256xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<16384x1536xf32>, tensor<64x6x256x256xf32>, tensor<384x1152xf32>, tensor<64x256x384xf32>, tensor<1536x384xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<16384x384xf32>, tensor<64x256x384xf32>, tensor<1x256xi64>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<f32>, tensor<384x1152xf32>, tensor<64x256x1536xf32>, tensor<1536x384xf32>, tensor<64x256x1xf32>, tensor<384x64x256xf32>, tensor<64x256x1xf32>) -> ()
  }) {function_type = (tensor<65x384xf32>, tensor<256x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<65x384xf32>, tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>, tensor<64x256xi64>, tensor<64x256xi64>) -> (tensor<64x256x65xf32>, tensor<f32>, tensor<16384x65xf32>, tensor<1x1x256x256xi1>, tensor<64x6x256x256xf32>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<384x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<384x1536xf32>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<16384x384xf32>, tensor<1x1x256x256xi1>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<384x384xf32>, tensor<16384x384xf32>, tensor<384x64x256xf32>, tensor<1536x384xf32>, tensor<64x256x1xf32>, tensor<384x256x64xf32>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x256xi64>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<16384xi64>, tensor<64x256x384xf32>, tensor<384x384xf32>, tensor<64x256x384xf32>, tensor<384x384xf32>, tensor<384x256x256xf32>, tensor<64x256x1536xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<384x64x256xf32>, tensor<384x384xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<384x256x64xf32>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1536xf32>, tensor<1536x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<64x256x384xf32>, tensor<64x6x256x256xf32>, tensor<384x1536xf32>, tensor<384x1152xf32>, tensor<64x6x256x256xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1x1x256x256xi1>, tensor<64x6x256x256xf32>, tensor<384x64x256xf32>, tensor<384x64x256xf32>, tensor<64x6x256x256xf32>, tensor<1x1x256x256xi1>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1xf32>, tensor<64x6x256x256xf32>, tensor<384x256x64xf32>, tensor<384x256x64xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<64x256x1536xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<64x256x1536xf32>, tensor<384x256x256xf32>, tensor<16384x1536xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1152xf32>, tensor<384x256x256xf32>, tensor<16384x1536xf32>, tensor<384x64x256xf32>, tensor<64x256x1536xf32>, tensor<384x384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<1x1x256x256xi1>, tensor<16384x384xf32>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1xf32>, tensor<384x256x64xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1xf32>, tensor<64x256x1536xf32>, tensor<1536x384xf32>, tensor<64x256x384xf32>, tensor<384x256x256xf32>, tensor<64x256x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x1xf32>, tensor<384x256x256xf32>, tensor<1x1x256x256xi1>, tensor<384x1536xf32>, tensor<384x1152xf32>, tensor<16384x1536xf32>, tensor<384x65xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<16384x65xf32>, tensor<16384x384xf32>, tensor<64x256x384xf32>, tensor<16384x384xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<16384x1536xf32>, tensor<384x1152xf32>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<16384x1536xf32>, tensor<1536x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<384x1536xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<384x256x64xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<384x256x256xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<16384x1536xf32>, tensor<64x6x256x256xf32>, tensor<384x1152xf32>, tensor<64x256x384xf32>, tensor<1536x384xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<16384x384xf32>, tensor<64x256x384xf32>, tensor<1x256xi64>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<f32>, tensor<384x1152xf32>, tensor<64x256x1536xf32>, tensor<1536x384xf32>, tensor<64x256x1xf32>, tensor<384x64x256xf32>, tensor<64x256x1xf32>), sym_name = "forward"} : () -> ()
}) {torch.debug_module_name = "GraphModule"} : () -> ()

