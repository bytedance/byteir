// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
!tuple = tuple<tensor<2x128x30522xf32>, tensor<f32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>>
module  {
  func.func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<2x128xi64>, %arg2: tensor<1x512xi64>, %arg3: tensor<1x512xi64>, %arg4: tensor<30522x128xf32>, %arg5: tensor<2x128xf32>, %arg6: tensor<512x128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128x128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<512x128xf32>, %arg20: tensor<512xf32>, %arg21: tensor<128x512xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128x128xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<512x128xf32>, %arg36: tensor<512xf32>, %arg37: tensor<128x512xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128x128xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<30522xf32>) -> !tuple {
    %0 = call @aten.view.109(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %1 = call @aten.index_select.129(%arg4, %0) : (tensor<30522x128xf32>, tensor<256xi64>) -> tensor<256x128xf32>
    %2 = call @aten.view.119(%1) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %4 = "mhlo.slice"(%3) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %5 = call @aten.expand.103(%4) : (tensor<1x128xi64>) -> tensor<2x128xi64>
    %6 = call @aten.view.109(%5) : (tensor<2x128xi64>) -> tensor<256xi64>
    %7 = call @aten.index_select.113(%arg5, %6) : (tensor<2x128xf32>, tensor<256xi64>) -> tensor<256x128xf32>
    %8 = call @aten.view.119(%7) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %9 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = call @aten.expand.94(%9) : (tensor<f32>) -> tensor<2x128x128xf32>
    %11 = call @aten.mul.123(%8, %10) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %12 = call @aten.add.136(%2, %11) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %13 = "mhlo.slice"(%arg3) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %14 = "mhlo.slice"(%13) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %15 = call @aten.view.74(%14) : (tensor<1x128xi64>) -> tensor<128xi64>
    %16 = call @aten.index_select.78(%arg6, %15) : (tensor<512x128xf32>, tensor<128xi64>) -> tensor<128x128xf32>
    %17 = call @aten.view.84(%16) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %18 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %19 = call @aten.expand.65(%18) : (tensor<f32>) -> tensor<1x128x128xf32>
    %20 = call @aten.mul.88(%17, %19) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %21 = call @aten.add.141(%12, %20) : (tensor<2x128x128xf32>, tensor<1x128x128xf32>) -> tensor<2x128x128xf32>
    %22 = "mhlo.custom_call"(%21, %arg7, %arg8) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %23 = "mhlo.get_tuple_element"(%22) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %24 = "mhlo.custom_call"(%23, %arg9, %arg10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %25 = "mhlo.custom_call"(%23, %arg11, %arg12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %26 = "mhlo.custom_call"(%24, %25) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %27 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = call @aten.expand.174(%27) : (tensor<f32>) -> tensor<2x1x1x128xf32>
    %29 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %30 = call @aten.expand.174(%29) : (tensor<f32>) -> tensor<2x1x1x128xf32>
    %31 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %32 = call @aten.expand.156(%31) : (tensor<f32>) -> tensor<2x128xf32>
    %33 = "mhlo.slice"(%32) {limit_indices = dense<[2, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %34 = call @aten.view.164(%33) : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %35 = call @aten.view.168(%34) : (tensor<2x1x128xf32>) -> tensor<2x1x1x128xf32>
    %36 = "mhlo.slice"(%35) {limit_indices = dense<[2, 1, 1, 128]> : tensor<4xi64>, start_indices = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32>
    %37 = call @aten.mul.181(%30, %36) : (tensor<2x1x1x128xf32>, tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32>
    %38 = call @aten.sub.188(%28, %37) : (tensor<2x1x1x128xf32>, tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32>
    %39 = mhlo.constant dense<-1.000000e+04> : tensor<f32>
    %40 = call @aten.mul.193(%38, %39) : (tensor<2x1x1x128xf32>, tensor<f32>) -> tensor<2x1x1x128xf32>
    %41 = "mhlo.slice"(%40) {limit_indices = dense<[2, 1, 1, 128]> : tensor<4xi64>, start_indices = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32>
    %42 = call @aten.view.200(%41) : (tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32>
    %43 = call @aten.view.203(%42) : (tensor<2x1x1x128xf32>) -> tensor<2x1x128xf32>
    %44 = "mhlo.slice"(%43) {limit_indices = dense<[2, 1, 128]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x1x128xf32>) -> tensor<2x1x128xf32>
    %45 = call @aten.expand.208(%44) : (tensor<2x1x128xf32>) -> tensor<2x128x128xf32>
    %46 = "mhlo.custom_call"(%26, %45) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %47 = "mhlo.get_tuple_element"(%46) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %48 = "mhlo.get_tuple_element"(%46) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %49 = "mhlo.custom_call"(%23, %arg13, %arg14) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %50 = "mhlo.custom_call"(%48, %49) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %51 = "mhlo.custom_call"(%50) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %52 = call @aten.view.223(%51) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %53 = "mhlo.custom_call"(%52, %arg15, %arg16) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %54 = "mhlo.custom_call"(%53, %arg17, %arg18, %23) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %55 = "mhlo.get_tuple_element"(%54) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %56 = "mhlo.custom_call"(%55, %arg19, %arg20) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %57 = "mhlo.get_tuple_element"(%56) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %58 = "mhlo.custom_call"(%57, %arg21, %arg22) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %59 = "mhlo.custom_call"(%58, %arg23, %arg24, %55) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %60 = "mhlo.get_tuple_element"(%59) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %61 = "mhlo.custom_call"(%60, %arg25, %arg26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %62 = "mhlo.custom_call"(%60, %arg27, %arg28) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %63 = "mhlo.custom_call"(%61, %62) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %64 = "mhlo.slice"(%40) {limit_indices = dense<[2, 1, 1, 128]> : tensor<4xi64>, start_indices = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32>
    %65 = call @aten.view.200(%64) : (tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32>
    %66 = call @aten.view.203(%65) : (tensor<2x1x1x128xf32>) -> tensor<2x1x128xf32>
    %67 = "mhlo.slice"(%66) {limit_indices = dense<[2, 1, 128]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x1x128xf32>) -> tensor<2x1x128xf32>
    %68 = call @aten.expand.208(%67) : (tensor<2x1x128xf32>) -> tensor<2x128x128xf32>
    %69 = "mhlo.custom_call"(%63, %68) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %70 = "mhlo.get_tuple_element"(%69) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %71 = "mhlo.get_tuple_element"(%69) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %72 = "mhlo.custom_call"(%60, %arg29, %arg30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %73 = "mhlo.custom_call"(%71, %72) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %74 = "mhlo.custom_call"(%73) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %75 = call @aten.view.223(%74) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %76 = "mhlo.custom_call"(%75, %arg31, %arg32) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %77 = "mhlo.custom_call"(%76, %arg33, %arg34, %60) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %78 = "mhlo.get_tuple_element"(%77) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %79 = "mhlo.custom_call"(%78, %arg35, %arg36) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %80 = "mhlo.get_tuple_element"(%79) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %81 = "mhlo.custom_call"(%80, %arg37, %arg38) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %82 = "mhlo.custom_call"(%81, %arg39, %arg40, %78) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %83 = "mhlo.get_tuple_element"(%82) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %84 = "mhlo.custom_call"(%83, %arg41, %arg42) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>
    %85 = "mhlo.get_tuple_element"(%84) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %86 = "mhlo.custom_call"(%85, %arg43, %arg44) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %87 = "mhlo.get_tuple_element"(%86) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %88 = call @aten.view.283(%87) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %89 = call @aten.permute.60(%arg4) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %90 = call @aten.mm.287(%88, %89) : (tensor<256x128xf32>, tensor<128x30522xf32>) -> tensor<256x30522xf32>
    %91 = call @aten.view.292(%90) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %92 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %93 = call @aten.expand.48(%92) : (tensor<f32>) -> tensor<30522xf32>
    %94 = call @aten.mul.55(%arg45, %93) : (tensor<30522xf32>, tensor<30522xf32>) -> tensor<30522xf32>
    %95 = call @aten.add.296(%91, %94) : (tensor<2x128x30522xf32>, tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %96 = call @aten.view.302(%95) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %97 = call @aten.view.292(%96) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %98 = call @aten.view.292(%96) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %99 = call @aten.view.302(%98) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %100 = call @aten.log_softmax.318(%99) : (tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %101 = call @aten.view.109(%arg1) : (tensor<2x128xi64>) -> tensor<256xi64>
    %102 = call @aten.nll_loss.335(%100, %101) : (tensor<256x30522xf32>, tensor<256xi64>) -> tensor<f32>
    %103 = call @aten.view.283(%87) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %104 = call @aten.permute.667(%103) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %105 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %106 = call @aten.nll_loss_backward.404(%105, %100, %101) : (tensor<f32>, tensor<256x30522xf32>, tensor<256xi64>) -> tensor<256x30522xf32>
    %107 = call @aten._log_softmax_backward_data.444(%106, %100) : (tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %108 = call @aten.view.292(%107) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %109 = call @aten.view.302(%108) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %110 = call @aten.mm.671(%104, %109) : (tensor<128x256xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    %111 = call @aten.permute.676(%110) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %112 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %113 = call @aten.expand.371(%112) : (tensor<f32>) -> tensor<30522x128xf32>
    %114 = call @aten.view.109(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %115 = mhlo.constant dense<0> : tensor<i64>
    %116 = call @aten.lt.627(%114, %115) : (tensor<256xi64>, tensor<i64>) -> tensor<256xi1>
    %117 = mhlo.constant dense<30522> : tensor<i64>
    %118 = call @aten.expand.614(%117) : (tensor<i64>) -> tensor<256xi64>
    %119 = call @aten.add.621(%114, %118) : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %120 = call @aten.where.633(%116, %119, %114) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %121 = call @aten.stack.639(%120) : (tensor<256xi64>) -> tensor<256x1xi64>
    %122 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %123 = call @aten.ne.590(%114, %122) : (tensor<256xi64>, tensor<f64>) -> tensor<256xi1>
    %124 = call @aten.view.597(%123) : (tensor<256xi1>) -> tensor<256x1xi1>
    %125 = call @aten.expand.601(%124) : (tensor<256x1xi1>) -> tensor<256x128xi1>
    %126 = call @aten.permute.60(%arg4) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %127 = call @aten.permute.395(%126) : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %128 = call @aten.mm.456(%109, %127) : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    %129 = call @aten.view.119(%128) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %130 = "mhlo.get_tuple_element"(%86) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %131 = "mhlo.get_tuple_element"(%86) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %132 = "mhlo.custom_call"(%129, %85, %arg43, %130, %131) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %133 = "mhlo.get_tuple_element"(%132) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %134 = "mhlo.get_tuple_element"(%84) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %135 = "mhlo.get_tuple_element"(%84) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %136 = "mhlo.custom_call"(%133, %83, %arg41, %134, %135) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %137 = "mhlo.get_tuple_element"(%136) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %138 = "mhlo.get_tuple_element"(%82) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %139 = "mhlo.get_tuple_element"(%82) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %140 = "mhlo.get_tuple_element"(%82) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %141 = "mhlo.custom_call"(%137, %138, %arg39, %139, %140) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %142 = "mhlo.get_tuple_element"(%141) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %143 = "mhlo.get_tuple_element"(%141) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %144 = "mhlo.custom_call"(%143, %80, %arg37) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %145 = "mhlo.get_tuple_element"(%144) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %146 = "mhlo.get_tuple_element"(%79) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %147 = "mhlo.get_tuple_element"(%79) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %148 = "mhlo.custom_call"(%145, %78, %arg35, %146, %147) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %149 = "mhlo.get_tuple_element"(%148) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %150 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %151 = call @aten.expand.94(%150) : (tensor<f32>) -> tensor<2x128x128xf32>
    %152 = call @aten.mul.123(%149, %151) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %153 = call @aten.add.136(%142, %152) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %154 = "mhlo.get_tuple_element"(%77) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %155 = "mhlo.get_tuple_element"(%77) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %156 = "mhlo.get_tuple_element"(%77) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %157 = "mhlo.custom_call"(%153, %154, %arg33, %155, %156) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %158 = "mhlo.get_tuple_element"(%157) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %159 = "mhlo.get_tuple_element"(%157) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %160 = "mhlo.custom_call"(%159, %75, %arg31) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %161 = "mhlo.get_tuple_element"(%160) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %162 = call @aten.view.494(%161) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %163 = "mhlo.custom_call"(%162) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %164 = "mhlo.custom_call"(%163, %71, %72) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %165 = "mhlo.get_tuple_element"(%164) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %166 = "mhlo.get_tuple_element"(%69) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %167 = "mhlo.custom_call"(%165, %71, %166) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %168 = "mhlo.custom_call"(%167, %61, %62) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %169 = "mhlo.get_tuple_element"(%168) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %170 = "mhlo.custom_call"(%169, %60, %arg25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %171 = "mhlo.get_tuple_element"(%170) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %172 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %173 = call @aten.expand.94(%172) : (tensor<f32>) -> tensor<2x128x128xf32>
    %174 = call @aten.mul.123(%171, %173) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %175 = call @aten.add.136(%158, %174) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %176 = "mhlo.get_tuple_element"(%164) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %177 = "mhlo.custom_call"(%176, %60, %arg29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %178 = "mhlo.get_tuple_element"(%177) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %179 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %180 = call @aten.expand.94(%179) : (tensor<f32>) -> tensor<2x128x128xf32>
    %181 = call @aten.mul.123(%178, %180) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %182 = call @aten.add.136(%175, %181) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %183 = "mhlo.get_tuple_element"(%168) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %184 = "mhlo.custom_call"(%183, %60, %arg27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %185 = "mhlo.get_tuple_element"(%184) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %186 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %187 = call @aten.expand.94(%186) : (tensor<f32>) -> tensor<2x128x128xf32>
    %188 = call @aten.mul.123(%185, %187) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %189 = call @aten.add.136(%182, %188) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %190 = "mhlo.get_tuple_element"(%59) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %191 = "mhlo.get_tuple_element"(%59) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %192 = "mhlo.get_tuple_element"(%59) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %193 = "mhlo.custom_call"(%189, %190, %arg23, %191, %192) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %194 = "mhlo.get_tuple_element"(%193) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %195 = "mhlo.get_tuple_element"(%193) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %196 = "mhlo.custom_call"(%195, %57, %arg21) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %197 = "mhlo.get_tuple_element"(%196) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %198 = "mhlo.get_tuple_element"(%56) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %199 = "mhlo.get_tuple_element"(%56) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %200 = "mhlo.custom_call"(%197, %55, %arg19, %198, %199) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %201 = "mhlo.get_tuple_element"(%200) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %202 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %203 = call @aten.expand.94(%202) : (tensor<f32>) -> tensor<2x128x128xf32>
    %204 = call @aten.mul.123(%201, %203) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %205 = call @aten.add.136(%194, %204) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %206 = "mhlo.get_tuple_element"(%54) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %207 = "mhlo.get_tuple_element"(%54) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %208 = "mhlo.get_tuple_element"(%54) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %209 = "mhlo.custom_call"(%205, %206, %arg17, %207, %208) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %210 = "mhlo.get_tuple_element"(%209) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %211 = "mhlo.get_tuple_element"(%209) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %212 = "mhlo.custom_call"(%211, %52, %arg15) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %213 = "mhlo.get_tuple_element"(%212) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %214 = call @aten.view.494(%213) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %215 = "mhlo.custom_call"(%214) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %216 = "mhlo.custom_call"(%215, %48, %49) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %217 = "mhlo.get_tuple_element"(%216) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %218 = "mhlo.get_tuple_element"(%46) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %219 = "mhlo.custom_call"(%217, %48, %218) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %220 = "mhlo.custom_call"(%219, %24, %25) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %221 = "mhlo.get_tuple_element"(%220) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %222 = "mhlo.custom_call"(%221, %23, %arg9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %223 = "mhlo.get_tuple_element"(%222) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %224 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %225 = call @aten.expand.94(%224) : (tensor<f32>) -> tensor<2x128x128xf32>
    %226 = call @aten.mul.123(%223, %225) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %227 = call @aten.add.136(%210, %226) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %228 = "mhlo.get_tuple_element"(%216) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %229 = "mhlo.custom_call"(%228, %23, %arg13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %230 = "mhlo.get_tuple_element"(%229) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %231 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %232 = call @aten.expand.94(%231) : (tensor<f32>) -> tensor<2x128x128xf32>
    %233 = call @aten.mul.123(%230, %232) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %234 = call @aten.add.136(%227, %233) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %235 = "mhlo.get_tuple_element"(%220) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %236 = "mhlo.custom_call"(%235, %23, %arg11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %237 = "mhlo.get_tuple_element"(%236) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %238 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %239 = call @aten.expand.94(%238) : (tensor<f32>) -> tensor<2x128x128xf32>
    %240 = call @aten.mul.123(%237, %239) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %241 = call @aten.add.136(%234, %240) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %242 = "mhlo.get_tuple_element"(%22) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %243 = "mhlo.get_tuple_element"(%22) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %244 = "mhlo.custom_call"(%241, %21, %arg7, %242, %243) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %245 = "mhlo.get_tuple_element"(%244) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %246 = call @aten.view.283(%245) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %247 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %248 = call @aten.expand.379(%247) : (tensor<f32>) -> tensor<256x128xf32>
    %249 = call @aten.where.607(%125, %246, %248) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %250 = call @aten.index_put.650(%113, %121, %249) : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %251 = call @aten.permute.657(%250) : (tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %252 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %253 = call @aten.expand.371(%252) : (tensor<f32>) -> tensor<30522x128xf32>
    %254 = call @aten.mul.661(%251, %253) : (tensor<30522x128xf32>, tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %255 = call @aten.add.680(%111, %254) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<30522x128xf32>, tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %256 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %257 = call @aten.expand.156(%256) : (tensor<f32>) -> tensor<2x128xf32>
    %258 = call @aten.view.109(%5) : (tensor<2x128xi64>) -> tensor<256xi64>
    %259 = mhlo.constant dense<0> : tensor<i64>
    %260 = call @aten.lt.627(%258, %259) : (tensor<256xi64>, tensor<i64>) -> tensor<256xi1>
    %261 = mhlo.constant dense<2> : tensor<i64>
    %262 = call @aten.expand.614(%261) : (tensor<i64>) -> tensor<256xi64>
    %263 = call @aten.add.621(%258, %262) : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %264 = call @aten.where.633(%260, %263, %258) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %265 = call @aten.stack.639(%264) : (tensor<256xi64>) -> tensor<256x1xi64>
    %266 = mhlo.constant dense<-1.000000e+00> : tensor<f64>
    %267 = call @aten.ne.590(%258, %266) : (tensor<256xi64>, tensor<f64>) -> tensor<256xi1>
    %268 = call @aten.view.597(%267) : (tensor<256xi1>) -> tensor<256x1xi1>
    %269 = call @aten.expand.601(%268) : (tensor<256x1xi1>) -> tensor<256x128xi1>
    %270 = call @aten.view.283(%245) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %271 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %272 = call @aten.expand.379(%271) : (tensor<f32>) -> tensor<256x128xf32>
    %273 = call @aten.where.607(%269, %270, %272) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %274 = call @aten.index_put.707(%257, %265, %273) : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %275 = call @aten.permute.714(%274) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %276 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %277 = call @aten.expand.800(%276) : (tensor<f32>) -> tensor<512x128xf32>
    %278 = "mhlo.slice"(%arg3) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %279 = "mhlo.slice"(%278) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %280 = call @aten.view.74(%279) : (tensor<1x128xi64>) -> tensor<128xi64>
    %281 = mhlo.constant dense<0> : tensor<i64>
    %282 = call @aten.lt.782(%280, %281) : (tensor<128xi64>, tensor<i64>) -> tensor<128xi1>
    %283 = mhlo.constant dense<512> : tensor<i64>
    %284 = call @aten.expand.769(%283) : (tensor<i64>) -> tensor<128xi64>
    %285 = call @aten.add.776(%280, %284) : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %286 = call @aten.where.788(%282, %285, %280) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %287 = call @aten.stack.794(%286) : (tensor<128xi64>) -> tensor<128x1xi64>
    %288 = mhlo.constant dense<-1.000000e+00> : tensor<f64>
    %289 = call @aten.ne.745(%280, %288) : (tensor<128xi64>, tensor<f64>) -> tensor<128xi1>
    %290 = call @aten.view.752(%289) : (tensor<128xi1>) -> tensor<128x1xi1>
    %291 = call @aten.expand.756(%290) : (tensor<128x1xi1>) -> tensor<128x128xi1>
    %292 = call @aten.sum.730(%245) : (tensor<2x128x128xf32>) -> tensor<1x128x128xf32>
    %293 = call @aten.view.737(%292) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %294 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %295 = call @aten.expand.719(%294) : (tensor<f32>) -> tensor<128x128xf32>
    %296 = call @aten.where.762(%291, %293, %295) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %297 = call @aten.index_put.811(%277, %287, %296) : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %298 = call @aten.permute.818(%297) : (tensor<512x128xf32>) -> tensor<512x128xf32>
    %299 = "mhlo.get_tuple_element"(%244) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %300 = "mhlo.get_tuple_element"(%244) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %301 = "mhlo.get_tuple_element"(%222) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %302 = "mhlo.get_tuple_element"(%222) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %303 = "mhlo.get_tuple_element"(%236) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %304 = "mhlo.get_tuple_element"(%236) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %305 = "mhlo.get_tuple_element"(%229) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %306 = "mhlo.get_tuple_element"(%229) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %307 = "mhlo.get_tuple_element"(%212) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %308 = "mhlo.get_tuple_element"(%212) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %309 = "mhlo.get_tuple_element"(%209) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %310 = "mhlo.get_tuple_element"(%209) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %311 = "mhlo.get_tuple_element"(%200) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %312 = "mhlo.get_tuple_element"(%200) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %313 = "mhlo.get_tuple_element"(%196) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %314 = "mhlo.get_tuple_element"(%196) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %315 = "mhlo.get_tuple_element"(%193) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %316 = "mhlo.get_tuple_element"(%193) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %317 = "mhlo.get_tuple_element"(%170) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %318 = "mhlo.get_tuple_element"(%170) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %319 = "mhlo.get_tuple_element"(%184) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %320 = "mhlo.get_tuple_element"(%184) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %321 = "mhlo.get_tuple_element"(%177) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %322 = "mhlo.get_tuple_element"(%177) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %323 = "mhlo.get_tuple_element"(%160) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %324 = "mhlo.get_tuple_element"(%160) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %325 = "mhlo.get_tuple_element"(%157) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %326 = "mhlo.get_tuple_element"(%157) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %327 = "mhlo.get_tuple_element"(%148) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %328 = "mhlo.get_tuple_element"(%148) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %329 = "mhlo.get_tuple_element"(%144) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %330 = "mhlo.get_tuple_element"(%144) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %331 = "mhlo.get_tuple_element"(%141) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %332 = "mhlo.get_tuple_element"(%141) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %333 = "mhlo.get_tuple_element"(%136) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %334 = "mhlo.get_tuple_element"(%136) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %335 = "mhlo.get_tuple_element"(%132) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %336 = "mhlo.get_tuple_element"(%132) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %337 = call @aten.view.292(%107) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %338 = call @aten.sum.827(%337) : (tensor<2x128x30522xf32>) -> tensor<1x1x30522xf32>
    %339 = call @aten.view.834(%338) : (tensor<1x1x30522xf32>) -> tensor<30522xf32>
    %340 = "mhlo.tuple"(%97, %102, %255, %275, %298, %299, %300, %301, %302, %303, %304, %305, %306, %307, %308, %309, %310, %311, %312, %313, %314, %315, %316, %317, %318, %319, %320, %321, %322, %323, %324, %325, %326, %327, %328, %329, %330, %331, %332, %333, %334, %335, %336, %339) : (tensor<2x128x30522xf32>, tensor<f32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>) -> !tuple
    return %340 : !tuple
  }
  func.func private @aten.view.109(%arg0: tensor<2x128xi64>) -> tensor<256xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func.func private @aten.index_select.129(%arg0: tensor<30522x128xf32>, %arg1: tensor<256xi64>) -> tensor<256x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<256xi64>) -> tensor<256xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
  func.func private @aten.view.119(%arg0: tensor<256x128xf32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func.func private @aten.expand.103(%arg0: tensor<1x128xi64>) -> tensor<2x128xi64> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128xi64>) -> tensor<1x128xi64>
    %1 = "mhlo.reshape"(%0) : (tensor<1x128xi64>) -> tensor<128xi64>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xi64>) -> tensor<2x128xi64>
    return %2 : tensor<2x128xi64>
  }
  func.func private @aten.index_select.113(%arg0: tensor<2x128xf32>, %arg1: tensor<256xi64>) -> tensor<256x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<256xi64>) -> tensor<256xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
  func.func private @aten.expand.94(%arg0: tensor<f32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128x128xf32>
    return %3 : tensor<2x128x128xf32>
  }
  func.func private @aten.mul.123(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func.func private @aten.add.136(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func.func private @aten.view.74(%arg0: tensor<1x128xi64>) -> tensor<128xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128xi64>) -> tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func.func private @aten.index_select.78(%arg0: tensor<512x128xf32>, %arg1: tensor<128xi64>) -> tensor<128x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<128xi64>) -> tensor<128xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func.func private @aten.view.84(%arg0: tensor<128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    return %0 : tensor<1x128x128xf32>
  }
  func.func private @aten.expand.65(%arg0: tensor<f32>) -> tensor<1x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x128x128xf32>
    return %3 : tensor<1x128x128xf32>
  }
  func.func private @aten.mul.88(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x128x128xf32>
    return %0 : tensor<1x128x128xf32>
  }
  func.func private @aten.add.141(%arg0: tensor<2x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<2x128x128xf32>
    %3 = mhlo.add %arg0, %2 : tensor<2x128x128xf32>
    return %3 : tensor<2x128x128xf32>
  }
  func.func private @aten.expand.174(%arg0: tensor<f32>) -> tensor<2x1x1x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<1x1xf32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<2x1x1x128xf32>
    return %3 : tensor<2x1x1x128xf32>
  }
  func.func private @aten.expand.156(%arg0: tensor<f32>) -> tensor<2x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128xf32>
    return %3 : tensor<2x128xf32>
  }
  func.func private @aten.view.164(%arg0: tensor<2x128xf32>) -> tensor<2x1x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    return %0 : tensor<2x1x128xf32>
  }
  func.func private @aten.view.168(%arg0: tensor<2x1x128xf32>) -> tensor<2x1x1x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x1x128xf32>) -> tensor<2x1x1x128xf32>
    return %0 : tensor<2x1x1x128xf32>
  }
  func.func private @aten.mul.181(%arg0: tensor<2x1x1x128xf32>, %arg1: tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x1x1x128xf32>
    return %0 : tensor<2x1x1x128xf32>
  }
  func.func private @aten.sub.188(%arg0: tensor<2x1x1x128xf32>, %arg1: tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32> {
    %0 = mhlo.subtract %arg0, %arg1 : tensor<2x1x1x128xf32>
    return %0 : tensor<2x1x1x128xf32>
  }
  func.func private @aten.mul.193(%arg0: tensor<2x1x1x128xf32>, %arg1: tensor<f32>) -> tensor<2x1x1x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x1x1x128xf32>
    %1 = mhlo.multiply %arg0, %0 : tensor<2x1x1x128xf32>
    return %1 : tensor<2x1x1x128xf32>
  }
  func.func private @aten.view.200(%arg0: tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32> {
    return %arg0 : tensor<2x1x1x128xf32>
  }
  func.func private @aten.view.203(%arg0: tensor<2x1x1x128xf32>) -> tensor<2x1x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x1x1x128xf32>) -> tensor<2x1x128xf32>
    return %0 : tensor<2x1x128xf32>
  }
  func.func private @aten.expand.208(%arg0: tensor<2x1x128xf32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x1x128xf32>) -> tensor<2x1x128xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<2x128xf32>) -> tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func.func private @aten.view.223(%arg0: tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func.func private @aten.view.283(%arg0: tensor<2x128x128xf32>) -> tensor<256x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
  func.func private @aten.permute.60(%arg0: tensor<30522x128xf32>) -> tensor<128x30522xf32> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    return %0 : tensor<128x30522xf32>
  }
  func.func private @aten.mm.287(%arg0: tensor<256x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<256x30522xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x128xf32>, tensor<128x30522xf32>) -> tensor<256x30522xf32>
    return %0 : tensor<256x30522xf32>
  }
  func.func private @aten.view.292(%arg0: tensor<256x30522xf32>) -> tensor<2x128x30522xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    return %0 : tensor<2x128x30522xf32>
  }
  func.func private @aten.expand.48(%arg0: tensor<f32>) -> tensor<30522xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<30522xf32>
    return %3 : tensor<30522xf32>
  }
  func.func private @aten.mul.55(%arg0: tensor<30522xf32>, %arg1: tensor<30522xf32>) -> tensor<30522xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<30522xf32>
    return %0 : tensor<30522xf32>
  }
  func.func private @aten.add.296(%arg0: tensor<2x128x30522xf32>, %arg1: tensor<30522xf32>) -> tensor<2x128x30522xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %1 = mhlo.add %arg0, %0 : tensor<2x128x30522xf32>
    return %1 : tensor<2x128x30522xf32>
  }
  func.func private @aten.view.302(%arg0: tensor<2x128x30522xf32>) -> tensor<256x30522xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    return %0 : tensor<256x30522xf32>
  }
  func.func private @aten.log_softmax.318(%arg0: tensor<256x30522xf32>) -> tensor<256x30522xf32> {
    %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %1 = "mhlo.reduce"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %10 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%10) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %3 = mhlo.subtract %arg0, %2 : tensor<256x30522xf32>
    %4 = "mhlo.exponential"(%3) : (tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = "mhlo.reduce"(%4, %5) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %10 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%10) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
    %7 = "mhlo.log"(%6) : (tensor<256xf32>) -> tensor<256xf32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %9 = mhlo.subtract %3, %8 : tensor<256x30522xf32>
    return %9 : tensor<256x30522xf32>
  }
  func.func private @aten.nll_loss.335(%arg0: tensor<256x30522xf32>, %arg1: tensor<256xi64>) -> tensor<f32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi64>) -> tensor<256x30522xi64>
    %1 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<1x30522xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1x30522xi64>) -> tensor<30522xi64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<30522xi64>) -> tensor<256x30522xi64>
    %4 = "mhlo.compare"(%0, %3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<256x30522xi64>, tensor<256x30522xi64>) -> tensor<256x30522xi1>
    %5 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %7 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %9 = "mhlo.select"(%4, %6, %8) : (tensor<256x30522xi1>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %10 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %11 = "mhlo.compare"(%9, %10) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xi1>
    %12 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %13 = "mhlo.negate"(%9) : (tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %14 = mhlo.multiply %13, %arg0 : tensor<256x30522xf32>
    %15 = "mhlo.select"(%11, %12, %14) : (tensor<256x30522xi1>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %16 = mhlo.constant dense<-100> : tensor<i64>
    %17 = "mhlo.broadcast_in_dim"(%16) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<256xi64>
    %18 = "mhlo.compare"(%arg1, %17) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %19 = "mhlo.broadcast_in_dim"(%18) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x30522xi1>
    %20 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %21 = "mhlo.broadcast_in_dim"(%20) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %22 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %23 = "mhlo.broadcast_in_dim"(%22) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %24 = "mhlo.select"(%19, %21, %23) : (tensor<256x30522xi1>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %25 = mhlo.multiply %24, %9 : tensor<256x30522xf32>
    %26 = mhlo.multiply %15, %25 : tensor<256x30522xf32>
    %27 = "mhlo.reduce"(%26, %7) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
      %31 = mhlo.add %arg2, %arg3 : tensor<f32>
      "mhlo.return"(%31) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x30522xf32>, tensor<f32>) -> tensor<f32>
    %28 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %29 = "mhlo.reduce"(%25, %28) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
      %31 = mhlo.add %arg2, %arg3 : tensor<f32>
      "mhlo.return"(%31) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x30522xf32>, tensor<f32>) -> tensor<f32>
    %30 = mhlo.divide %27, %29 : tensor<f32>
    return %30 : tensor<f32>
  }
  func.func private @aten.permute.667(%arg0: tensor<256x128xf32>) -> tensor<128x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
  }
  func.func private @aten.nll_loss_backward.404(%arg0: tensor<f32>, %arg1: tensor<256x30522xf32>, %arg2: tensor<256xi64>) -> tensor<256x30522xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi64>) -> tensor<256x30522xi64>
    %1 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<1x30522xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1x30522xi64>) -> tensor<30522xi64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<30522xi64>) -> tensor<256x30522xi64>
    %4 = "mhlo.compare"(%0, %3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<256x30522xi64>, tensor<256x30522xi64>) -> tensor<256x30522xi1>
    %5 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %7 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %9 = "mhlo.select"(%4, %6, %8) : (tensor<256x30522xi1>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %10 = "mhlo.negate"(%9) : (tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %11 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %12 = mhlo.multiply %10, %11 : tensor<256x30522xf32>
    %13 = mhlo.constant dense<-100> : tensor<i64>
    %14 = "mhlo.broadcast_in_dim"(%13) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<256xi64>
    %15 = "mhlo.compare"(%arg2, %14) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %16 = "mhlo.broadcast_in_dim"(%15) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x30522xi1>
    %17 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = "mhlo.broadcast_in_dim"(%17) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %19 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = "mhlo.broadcast_in_dim"(%19) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %21 = "mhlo.select"(%16, %18, %20) : (tensor<256x30522xi1>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %22 = mhlo.multiply %21, %9 : tensor<256x30522xf32>
    %23 = mhlo.multiply %12, %22 : tensor<256x30522xf32>
    %24 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = "mhlo.reduce"(%22, %24) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %31 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%31) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x30522xf32>, tensor<f32>) -> tensor<f32>
    %26 = "mhlo.compare"(%25, %24) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %27 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = "mhlo.select"(%26, %25, %27) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %29 = "mhlo.broadcast_in_dim"(%28) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %30 = mhlo.divide %23, %29 : tensor<256x30522xf32>
    return %30 : tensor<256x30522xf32>
  }
  func.func private @aten._log_softmax_backward_data.444(%arg0: tensor<256x30522xf32>, %arg1: tensor<256x30522xf32>) -> tensor<256x30522xf32> {
    %0 = "mhlo.exponential"(%arg1) : (tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = "mhlo.reduce"(%arg0, %1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
      %6 = mhlo.add %arg2, %arg3 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %4 = mhlo.multiply %0, %3 : tensor<256x30522xf32>
    %5 = mhlo.subtract %arg0, %4 : tensor<256x30522xf32>
    return %5 : tensor<256x30522xf32>
  }
  func.func private @aten.mm.671(%arg0: tensor<128x256xf32>, %arg1: tensor<256x30522xf32>) -> tensor<128x30522xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<128x256xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    return %0 : tensor<128x30522xf32>
  }
  func.func private @aten.permute.676(%arg0: tensor<128x30522xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func.func private @aten.expand.371(%arg0: tensor<f32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<30522x128xf32>
    return %3 : tensor<30522x128xf32>
  }
  func.func private @aten.lt.627(%arg0: tensor<256xi64>, %arg1: tensor<i64>) -> tensor<256xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<256xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    return %1 : tensor<256xi1>
  }
  func.func private @aten.expand.614(%arg0: tensor<i64>) -> tensor<256xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<i64>) -> tensor<1xi64>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi64>) -> tensor<i64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<256xi64>
    return %3 : tensor<256xi64>
  }
  func.func private @aten.add.621(%arg0: tensor<256xi64>, %arg1: tensor<256xi64>) -> tensor<256xi64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func.func private @aten.where.633(%arg0: tensor<256xi1>, %arg1: tensor<256xi64>, %arg2: tensor<256xi64>) -> tensor<256xi64> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func.func private @aten.stack.639(%arg0: tensor<256xi64>) -> tensor<256x1xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256xi64>) -> tensor<256x1xi64>
    %1 = "mhlo.concatenate"(%0) {dimension = 1 : i64} : (tensor<256x1xi64>) -> tensor<256x1xi64>
    return %1 : tensor<256x1xi64>
  }
  func.func private @aten.ne.590(%arg0: tensor<256xi64>, %arg1: tensor<f64>) -> tensor<256xi1> {
    %0 = "mhlo.convert"(%arg0) : (tensor<256xi64>) -> tensor<256xf64>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<256xf64>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %2 : tensor<256xi1>
  }
  func.func private @aten.view.597(%arg0: tensor<256xi1>) -> tensor<256x1xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256xi1>) -> tensor<256x1xi1>
    return %0 : tensor<256x1xi1>
  }
  func.func private @aten.expand.601(%arg0: tensor<256x1xi1>) -> tensor<256x128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x1xi1>) -> tensor<256x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<256x1xi1>) -> tensor<256xi1>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    return %2 : tensor<256x128xi1>
  }
  func.func private @aten.permute.395(%arg0: tensor<128x30522xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func.func private @aten.mm.456(%arg0: tensor<256x30522xf32>, %arg1: tensor<30522x128xf32>) -> tensor<256x128xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
  func.func private @aten.view.494(%arg0: tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    return %0 : tensor<2x128x2x64xf32>
  }
  func.func private @aten.expand.379(%arg0: tensor<f32>) -> tensor<256x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x128xf32>
    return %3 : tensor<256x128xf32>
  }
  func.func private @aten.where.607(%arg0: tensor<256x128xi1>, %arg1: tensor<256x128xf32>, %arg2: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
  func.func private @aten.index_put.650(%arg0: tensor<30522x128xf32>, %arg1: tensor<256x1xi64>, %arg2: tensor<256x128xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func.func private @aten.permute.657(%arg0: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<30522x128xf32>) -> tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func.func private @aten.mul.661(%arg0: tensor<30522x128xf32>, %arg1: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func.func private @aten.add.680(%arg0: tensor<30522x128xf32>, %arg1: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.add %arg0, %arg1 {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func.func private @aten.index_put.707(%arg0: tensor<2x128xf32>, %arg1: tensor<256x1xi64>, %arg2: tensor<256x128xf32>) -> tensor<2x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    return %1 : tensor<2x128xf32>
  }
  func.func private @aten.permute.714(%arg0: tensor<2x128xf32>) -> tensor<2x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128xf32>) -> tensor<2x128xf32>
    return %0 : tensor<2x128xf32>
  }
  func.func private @aten.expand.800(%arg0: tensor<f32>) -> tensor<512x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512x128xf32>
    return %3 : tensor<512x128xf32>
  }
  func.func private @aten.lt.782(%arg0: tensor<128xi64>, %arg1: tensor<i64>) -> tensor<128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<128xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    return %1 : tensor<128xi1>
  }
  func.func private @aten.expand.769(%arg0: tensor<i64>) -> tensor<128xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<i64>) -> tensor<1xi64>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi64>) -> tensor<i64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<128xi64>
    return %3 : tensor<128xi64>
  }
  func.func private @aten.add.776(%arg0: tensor<128xi64>, %arg1: tensor<128xi64>) -> tensor<128xi64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func.func private @aten.where.788(%arg0: tensor<128xi1>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>) -> tensor<128xi64> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func.func private @aten.stack.794(%arg0: tensor<128xi64>) -> tensor<128x1xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128xi64>) -> tensor<128x1xi64>
    %1 = "mhlo.concatenate"(%0) {dimension = 1 : i64} : (tensor<128x1xi64>) -> tensor<128x1xi64>
    return %1 : tensor<128x1xi64>
  }
  func.func private @aten.ne.745(%arg0: tensor<128xi64>, %arg1: tensor<f64>) -> tensor<128xi1> {
    %0 = "mhlo.convert"(%arg0) : (tensor<128xi64>) -> tensor<128xf64>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<128xf64>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %2 : tensor<128xi1>
  }
  func.func private @aten.view.752(%arg0: tensor<128xi1>) -> tensor<128x1xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128xi1>) -> tensor<128x1xi1>
    return %0 : tensor<128x1xi1>
  }
  func.func private @aten.expand.756(%arg0: tensor<128x1xi1>) -> tensor<128x128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<128x1xi1>) -> tensor<128x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<128x1xi1>) -> tensor<128xi1>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    return %2 : tensor<128x128xi1>
  }
  func.func private @aten.sum.730(%arg0: tensor<2x128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = mhlo.constant dense<2> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = "mhlo.reduce"(%arg0, %1) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    return %3 : tensor<1x128x128xf32>
  }
  func.func private @aten.view.737(%arg0: tensor<1x128x128xf32>) -> tensor<128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
  func.func private @aten.expand.719(%arg0: tensor<f32>) -> tensor<128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128x128xf32>
    return %3 : tensor<128x128xf32>
  }
  func.func private @aten.where.762(%arg0: tensor<128x128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
  func.func private @aten.index_put.811(%arg0: tensor<512x128xf32>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x128xf32>) -> tensor<512x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    return %1 : tensor<512x128xf32>
  }
  func.func private @aten.permute.818(%arg0: tensor<512x128xf32>) -> tensor<512x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<512x128xf32>) -> tensor<512x128xf32>
    return %0 : tensor<512x128xf32>
  }
  func.func private @aten.sum.827(%arg0: tensor<2x128x30522xf32>) -> tensor<1x1x30522xf32> {
    %0 = mhlo.constant dense<256> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = "mhlo.reduce"(%arg0, %1) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<30522xf32>) -> tensor<1x1x30522xf32>
    return %3 : tensor<1x1x30522xf32>
  }
  func.func private @aten.view.834(%arg0: tensor<1x1x30522xf32>) -> tensor<30522xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x1x30522xf32>) -> tensor<30522xf32>
    return %0 : tensor<30522xf32>
  }
}

