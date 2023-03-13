Testcase(contents=[Content(stages=[Input], content=r"""
!tuple = tuple<tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>>
module  {
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<256xf32>, %arg30: tensor<512xf32>, %arg31: tensor<512xf32>, %arg32: tensor<512xf32>, %arg33: tensor<512xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<512xf32>, %arg38: tensor<512xf32>, %arg39: tensor<512xf32>, %arg40: tensor<64xf32>, %arg41: tensor<64xf32>, %arg42: tensor<64xf32>, %arg43: tensor<64xf32>, %arg44: tensor<64xf32>, %arg45: tensor<64xf32>, %arg46: tensor<64xf32>, %arg47: tensor<64xf32>, %arg48: tensor<64xf32>, %arg49: tensor<64xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<128xf32>, %arg53: tensor<128xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<512xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<64x3x7x7xf16>, %arg81: tensor<1x3x224x224xf16>, %arg82: tensor<1x64x112x112xf16>, %arg83: tensor<1x64x112x112xf16>, %arg84: tensor<1x64x56x56xf16>, %arg85: tensor<64x64x3x3xf16>, %arg86: tensor<1x64x56x56xf16>, %arg87: tensor<1x64x56x56xf16>, %arg88: tensor<64x64x3x3xf16>, %arg89: tensor<1x64x56x56xf16>, %arg90: tensor<1x64x56x56xf16>, %arg91: tensor<64x64x3x3xf16>, %arg92: tensor<1x64x56x56xf16>, %arg93: tensor<1x64x56x56xf16>, %arg94: tensor<64x64x3x3xf16>, %arg95: tensor<1x64x56x56xf16>, %arg96: tensor<1x64x56x56xf16>, %arg97: tensor<128x64x3x3xf16>, %arg98: tensor<1x128x28x28xf16>, %arg99: tensor<1x128x28x28xf16>, %arg100: tensor<128x128x3x3xf16>, %arg101: tensor<1x128x28x28xf16>, %arg102: tensor<128x64x1x1xf16>, %arg103: tensor<1x128x28x28xf16>, %arg104: tensor<1x128x28x28xf16>, %arg105: tensor<128x128x3x3xf16>, %arg106: tensor<1x128x28x28xf16>, %arg107: tensor<1x128x28x28xf16>, %arg108: tensor<128x128x3x3xf16>, %arg109: tensor<1x128x28x28xf16>, %arg110: tensor<1x128x28x28xf16>, %arg111: tensor<256x128x3x3xf16>, %arg112: tensor<1x256x14x14xf16>, %arg113: tensor<1x256x14x14xf16>, %arg114: tensor<256x256x3x3xf16>, %arg115: tensor<1x256x14x14xf16>, %arg116: tensor<256x128x1x1xf16>, %arg117: tensor<1x256x14x14xf16>, %arg118: tensor<1x256x14x14xf16>, %arg119: tensor<256x256x3x3xf16>, %arg120: tensor<1x256x14x14xf16>, %arg121: tensor<1x256x14x14xf16>, %arg122: tensor<256x256x3x3xf16>, %arg123: tensor<1x256x14x14xf16>, %arg124: tensor<1x256x14x14xf16>, %arg125: tensor<512x256x3x3xf16>, %arg126: tensor<1x512x7x7xf16>, %arg127: tensor<1x512x7x7xf16>, %arg128: tensor<512x512x3x3xf16>, %arg129: tensor<1x512x7x7xf16>, %arg130: tensor<512x256x1x1xf16>, %arg131: tensor<1x512x7x7xf16>, %arg132: tensor<1x512x7x7xf16>, %arg133: tensor<512x512x3x3xf16>, %arg134: tensor<1x512x7x7xf16>, %arg135: tensor<1x512x7x7xf16>, %arg136: tensor<512x512x3x3xf16>, %arg137: tensor<1x512x7x7xf16>, %arg138: tensor<1x512x7x7xf16>, %arg139: tensor<1x512xf16>, %arg140: tensor<512x1000xf16>, %arg141: tensor<1x1000xf16>) -> !tuple {
    %0 = call @aten.native_batch_norm.143(%arg82, %arg1, %arg0, %arg40, %arg41) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x112x112xf16>
    %2 = "mhlo.get_tuple_element"(%0) {index = 2 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %3 = call @aten.native_batch_norm.173(%arg86, %arg3, %arg2, %arg42, %arg43) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %4 = "mhlo.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %5 = "mhlo.get_tuple_element"(%3) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %6 = call @aten.native_batch_norm.173(%arg89, %arg5, %arg4, %arg44, %arg45) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %7 = "mhlo.get_tuple_element"(%6) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %8 = "mhlo.get_tuple_element"(%6) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %9 = call @aten.native_batch_norm.173(%arg92, %arg7, %arg6, %arg46, %arg47) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %10 = "mhlo.get_tuple_element"(%9) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %11 = "mhlo.get_tuple_element"(%9) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %12 = call @aten.native_batch_norm.173(%arg95, %arg9, %arg8, %arg48, %arg49) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %13 = "mhlo.get_tuple_element"(%12) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %14 = "mhlo.get_tuple_element"(%12) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %15 = call @aten.native_batch_norm.214(%arg98, %arg11, %arg10, %arg50, %arg51) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %16 = "mhlo.get_tuple_element"(%15) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %17 = "mhlo.get_tuple_element"(%15) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %18 = call @aten.native_batch_norm.214(%arg101, %arg13, %arg12, %arg52, %arg53) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %19 = "mhlo.get_tuple_element"(%18) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %20 = "mhlo.get_tuple_element"(%18) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %21 = call @aten.native_batch_norm.214(%arg106, %arg17, %arg16, %arg56, %arg57) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %22 = "mhlo.get_tuple_element"(%21) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %23 = "mhlo.get_tuple_element"(%21) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %24 = call @aten.native_batch_norm.214(%arg109, %arg19, %arg18, %arg58, %arg59) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %25 = "mhlo.get_tuple_element"(%24) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %26 = "mhlo.get_tuple_element"(%24) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %27 = call @aten.native_batch_norm.261(%arg112, %arg21, %arg20, %arg60, %arg61) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %28 = "mhlo.get_tuple_element"(%27) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %29 = "mhlo.get_tuple_element"(%27) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %30 = call @aten.native_batch_norm.261(%arg115, %arg23, %arg22, %arg62, %arg63) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %31 = "mhlo.get_tuple_element"(%30) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %32 = "mhlo.get_tuple_element"(%30) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %33 = call @aten.native_batch_norm.261(%arg120, %arg27, %arg26, %arg66, %arg67) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %34 = "mhlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %35 = "mhlo.get_tuple_element"(%33) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %36 = call @aten.native_batch_norm.261(%arg123, %arg29, %arg28, %arg68, %arg69) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %37 = "mhlo.get_tuple_element"(%36) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %38 = "mhlo.get_tuple_element"(%36) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %39 = call @aten.native_batch_norm.308(%arg126, %arg31, %arg30, %arg70, %arg71) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %40 = "mhlo.get_tuple_element"(%39) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %41 = "mhlo.get_tuple_element"(%39) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %42 = call @aten.native_batch_norm.308(%arg129, %arg33, %arg32, %arg72, %arg73) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %43 = "mhlo.get_tuple_element"(%42) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %44 = "mhlo.get_tuple_element"(%42) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %45 = call @aten.native_batch_norm.308(%arg134, %arg37, %arg36, %arg76, %arg77) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %46 = "mhlo.get_tuple_element"(%45) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %47 = "mhlo.get_tuple_element"(%45) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %48 = call @aten.native_batch_norm.308(%arg137, %arg39, %arg38, %arg78, %arg79) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %49 = "mhlo.get_tuple_element"(%48) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %50 = "mhlo.get_tuple_element"(%48) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %51 = call @aten.permute.354(%arg140) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    %52 = call @aten.mm.358(%arg141, %51) : (tensor<1x1000xf16>, tensor<1000x512xf16>) -> tensor<1x512xf16>
    %53 = call @aten.view.363(%52) : (tensor<1x512xf16>) -> tensor<1x512x1x1xf16>
    %54 = call @aten.expand.367(%53) : (tensor<1x512x1x1xf16>) -> tensor<1x512x7x7xf16>
    %55 = mhlo.constant dense<4.900000e+01> : tensor<f16>
    %56 = call @aten.div.373(%54, %55) : (tensor<1x512x7x7xf16>, tensor<f16>) -> tensor<1x512x7x7xf16>
    %57 = call @aten.threshold_backward.379(%56, %arg138) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %58 = "mhlo.get_tuple_element"(%48) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %59 = "mhlo.get_tuple_element"(%48) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %60 = call @aten.native_batch_norm_backward.389(%57, %arg137, %arg39, %58, %59) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %61 = "mhlo.get_tuple_element"(%60) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %62 = call @aten.convolution_backward_overrideable.418(%61, %arg135, %arg136) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %63 = "mhlo.get_tuple_element"(%62) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %64 = "mhlo.get_tuple_element"(%62) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<1x512x7x7xf16>
    %65 = call @aten.threshold_backward.379(%64, %arg135) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %66 = "mhlo.get_tuple_element"(%45) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %67 = "mhlo.get_tuple_element"(%45) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %68 = call @aten.native_batch_norm_backward.389(%65, %arg134, %arg37, %66, %67) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %69 = "mhlo.get_tuple_element"(%68) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %70 = call @aten.convolution_backward_overrideable.418(%69, %arg132, %arg133) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %71 = "mhlo.get_tuple_element"(%70) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %72 = "mhlo.get_tuple_element"(%70) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<1x512x7x7xf16>
    %73 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %74 = call @aten.expand.336(%73) : (tensor<f16>) -> tensor<1x512x7x7xf16>
    %75 = call @aten.mul.443(%72, %74) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %76 = call @aten.add.448(%57, %75) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %77 = call @aten.threshold_backward.379(%76, %arg132) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %78 = "mhlo.get_tuple_element"(%42) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %79 = "mhlo.get_tuple_element"(%42) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %80 = call @aten.native_batch_norm_backward.389(%77, %arg129, %arg33, %78, %79) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %81 = "mhlo.get_tuple_element"(%80) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %82 = call @aten.convolution_backward_overrideable.418(%81, %arg127, %arg128) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %83 = "mhlo.get_tuple_element"(%82) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %84 = "mhlo.get_tuple_element"(%82) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<1x512x7x7xf16>
    %85 = call @aten.threshold_backward.379(%84, %arg127) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %86 = "mhlo.get_tuple_element"(%39) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %87 = "mhlo.get_tuple_element"(%39) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %88 = call @aten.native_batch_norm_backward.389(%85, %arg126, %arg31, %86, %87) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %89 = "mhlo.get_tuple_element"(%88) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %90 = call @aten.convolution_backward_overrideable.471(%89, %arg124, %arg125) : (tensor<1x512x7x7xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
    %91 = "mhlo.get_tuple_element"(%90) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %92 = call @aten.native_batch_norm.308(%arg131, %arg35, %arg34, %arg74, %arg75) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %93 = "mhlo.get_tuple_element"(%92) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %94 = "mhlo.get_tuple_element"(%92) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %95 = "mhlo.get_tuple_element"(%92) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %96 = "mhlo.get_tuple_element"(%92) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %97 = call @aten.native_batch_norm_backward.389(%77, %arg131, %arg35, %95, %96) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %98 = "mhlo.get_tuple_element"(%97) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %99 = call @aten.convolution_backward_overrideable.505(%98, %arg124, %arg130) : (tensor<1x512x7x7xf16>, tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
    %100 = "mhlo.get_tuple_element"(%99) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %101 = "mhlo.get_tuple_element"(%99) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<1x256x14x14xf16>
    %102 = "mhlo.get_tuple_element"(%90) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<1x256x14x14xf16>
    %103 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %104 = call @aten.expand.289(%103) : (tensor<f16>) -> tensor<1x256x14x14xf16>
    %105 = call @aten.mul.487(%102, %104) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %106 = call @aten.add.521(%101, %105) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %107 = call @aten.threshold_backward.526(%106, %arg124) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %108 = "mhlo.get_tuple_element"(%36) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %109 = "mhlo.get_tuple_element"(%36) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %110 = call @aten.native_batch_norm_backward.536(%107, %arg123, %arg29, %108, %109) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %111 = "mhlo.get_tuple_element"(%110) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %112 = call @aten.convolution_backward_overrideable.565(%111, %arg121, %arg122) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %113 = "mhlo.get_tuple_element"(%112) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %114 = "mhlo.get_tuple_element"(%112) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<1x256x14x14xf16>
    %115 = call @aten.threshold_backward.526(%114, %arg121) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %116 = "mhlo.get_tuple_element"(%33) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %117 = "mhlo.get_tuple_element"(%33) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %118 = call @aten.native_batch_norm_backward.536(%115, %arg120, %arg27, %116, %117) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %119 = "mhlo.get_tuple_element"(%118) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %120 = call @aten.convolution_backward_overrideable.565(%119, %arg118, %arg119) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %121 = "mhlo.get_tuple_element"(%120) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %122 = "mhlo.get_tuple_element"(%120) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<1x256x14x14xf16>
    %123 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %124 = call @aten.expand.289(%123) : (tensor<f16>) -> tensor<1x256x14x14xf16>
    %125 = call @aten.mul.487(%122, %124) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %126 = call @aten.add.521(%107, %125) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %127 = call @aten.threshold_backward.526(%126, %arg118) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %128 = "mhlo.get_tuple_element"(%30) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %129 = "mhlo.get_tuple_element"(%30) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %130 = call @aten.native_batch_norm_backward.536(%127, %arg115, %arg23, %128, %129) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %131 = "mhlo.get_tuple_element"(%130) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %132 = call @aten.convolution_backward_overrideable.565(%131, %arg113, %arg114) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %133 = "mhlo.get_tuple_element"(%132) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %134 = "mhlo.get_tuple_element"(%132) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<1x256x14x14xf16>
    %135 = call @aten.threshold_backward.526(%134, %arg113) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %136 = "mhlo.get_tuple_element"(%27) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %137 = "mhlo.get_tuple_element"(%27) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %138 = call @aten.native_batch_norm_backward.536(%135, %arg112, %arg21, %136, %137) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %139 = "mhlo.get_tuple_element"(%138) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %140 = call @aten.convolution_backward_overrideable.610(%139, %arg110, %arg111) : (tensor<1x256x14x14xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
    %141 = "mhlo.get_tuple_element"(%140) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %142 = call @aten.native_batch_norm.261(%arg117, %arg25, %arg24, %arg64, %arg65) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %143 = "mhlo.get_tuple_element"(%142) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %144 = "mhlo.get_tuple_element"(%142) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %145 = "mhlo.get_tuple_element"(%142) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %146 = "mhlo.get_tuple_element"(%142) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %147 = call @aten.native_batch_norm_backward.536(%127, %arg117, %arg25, %145, %146) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %148 = "mhlo.get_tuple_element"(%147) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %149 = call @aten.convolution_backward_overrideable.644(%148, %arg110, %arg116) : (tensor<1x256x14x14xf16>, tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
    %150 = "mhlo.get_tuple_element"(%149) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %151 = "mhlo.get_tuple_element"(%149) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<1x128x28x28xf16>
    %152 = "mhlo.get_tuple_element"(%140) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<1x128x28x28xf16>
    %153 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %154 = call @aten.expand.242(%153) : (tensor<f16>) -> tensor<1x128x28x28xf16>
    %155 = call @aten.mul.626(%152, %154) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %156 = call @aten.add.660(%151, %155) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %157 = call @aten.threshold_backward.665(%156, %arg110) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %158 = "mhlo.get_tuple_element"(%24) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %159 = "mhlo.get_tuple_element"(%24) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %160 = call @aten.native_batch_norm_backward.675(%157, %arg109, %arg19, %158, %159) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %161 = "mhlo.get_tuple_element"(%160) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %162 = call @aten.convolution_backward_overrideable.704(%161, %arg107, %arg108) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %163 = "mhlo.get_tuple_element"(%162) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %164 = "mhlo.get_tuple_element"(%162) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<1x128x28x28xf16>
    %165 = call @aten.threshold_backward.665(%164, %arg107) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %166 = "mhlo.get_tuple_element"(%21) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %167 = "mhlo.get_tuple_element"(%21) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %168 = call @aten.native_batch_norm_backward.675(%165, %arg106, %arg17, %166, %167) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %169 = "mhlo.get_tuple_element"(%168) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %170 = call @aten.convolution_backward_overrideable.704(%169, %arg104, %arg105) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %171 = "mhlo.get_tuple_element"(%170) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %172 = "mhlo.get_tuple_element"(%170) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<1x128x28x28xf16>
    %173 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %174 = call @aten.expand.242(%173) : (tensor<f16>) -> tensor<1x128x28x28xf16>
    %175 = call @aten.mul.626(%172, %174) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %176 = call @aten.add.660(%157, %175) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %177 = call @aten.threshold_backward.665(%176, %arg104) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %178 = "mhlo.get_tuple_element"(%18) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %179 = "mhlo.get_tuple_element"(%18) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %180 = call @aten.native_batch_norm_backward.675(%177, %arg101, %arg13, %178, %179) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %181 = "mhlo.get_tuple_element"(%180) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %182 = call @aten.convolution_backward_overrideable.704(%181, %arg99, %arg100) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %183 = "mhlo.get_tuple_element"(%182) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %184 = "mhlo.get_tuple_element"(%182) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<1x128x28x28xf16>
    %185 = call @aten.threshold_backward.665(%184, %arg99) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %186 = "mhlo.get_tuple_element"(%15) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %187 = "mhlo.get_tuple_element"(%15) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %188 = call @aten.native_batch_norm_backward.675(%185, %arg98, %arg11, %186, %187) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %189 = "mhlo.get_tuple_element"(%188) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %190 = call @aten.convolution_backward_overrideable.749(%189, %arg96, %arg97) : (tensor<1x128x28x28xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
    %191 = "mhlo.get_tuple_element"(%190) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %192 = call @aten.native_batch_norm.214(%arg103, %arg15, %arg14, %arg54, %arg55) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %193 = "mhlo.get_tuple_element"(%192) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %194 = "mhlo.get_tuple_element"(%192) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %195 = "mhlo.get_tuple_element"(%192) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %196 = "mhlo.get_tuple_element"(%192) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %197 = call @aten.native_batch_norm_backward.675(%177, %arg103, %arg15, %195, %196) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %198 = "mhlo.get_tuple_element"(%197) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %199 = call @aten.convolution_backward_overrideable.783(%198, %arg96, %arg102) : (tensor<1x128x28x28xf16>, tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
    %200 = "mhlo.get_tuple_element"(%199) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %201 = "mhlo.get_tuple_element"(%199) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<1x64x56x56xf16>
    %202 = "mhlo.get_tuple_element"(%190) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<1x64x56x56xf16>
    %203 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %204 = call @aten.expand.166(%203) : (tensor<f16>) -> tensor<1x64x56x56xf16>
    %205 = call @aten.mul.765(%202, %204) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %206 = call @aten.add.799(%201, %205) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %207 = call @aten.threshold_backward.804(%206, %arg96) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %208 = "mhlo.get_tuple_element"(%12) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %209 = "mhlo.get_tuple_element"(%12) {index = 3 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %210 = call @aten.native_batch_norm_backward.814(%207, %arg95, %arg9, %208, %209) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %211 = "mhlo.get_tuple_element"(%210) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %212 = call @aten.convolution_backward_overrideable.843(%211, %arg93, %arg94) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %213 = "mhlo.get_tuple_element"(%212) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %214 = "mhlo.get_tuple_element"(%212) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<1x64x56x56xf16>
    %215 = call @aten.threshold_backward.804(%214, %arg93) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %216 = "mhlo.get_tuple_element"(%9) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %217 = "mhlo.get_tuple_element"(%9) {index = 3 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %218 = call @aten.native_batch_norm_backward.814(%215, %arg92, %arg7, %216, %217) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %219 = "mhlo.get_tuple_element"(%218) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %220 = call @aten.convolution_backward_overrideable.843(%219, %arg90, %arg91) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %221 = "mhlo.get_tuple_element"(%220) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %222 = "mhlo.get_tuple_element"(%220) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<1x64x56x56xf16>
    %223 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %224 = call @aten.expand.166(%223) : (tensor<f16>) -> tensor<1x64x56x56xf16>
    %225 = call @aten.mul.765(%222, %224) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %226 = call @aten.add.799(%207, %225) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %227 = call @aten.threshold_backward.804(%226, %arg90) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %228 = "mhlo.get_tuple_element"(%6) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %229 = "mhlo.get_tuple_element"(%6) {index = 3 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %230 = call @aten.native_batch_norm_backward.814(%227, %arg89, %arg5, %228, %229) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %231 = "mhlo.get_tuple_element"(%230) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %232 = call @aten.convolution_backward_overrideable.843(%231, %arg87, %arg88) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %233 = "mhlo.get_tuple_element"(%232) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %234 = "mhlo.get_tuple_element"(%232) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<1x64x56x56xf16>
    %235 = call @aten.threshold_backward.804(%234, %arg87) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %236 = "mhlo.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %237 = "mhlo.get_tuple_element"(%3) {index = 3 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %238 = call @aten.native_batch_norm_backward.814(%235, %arg86, %arg3, %236, %237) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %239 = "mhlo.get_tuple_element"(%238) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %240 = call @aten.convolution_backward_overrideable.843(%239, %arg84, %arg85) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %241 = "mhlo.get_tuple_element"(%240) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %242 = "mhlo.get_tuple_element"(%240) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<1x64x56x56xf16>
    %243 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %244 = call @aten.expand.166(%243) : (tensor<f16>) -> tensor<1x64x56x56xf16>
    %245 = call @aten.mul.765(%242, %244) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %246 = call @aten.add.799(%227, %245) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %247 = call @aten.max_pool2d_with_indices_backward.898(%246, %arg83) : (tensor<1x64x56x56xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %248 = call @aten.threshold_backward.904(%247, %arg83) : (tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %249 = "mhlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %250 = "mhlo.get_tuple_element"(%0) {index = 3 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %251 = call @aten.native_batch_norm_backward.914(%248, %arg82, %arg1, %249, %250) : (tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
    %252 = "mhlo.get_tuple_element"(%251) {index = 0 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x112x112xf16>
    %253 = call @aten.convolution_backward_overrideable.943(%252, %arg81, %arg80) : (tensor<1x64x112x112xf16>, tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tuple<tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
    %254 = "mhlo.get_tuple_element"(%253) {index = 0 : i32} : (tuple<tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<1x3x224x224xf16>
    %255 = "mhlo.get_tuple_element"(%253) {index = 2 : i32} : (tuple<tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %256 = "mhlo.get_tuple_element"(%251) {index = 2 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %257 = "mhlo.get_tuple_element"(%251) {index = 1 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %258 = "mhlo.get_tuple_element"(%253) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<64x3x7x7xf16>
    %259 = "mhlo.convert"(%258) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %260 = call @aten.sum.964(%arg141) : (tensor<1x1000xf16>) -> tensor<1x1000xf32>
    %261 = call @aten.view.972(%260) : (tensor<1x1000xf32>) -> tensor<1000xf32>
    %262 = "mhlo.convert"(%261) : (tensor<1000xf32>) -> tensor<1000xf16>
    %263 = "mhlo.convert"(%262) : (tensor<1000xf16>) -> tensor<1000xf32>
    %264 = call @aten.permute.978(%arg141) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1x1000xf16>) -> tensor<1000x1xf16>
    %265 = call @aten.mm.982(%264, %arg139) : (tensor<1000x1xf16>, tensor<1x512xf16>) -> tensor<1000x512xf16>
    %266 = call @aten.permute.987(%265) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    %267 = call @aten.permute.991(%266) : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    %268 = "mhlo.convert"(%267) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %269 = "mhlo.get_tuple_element"(%238) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %270 = "mhlo.get_tuple_element"(%238) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %271 = "mhlo.get_tuple_element"(%230) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %272 = "mhlo.get_tuple_element"(%230) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %273 = "mhlo.get_tuple_element"(%240) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %274 = "mhlo.convert"(%273) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %275 = "mhlo.get_tuple_element"(%232) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %276 = "mhlo.convert"(%275) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %277 = "mhlo.get_tuple_element"(%218) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %278 = "mhlo.get_tuple_element"(%218) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %279 = "mhlo.get_tuple_element"(%210) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %280 = "mhlo.get_tuple_element"(%210) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %281 = "mhlo.get_tuple_element"(%220) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %282 = "mhlo.convert"(%281) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %283 = "mhlo.get_tuple_element"(%212) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %284 = "mhlo.convert"(%283) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %285 = "mhlo.get_tuple_element"(%188) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %286 = "mhlo.get_tuple_element"(%188) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %287 = "mhlo.get_tuple_element"(%180) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %288 = "mhlo.get_tuple_element"(%180) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %289 = "mhlo.get_tuple_element"(%190) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<128x64x3x3xf16>
    %290 = "mhlo.convert"(%289) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %291 = "mhlo.get_tuple_element"(%182) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %292 = "mhlo.convert"(%291) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %293 = "mhlo.get_tuple_element"(%199) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<128x64x1x1xf16>
    %294 = "mhlo.convert"(%293) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %295 = "mhlo.get_tuple_element"(%197) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %296 = "mhlo.get_tuple_element"(%197) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %297 = "mhlo.get_tuple_element"(%168) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %298 = "mhlo.get_tuple_element"(%168) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %299 = "mhlo.get_tuple_element"(%160) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %300 = "mhlo.get_tuple_element"(%160) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %301 = "mhlo.get_tuple_element"(%170) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %302 = "mhlo.convert"(%301) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %303 = "mhlo.get_tuple_element"(%162) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %304 = "mhlo.convert"(%303) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %305 = "mhlo.get_tuple_element"(%138) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %306 = "mhlo.get_tuple_element"(%138) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %307 = "mhlo.get_tuple_element"(%130) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %308 = "mhlo.get_tuple_element"(%130) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %309 = "mhlo.get_tuple_element"(%140) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<256x128x3x3xf16>
    %310 = "mhlo.convert"(%309) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %311 = "mhlo.get_tuple_element"(%132) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %312 = "mhlo.convert"(%311) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %313 = "mhlo.get_tuple_element"(%149) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<256x128x1x1xf16>
    %314 = "mhlo.convert"(%313) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %315 = "mhlo.get_tuple_element"(%147) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %316 = "mhlo.get_tuple_element"(%147) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %317 = "mhlo.get_tuple_element"(%118) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %318 = "mhlo.get_tuple_element"(%118) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %319 = "mhlo.get_tuple_element"(%110) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %320 = "mhlo.get_tuple_element"(%110) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %321 = "mhlo.get_tuple_element"(%120) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %322 = "mhlo.convert"(%321) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %323 = "mhlo.get_tuple_element"(%112) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %324 = "mhlo.convert"(%323) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %325 = "mhlo.get_tuple_element"(%88) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %326 = "mhlo.get_tuple_element"(%88) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %327 = "mhlo.get_tuple_element"(%80) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %328 = "mhlo.get_tuple_element"(%80) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %329 = "mhlo.get_tuple_element"(%90) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<512x256x3x3xf16>
    %330 = "mhlo.convert"(%329) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %331 = "mhlo.get_tuple_element"(%82) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %332 = "mhlo.convert"(%331) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %333 = "mhlo.get_tuple_element"(%99) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<512x256x1x1xf16>
    %334 = "mhlo.convert"(%333) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %335 = "mhlo.get_tuple_element"(%97) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %336 = "mhlo.get_tuple_element"(%97) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %337 = "mhlo.get_tuple_element"(%68) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %338 = "mhlo.get_tuple_element"(%68) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %339 = "mhlo.get_tuple_element"(%60) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %340 = "mhlo.get_tuple_element"(%60) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %341 = "mhlo.get_tuple_element"(%70) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %342 = "mhlo.convert"(%341) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %343 = "mhlo.get_tuple_element"(%62) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %344 = "mhlo.convert"(%343) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %345 = "mhlo.tuple"(%256, %257, %259, %263, %268, %269, %270, %271, %272, %274, %276, %277, %278, %279, %280, %282, %284, %285, %286, %287, %288, %290, %292, %294, %295, %296, %297, %298, %299, %300, %302, %304, %305, %306, %307, %308, %310, %312, %314, %315, %316, %317, %318, %319, %320, %322, %324, %325, %326, %327, %328, %330, %332, %334, %335, %336, %337, %338, %339, %340, %342, %344) : (tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>) -> !tuple
    return %345 : !tuple
  }
  func.func private @aten.native_batch_norm.143(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %8 = mhlo.add %1#2, %7 : tensor<64xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<64xf32>) -> tensor<64xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.native_batch_norm.173(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %8 = mhlo.add %1#2, %7 : tensor<64xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<64xf32>) -> tensor<64xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.native_batch_norm.214(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %8 = mhlo.add %1#2, %7 : tensor<128xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<128xf32>) -> tensor<128xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %10 : tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.native_batch_norm.261(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %8 = mhlo.add %1#2, %7 : tensor<256xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<256xf32>) -> tensor<256xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %10 : tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.native_batch_norm.308(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %8 = mhlo.add %1#2, %7 : tensor<512xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<512xf32>) -> tensor<512xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %10 : tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.permute.354(%arg0: tensor<512x1000xf16>) -> tensor<1000x512xf16> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func.func private @aten.mm.358(%arg0: tensor<1x1000xf16>, %arg1: tensor<1000x512xf16>) -> tensor<1x512xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x1000xf16>, tensor<1000x512xf16>) -> tensor<1x512xf16>
    return %0 : tensor<1x512xf16>
  }
  func.func private @aten.view.363(%arg0: tensor<1x512xf16>) -> tensor<1x512x1x1xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x512xf16>) -> tensor<1x512x1x1xf16>
    return %0 : tensor<1x512x1x1xf16>
  }
  func.func private @aten.expand.367(%arg0: tensor<1x512x1x1xf16>) -> tensor<1x512x7x7xf16> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x512x1x1xf16>) -> tensor<1x512x1x1xf16>
    %1 = "mhlo.reshape"(%0) : (tensor<1x512x1x1xf16>) -> tensor<1x512xf16>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x512xf16>) -> tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.div.373(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<f16>) -> tensor<1x512x7x7xf16> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x512x7x7xf16>
    %1 = mhlo.divide %arg0, %0 : tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.threshold_backward.379(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x512x7x7xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x512x7x7xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<1x512x7x7xi1>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    return %5 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm_backward.389(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<512xf32>
    %4 = mhlo.multiply %3, %3 : tensor<512xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %7 = mhlo.subtract %4, %6 : tensor<512xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    return %14 : tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.418(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>, %arg2: tensor<512x512x3x3xf16>) -> tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<1x512x7x7xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>) -> tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
  }
  func.func private @aten.expand.336(%arg0: tensor<f16>) -> tensor<1x512x7x7xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<1xf16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf16>) -> tensor<1x512x7x7xf16>
    return %3 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.mul.443(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x512x7x7xf16>
    return %0 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.add.448(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x512x7x7xf16>
    return %0 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.convolution_backward_overrideable.471(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<512x256x3x3xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<1x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
  }
  func.func private @aten.convolution_backward_overrideable.505(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<512x256x1x1xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<1x1x256x512xf16>) -> tensor<1x1x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<1x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
  }
  func.func private @aten.expand.289(%arg0: tensor<f16>) -> tensor<1x256x14x14xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<1xf16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf16>) -> tensor<1x256x14x14xf16>
    return %3 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.mul.487(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x256x14x14xf16>
    return %0 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.add.521(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x256x14x14xf16>
    return %0 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.threshold_backward.526(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x256x14x14xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x256x14x14xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<1x256x14x14xi1>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    return %5 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm_backward.536(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<256xf32>
    %4 = mhlo.multiply %3, %3 : tensor<256xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %7 = mhlo.subtract %4, %6 : tensor<256xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    return %14 : tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.565(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<256x256x3x3xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<1x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>) -> tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
  }
  func.func private @aten.convolution_backward_overrideable.610(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<256x128x3x3xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<1x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
  }
  func.func private @aten.convolution_backward_overrideable.644(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<256x128x1x1xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<1x1x128x256xf16>) -> tensor<1x1x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<1x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
  }
  func.func private @aten.expand.242(%arg0: tensor<f16>) -> tensor<1x128x28x28xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<1xf16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf16>) -> tensor<1x128x28x28xf16>
    return %3 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.mul.626(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x128x28x28xf16>
    return %0 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.add.660(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x128x28x28xf16>
    return %0 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.threshold_backward.665(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x128x28x28xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x128x28x28xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<1x128x28x28xi1>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    return %5 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm_backward.675(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<128xf32>
    %4 = mhlo.multiply %3, %3 : tensor<128xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %7 = mhlo.subtract %4, %6 : tensor<128xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    return %14 : tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.704(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<128x128x3x3xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<1x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>) -> tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
  }
  func.func private @aten.convolution_backward_overrideable.749(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<128x64x3x3xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<1x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
  }
  func.func private @aten.convolution_backward_overrideable.783(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<128x64x1x1xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<1x1x64x128xf16>) -> tensor<1x1x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<1x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
  }
  func.func private @aten.expand.166(%arg0: tensor<f16>) -> tensor<1x64x56x56xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<1xf16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf16>) -> tensor<1x64x56x56xf16>
    return %3 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.mul.765(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x64x56x56xf16>
    return %0 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.add.799(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x64x56x56xf16>
    return %0 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.threshold_backward.804(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x64x56x56xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x64x56x56xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<1x64x56x56xi1>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    return %5 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm_backward.814(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    return %14 : tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.843(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<64x64x3x3xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<1x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x64x56x56xf16>, tensor<f16>) -> tensor<64xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
  }
  func.func private @aten.max_pool2d_with_indices_backward.898(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.select_and_scatter"(%arg1, %arg0, %0) ( {
    ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):  // no predecessors
      %2 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):  // no predecessors
      %2 = mhlo.add %arg2, %arg3 : tensor<f16>
      "mhlo.return"(%2) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<f16>) -> tensor<1x64x112x112xf16>
    return %1 : tensor<1x64x112x112xf16>
  }
  func.func private @aten.threshold_backward.904(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x64x112x112xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x64x112x112xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<1x64x112x112xi1>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    return %5 : tensor<1x64x112x112xf16>
  }
  func.func private @aten.native_batch_norm_backward.914(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<1x64x112x112xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x112x112xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
    return %14 : tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.943(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<1x3x224x224xf16>, %arg2: tensor<64x3x7x7xf16>) -> tuple<tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x3x7x7xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<7x7x3x64xf16>) -> tensor<7x7x3x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[3, 4], [3, 4]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x112x112xf16>, tensor<7x7x3x64xf16>) -> tensor<1x3x224x224xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x64x112x112xf16>, tensor<f16>) -> tensor<64xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>) -> tuple<tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
  }
  func.func private @aten.sum.964(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf32> {
    %0 = mhlo.constant dense<1> : tensor<i64>
    %1 = "mhlo.convert"(%arg0) : (tensor<1x1000xf16>) -> tensor<1x1000xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "mhlo.reduce"(%1, %2) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %5 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%5) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %4 = "mhlo.reshape"(%3) : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %4 : tensor<1x1000xf32>
  }
  func.func private @aten.view.972(%arg0: tensor<1x1000xf32>) -> tensor<1000xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x1000xf32>) -> tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func.func private @aten.permute.978(%arg0: tensor<1x1000xf16>) -> tensor<1000x1xf16> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x1000xf16>) -> tensor<1000x1xf16>
    return %0 : tensor<1000x1xf16>
  }
  func.func private @aten.mm.982(%arg0: tensor<1000x1xf16>, %arg1: tensor<1x512xf16>) -> tensor<1000x512xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1000x1xf16>, tensor<1x512xf16>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func.func private @aten.permute.987(%arg0: tensor<1000x512xf16>) -> tensor<512x1000xf16> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    return %0 : tensor<512x1000xf16>
  }
  func.func private @aten.permute.991(%arg0: tensor<512x1000xf16>) -> tensor<1000x512xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
}
""")], pipelines=[
    InputPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    HloOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    LinalgTensorOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    BufferizeOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    SCFOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    AffineOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    GPUOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    ByreOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    ByreHostPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    HostOutputPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    NVVMCodegenPipeline(R"""
// CHECK-LABEL: gpu.module @unified
"""),
    PTXCodegenPipeline(R"""
// CHECK-LABEL: .visible .entry Unknown
"""),
])