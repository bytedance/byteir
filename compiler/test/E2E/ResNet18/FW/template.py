Testcase(contents=[Content(stages=[Input], content=r"""
!tuple = tuple<tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>>
module  {
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<1000xf32>, %arg4: tensor<1000x512xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64x64x3x3xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64x64x3x3xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x64x3x3xf32>, %arg22: tensor<128x128x3x3xf32>, %arg23: tensor<128x64x1x1xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128x128x3x3xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x128x3x3xf32>, %arg37: tensor<256x256x3x3xf32>, %arg38: tensor<256x128x1x1xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<256xf32>, %arg44: tensor<256xf32>, %arg45: tensor<256x256x3x3xf32>, %arg46: tensor<256x256x3x3xf32>, %arg47: tensor<512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<512x256x3x3xf32>, %arg52: tensor<512x512x3x3xf32>, %arg53: tensor<512x256x1x1xf32>, %arg54: tensor<512xf32>, %arg55: tensor<512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<512xf32>, %arg59: tensor<512xf32>, %arg60: tensor<512x512x3x3xf32>, %arg61: tensor<512x512x3x3xf32>, %arg62: tensor<i64>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<i64>, %arg66: tensor<64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<i64>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<i64>, %arg72: tensor<64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<i64>, %arg75: tensor<64xf32>, %arg76: tensor<64xf32>, %arg77: tensor<i64>, %arg78: tensor<128xf32>, %arg79: tensor<128xf32>, %arg80: tensor<i64>, %arg81: tensor<128xf32>, %arg82: tensor<128xf32>, %arg83: tensor<i64>, %arg84: tensor<128xf32>, %arg85: tensor<128xf32>, %arg86: tensor<i64>, %arg87: tensor<128xf32>, %arg88: tensor<128xf32>, %arg89: tensor<i64>, %arg90: tensor<128xf32>, %arg91: tensor<128xf32>, %arg92: tensor<i64>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<i64>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<i64>, %arg99: tensor<256xf32>, %arg100: tensor<256xf32>, %arg101: tensor<i64>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<i64>, %arg105: tensor<256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<i64>, %arg108: tensor<512xf32>, %arg109: tensor<512xf32>, %arg110: tensor<i64>, %arg111: tensor<512xf32>, %arg112: tensor<512xf32>, %arg113: tensor<i64>, %arg114: tensor<512xf32>, %arg115: tensor<512xf32>, %arg116: tensor<i64>, %arg117: tensor<512xf32>, %arg118: tensor<512xf32>, %arg119: tensor<i64>, %arg120: tensor<512xf32>, %arg121: tensor<512xf32>, %arg122: tensor<1x3x224x224xf32>) -> !tuple {
    %0 = "mhlo.convert"(%arg122) : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16>
    %1 = "mhlo.convert"(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %2 = call @aten.convolution_overrideable.175(%0, %1) : (tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<1x64x112x112xf16>
    %3 = call @aten.native_batch_norm.180(%2, %arg1, %arg0, %arg63, %arg64) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %4 = "mhlo.get_tuple_element"(%3) {index = 3 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %5 = "mhlo.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x112x112xf16>
    %6 = call @aten.relu.202(%5) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %7 = call @aten.max_pool2d.277(%6) : (tensor<1x64x112x112xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<1x64x56x56xui32>>
    %8 = "mhlo.get_tuple_element"(%7) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<1x64x56x56xui32>>) -> tensor<1x64x56x56xui32>
    %9 = "mhlo.get_tuple_element"(%7) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<1x64x56x56xui32>>) -> tensor<1x64x56x56xf16>
    %10 = "mhlo.convert"(%arg9) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %11 = call @aten.convolution_overrideable.312(%9, %10) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %12 = call @aten.native_batch_norm.317(%11, %arg6, %arg5, %arg66, %arg67) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %13 = "mhlo.get_tuple_element"(%12) {index = 3 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %14 = "mhlo.get_tuple_element"(%12) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %15 = call @aten.relu.339(%14) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %16 = "mhlo.convert"(%arg10) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %17 = call @aten.convolution_overrideable.312(%15, %16) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %18 = call @aten.native_batch_norm.317(%17, %arg8, %arg7, %arg69, %arg70) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %19 = "mhlo.get_tuple_element"(%18) {index = 3 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %20 = "mhlo.get_tuple_element"(%18) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %21 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %22 = call @aten.expand.164(%21) : (tensor<f16>) -> tensor<1x64x56x56xf16>
    %23 = call @aten.mul.305(%9, %22) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %24 = call @aten.add.351(%20, %23) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %25 = call @aten.relu.339(%24) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %26 = "mhlo.convert"(%arg15) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %27 = call @aten.convolution_overrideable.312(%25, %26) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %28 = call @aten.native_batch_norm.317(%27, %arg12, %arg11, %arg72, %arg73) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %29 = "mhlo.get_tuple_element"(%28) {index = 3 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %30 = "mhlo.get_tuple_element"(%28) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %31 = call @aten.relu.339(%30) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %32 = "mhlo.convert"(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %33 = call @aten.convolution_overrideable.312(%31, %32) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %34 = call @aten.native_batch_norm.317(%33, %arg14, %arg13, %arg75, %arg76) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %35 = "mhlo.get_tuple_element"(%34) {index = 3 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %36 = "mhlo.get_tuple_element"(%34) {index = 0 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<1x64x56x56xf16>
    %37 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %38 = call @aten.expand.164(%37) : (tensor<f16>) -> tensor<1x64x56x56xf16>
    %39 = call @aten.mul.305(%25, %38) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %40 = call @aten.add.351(%36, %39) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %41 = call @aten.relu.339(%40) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %42 = "mhlo.convert"(%arg23) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %43 = call @aten.convolution_overrideable.375(%41, %42) : (tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<1x128x28x28xf16>
    %44 = call @aten.native_batch_norm.380(%43, %arg25, %arg24, %arg84, %arg85) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %45 = "mhlo.get_tuple_element"(%44) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %46 = "mhlo.convert"(%arg21) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %47 = call @aten.convolution_overrideable.409(%41, %46) : (tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<1x128x28x28xf16>
    %48 = call @aten.native_batch_norm.380(%47, %arg18, %arg17, %arg78, %arg79) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %49 = "mhlo.get_tuple_element"(%48) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %50 = "mhlo.get_tuple_element"(%48) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %51 = call @aten.relu.419(%50) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %52 = "mhlo.convert"(%arg22) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %53 = call @aten.convolution_overrideable.425(%51, %52) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %54 = call @aten.native_batch_norm.380(%53, %arg20, %arg19, %arg81, %arg82) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %55 = "mhlo.get_tuple_element"(%54) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %56 = "mhlo.get_tuple_element"(%54) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %57 = "mhlo.get_tuple_element"(%44) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %58 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %59 = call @aten.expand.153(%58) : (tensor<f16>) -> tensor<1x128x28x28xf16>
    %60 = call @aten.mul.402(%57, %59) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %61 = call @aten.add.435(%56, %60) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %62 = call @aten.relu.419(%61) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %63 = "mhlo.convert"(%arg30) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %64 = call @aten.convolution_overrideable.425(%62, %63) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %65 = call @aten.native_batch_norm.380(%64, %arg27, %arg26, %arg87, %arg88) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %66 = "mhlo.get_tuple_element"(%65) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %67 = "mhlo.get_tuple_element"(%65) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %68 = call @aten.relu.419(%67) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %69 = "mhlo.convert"(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %70 = call @aten.convolution_overrideable.425(%68, %69) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %71 = call @aten.native_batch_norm.380(%70, %arg29, %arg28, %arg90, %arg91) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %72 = "mhlo.get_tuple_element"(%71) {index = 3 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %73 = "mhlo.get_tuple_element"(%71) {index = 0 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x28x28xf16>
    %74 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %75 = call @aten.expand.153(%74) : (tensor<f16>) -> tensor<1x128x28x28xf16>
    %76 = call @aten.mul.402(%62, %75) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %77 = call @aten.add.435(%73, %76) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %78 = call @aten.relu.419(%77) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %79 = "mhlo.convert"(%arg38) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %80 = call @aten.convolution_overrideable.459(%78, %79) : (tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<1x256x14x14xf16>
    %81 = call @aten.native_batch_norm.464(%80, %arg40, %arg39, %arg99, %arg100) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %82 = "mhlo.get_tuple_element"(%81) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %83 = "mhlo.convert"(%arg36) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %84 = call @aten.convolution_overrideable.493(%78, %83) : (tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<1x256x14x14xf16>
    %85 = call @aten.native_batch_norm.464(%84, %arg33, %arg32, %arg93, %arg94) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %86 = "mhlo.get_tuple_element"(%85) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %87 = "mhlo.get_tuple_element"(%85) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %88 = call @aten.relu.503(%87) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %89 = "mhlo.convert"(%arg37) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %90 = call @aten.convolution_overrideable.509(%88, %89) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %91 = call @aten.native_batch_norm.464(%90, %arg35, %arg34, %arg96, %arg97) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %92 = "mhlo.get_tuple_element"(%91) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %93 = "mhlo.get_tuple_element"(%91) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %94 = "mhlo.get_tuple_element"(%81) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %95 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %96 = call @aten.expand.142(%95) : (tensor<f16>) -> tensor<1x256x14x14xf16>
    %97 = call @aten.mul.486(%94, %96) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %98 = call @aten.add.519(%93, %97) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %99 = call @aten.relu.503(%98) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %100 = "mhlo.convert"(%arg45) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %101 = call @aten.convolution_overrideable.509(%99, %100) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %102 = call @aten.native_batch_norm.464(%101, %arg42, %arg41, %arg102, %arg103) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %103 = "mhlo.get_tuple_element"(%102) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %104 = "mhlo.get_tuple_element"(%102) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %105 = call @aten.relu.503(%104) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %106 = "mhlo.convert"(%arg46) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %107 = call @aten.convolution_overrideable.509(%105, %106) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %108 = call @aten.native_batch_norm.464(%107, %arg44, %arg43, %arg105, %arg106) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %109 = "mhlo.get_tuple_element"(%108) {index = 3 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %110 = "mhlo.get_tuple_element"(%108) {index = 0 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<1x256x14x14xf16>
    %111 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %112 = call @aten.expand.142(%111) : (tensor<f16>) -> tensor<1x256x14x14xf16>
    %113 = call @aten.mul.486(%99, %112) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %114 = call @aten.add.519(%110, %113) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %115 = call @aten.relu.503(%114) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %116 = "mhlo.convert"(%arg53) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %117 = call @aten.convolution_overrideable.543(%115, %116) : (tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<1x512x7x7xf16>
    %118 = call @aten.native_batch_norm.548(%117, %arg55, %arg54, %arg114, %arg115) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %119 = "mhlo.get_tuple_element"(%118) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %120 = "mhlo.convert"(%arg51) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %121 = call @aten.convolution_overrideable.577(%115, %120) : (tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<1x512x7x7xf16>
    %122 = call @aten.native_batch_norm.548(%121, %arg48, %arg47, %arg108, %arg109) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %123 = "mhlo.get_tuple_element"(%122) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %124 = "mhlo.get_tuple_element"(%122) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %125 = call @aten.relu.587(%124) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %126 = "mhlo.convert"(%arg52) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %127 = call @aten.convolution_overrideable.593(%125, %126) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %128 = call @aten.native_batch_norm.548(%127, %arg50, %arg49, %arg111, %arg112) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %129 = "mhlo.get_tuple_element"(%128) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %130 = "mhlo.get_tuple_element"(%128) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %131 = "mhlo.get_tuple_element"(%118) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %132 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %133 = call @aten.expand.131(%132) : (tensor<f16>) -> tensor<1x512x7x7xf16>
    %134 = call @aten.mul.570(%131, %133) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %135 = call @aten.add.603(%130, %134) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %136 = call @aten.relu.587(%135) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %137 = "mhlo.convert"(%arg60) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %138 = call @aten.convolution_overrideable.593(%136, %137) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %139 = call @aten.native_batch_norm.548(%138, %arg57, %arg56, %arg117, %arg118) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %140 = "mhlo.get_tuple_element"(%139) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %141 = "mhlo.get_tuple_element"(%139) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %142 = call @aten.relu.587(%141) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %143 = "mhlo.convert"(%arg61) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %144 = call @aten.convolution_overrideable.593(%142, %143) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %145 = call @aten.native_batch_norm.548(%144, %arg59, %arg58, %arg120, %arg121) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %146 = "mhlo.get_tuple_element"(%145) {index = 3 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %147 = "mhlo.get_tuple_element"(%145) {index = 0 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<1x512x7x7xf16>
    %148 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %149 = call @aten.expand.131(%148) : (tensor<f16>) -> tensor<1x512x7x7xf16>
    %150 = call @aten.mul.570(%136, %149) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %151 = call @aten.add.603(%147, %150) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %152 = call @aten.relu.587(%151) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %153 = call @aten.mean.631(%152) : (tensor<1x512x7x7xf16>) -> tensor<1x512x1x1xf16>
    %154 = call @aten.view.648(%153) : (tensor<1x512x1x1xf16>) -> tensor<1x512xf16>
    %155 = "mhlo.convert"(%arg4) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %156 = call @aten.permute.126(%155) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    %157 = "mhlo.convert"(%arg3) : (tensor<1000xf32>) -> tensor<1000xf16>
    %158 = call @aten.addmm.652(%154, %156, %157) : (tensor<1x512xf16>, tensor<512x1000xf16>, tensor<1000xf16>) -> tensor<1x1000xf16>
    %159 = "mhlo.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %160 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %161 = "mhlo.broadcast_in_dim"(%160) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %162 = mhlo.multiply %159, %161 : tensor<64xf32>
    %163 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %164 = mhlo.subtract %163, %160 : tensor<f32>
    %165 = "mhlo.broadcast_in_dim"(%164) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %166 = mhlo.multiply %arg63, %165 : tensor<64xf32>
    %167 = mhlo.add %162, %166 : tensor<64xf32>
    %168 = "mhlo.get_tuple_element"(%3) {index = 2 : i32} : (tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %169 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %170 = "mhlo.broadcast_in_dim"(%169) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %171 = mhlo.multiply %168, %170 : tensor<64xf32>
    %172 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %173 = mhlo.subtract %172, %169 : tensor<f32>
    %174 = "mhlo.broadcast_in_dim"(%173) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %175 = mhlo.multiply %arg64, %174 : tensor<64xf32>
    %176 = mhlo.add %171, %175 : tensor<64xf32>
    %177 = "mhlo.get_tuple_element"(%12) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %178 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %179 = "mhlo.broadcast_in_dim"(%178) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %180 = mhlo.multiply %177, %179 : tensor<64xf32>
    %181 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %182 = mhlo.subtract %181, %178 : tensor<f32>
    %183 = "mhlo.broadcast_in_dim"(%182) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %184 = mhlo.multiply %arg66, %183 : tensor<64xf32>
    %185 = mhlo.add %180, %184 : tensor<64xf32>
    %186 = "mhlo.get_tuple_element"(%12) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %187 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %188 = "mhlo.broadcast_in_dim"(%187) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %189 = mhlo.multiply %186, %188 : tensor<64xf32>
    %190 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %191 = mhlo.subtract %190, %187 : tensor<f32>
    %192 = "mhlo.broadcast_in_dim"(%191) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %193 = mhlo.multiply %arg67, %192 : tensor<64xf32>
    %194 = mhlo.add %189, %193 : tensor<64xf32>
    %195 = "mhlo.get_tuple_element"(%18) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %196 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %197 = "mhlo.broadcast_in_dim"(%196) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %198 = mhlo.multiply %195, %197 : tensor<64xf32>
    %199 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %200 = mhlo.subtract %199, %196 : tensor<f32>
    %201 = "mhlo.broadcast_in_dim"(%200) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %202 = mhlo.multiply %arg69, %201 : tensor<64xf32>
    %203 = mhlo.add %198, %202 : tensor<64xf32>
    %204 = "mhlo.get_tuple_element"(%18) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %205 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %206 = "mhlo.broadcast_in_dim"(%205) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %207 = mhlo.multiply %204, %206 : tensor<64xf32>
    %208 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %209 = mhlo.subtract %208, %205 : tensor<f32>
    %210 = "mhlo.broadcast_in_dim"(%209) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %211 = mhlo.multiply %arg70, %210 : tensor<64xf32>
    %212 = mhlo.add %207, %211 : tensor<64xf32>
    %213 = "mhlo.get_tuple_element"(%28) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %214 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %215 = "mhlo.broadcast_in_dim"(%214) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %216 = mhlo.multiply %213, %215 : tensor<64xf32>
    %217 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %218 = mhlo.subtract %217, %214 : tensor<f32>
    %219 = "mhlo.broadcast_in_dim"(%218) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %220 = mhlo.multiply %arg72, %219 : tensor<64xf32>
    %221 = mhlo.add %216, %220 : tensor<64xf32>
    %222 = "mhlo.get_tuple_element"(%28) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %223 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %224 = "mhlo.broadcast_in_dim"(%223) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %225 = mhlo.multiply %222, %224 : tensor<64xf32>
    %226 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %227 = mhlo.subtract %226, %223 : tensor<f32>
    %228 = "mhlo.broadcast_in_dim"(%227) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %229 = mhlo.multiply %arg73, %228 : tensor<64xf32>
    %230 = mhlo.add %225, %229 : tensor<64xf32>
    %231 = "mhlo.get_tuple_element"(%34) {index = 1 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %232 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %233 = "mhlo.broadcast_in_dim"(%232) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %234 = mhlo.multiply %231, %233 : tensor<64xf32>
    %235 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %236 = mhlo.subtract %235, %232 : tensor<f32>
    %237 = "mhlo.broadcast_in_dim"(%236) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %238 = mhlo.multiply %arg75, %237 : tensor<64xf32>
    %239 = mhlo.add %234, %238 : tensor<64xf32>
    %240 = "mhlo.get_tuple_element"(%34) {index = 2 : i32} : (tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %241 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %242 = "mhlo.broadcast_in_dim"(%241) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %243 = mhlo.multiply %240, %242 : tensor<64xf32>
    %244 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %245 = mhlo.subtract %244, %241 : tensor<f32>
    %246 = "mhlo.broadcast_in_dim"(%245) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %247 = mhlo.multiply %arg76, %246 : tensor<64xf32>
    %248 = mhlo.add %243, %247 : tensor<64xf32>
    %249 = "mhlo.get_tuple_element"(%48) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %250 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %251 = "mhlo.broadcast_in_dim"(%250) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %252 = mhlo.multiply %249, %251 : tensor<128xf32>
    %253 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %254 = mhlo.subtract %253, %250 : tensor<f32>
    %255 = "mhlo.broadcast_in_dim"(%254) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %256 = mhlo.multiply %arg78, %255 : tensor<128xf32>
    %257 = mhlo.add %252, %256 : tensor<128xf32>
    %258 = "mhlo.get_tuple_element"(%48) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %259 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %260 = "mhlo.broadcast_in_dim"(%259) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %261 = mhlo.multiply %258, %260 : tensor<128xf32>
    %262 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %263 = mhlo.subtract %262, %259 : tensor<f32>
    %264 = "mhlo.broadcast_in_dim"(%263) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %265 = mhlo.multiply %arg79, %264 : tensor<128xf32>
    %266 = mhlo.add %261, %265 : tensor<128xf32>
    %267 = "mhlo.get_tuple_element"(%54) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %268 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %269 = "mhlo.broadcast_in_dim"(%268) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %270 = mhlo.multiply %267, %269 : tensor<128xf32>
    %271 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %272 = mhlo.subtract %271, %268 : tensor<f32>
    %273 = "mhlo.broadcast_in_dim"(%272) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %274 = mhlo.multiply %arg81, %273 : tensor<128xf32>
    %275 = mhlo.add %270, %274 : tensor<128xf32>
    %276 = "mhlo.get_tuple_element"(%54) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %277 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %278 = "mhlo.broadcast_in_dim"(%277) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %279 = mhlo.multiply %276, %278 : tensor<128xf32>
    %280 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %281 = mhlo.subtract %280, %277 : tensor<f32>
    %282 = "mhlo.broadcast_in_dim"(%281) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %283 = mhlo.multiply %arg82, %282 : tensor<128xf32>
    %284 = mhlo.add %279, %283 : tensor<128xf32>
    %285 = "mhlo.get_tuple_element"(%44) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %286 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %287 = "mhlo.broadcast_in_dim"(%286) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %288 = mhlo.multiply %285, %287 : tensor<128xf32>
    %289 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %290 = mhlo.subtract %289, %286 : tensor<f32>
    %291 = "mhlo.broadcast_in_dim"(%290) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %292 = mhlo.multiply %arg84, %291 : tensor<128xf32>
    %293 = mhlo.add %288, %292 : tensor<128xf32>
    %294 = "mhlo.get_tuple_element"(%44) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %295 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %296 = "mhlo.broadcast_in_dim"(%295) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %297 = mhlo.multiply %294, %296 : tensor<128xf32>
    %298 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %299 = mhlo.subtract %298, %295 : tensor<f32>
    %300 = "mhlo.broadcast_in_dim"(%299) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %301 = mhlo.multiply %arg85, %300 : tensor<128xf32>
    %302 = mhlo.add %297, %301 : tensor<128xf32>
    %303 = "mhlo.get_tuple_element"(%65) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %304 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %305 = "mhlo.broadcast_in_dim"(%304) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %306 = mhlo.multiply %303, %305 : tensor<128xf32>
    %307 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %308 = mhlo.subtract %307, %304 : tensor<f32>
    %309 = "mhlo.broadcast_in_dim"(%308) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %310 = mhlo.multiply %arg87, %309 : tensor<128xf32>
    %311 = mhlo.add %306, %310 : tensor<128xf32>
    %312 = "mhlo.get_tuple_element"(%65) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %313 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %314 = "mhlo.broadcast_in_dim"(%313) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %315 = mhlo.multiply %312, %314 : tensor<128xf32>
    %316 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %317 = mhlo.subtract %316, %313 : tensor<f32>
    %318 = "mhlo.broadcast_in_dim"(%317) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %319 = mhlo.multiply %arg88, %318 : tensor<128xf32>
    %320 = mhlo.add %315, %319 : tensor<128xf32>
    %321 = "mhlo.get_tuple_element"(%71) {index = 1 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %322 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %323 = "mhlo.broadcast_in_dim"(%322) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %324 = mhlo.multiply %321, %323 : tensor<128xf32>
    %325 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %326 = mhlo.subtract %325, %322 : tensor<f32>
    %327 = "mhlo.broadcast_in_dim"(%326) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %328 = mhlo.multiply %arg90, %327 : tensor<128xf32>
    %329 = mhlo.add %324, %328 : tensor<128xf32>
    %330 = "mhlo.get_tuple_element"(%71) {index = 2 : i32} : (tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %331 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %332 = "mhlo.broadcast_in_dim"(%331) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %333 = mhlo.multiply %330, %332 : tensor<128xf32>
    %334 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %335 = mhlo.subtract %334, %331 : tensor<f32>
    %336 = "mhlo.broadcast_in_dim"(%335) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %337 = mhlo.multiply %arg91, %336 : tensor<128xf32>
    %338 = mhlo.add %333, %337 : tensor<128xf32>
    %339 = "mhlo.get_tuple_element"(%85) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %340 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %341 = "mhlo.broadcast_in_dim"(%340) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %342 = mhlo.multiply %339, %341 : tensor<256xf32>
    %343 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %344 = mhlo.subtract %343, %340 : tensor<f32>
    %345 = "mhlo.broadcast_in_dim"(%344) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %346 = mhlo.multiply %arg93, %345 : tensor<256xf32>
    %347 = mhlo.add %342, %346 : tensor<256xf32>
    %348 = "mhlo.get_tuple_element"(%85) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %349 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %350 = "mhlo.broadcast_in_dim"(%349) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %351 = mhlo.multiply %348, %350 : tensor<256xf32>
    %352 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %353 = mhlo.subtract %352, %349 : tensor<f32>
    %354 = "mhlo.broadcast_in_dim"(%353) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %355 = mhlo.multiply %arg94, %354 : tensor<256xf32>
    %356 = mhlo.add %351, %355 : tensor<256xf32>
    %357 = "mhlo.get_tuple_element"(%91) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %358 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %359 = "mhlo.broadcast_in_dim"(%358) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %360 = mhlo.multiply %357, %359 : tensor<256xf32>
    %361 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %362 = mhlo.subtract %361, %358 : tensor<f32>
    %363 = "mhlo.broadcast_in_dim"(%362) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %364 = mhlo.multiply %arg96, %363 : tensor<256xf32>
    %365 = mhlo.add %360, %364 : tensor<256xf32>
    %366 = "mhlo.get_tuple_element"(%91) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %367 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %368 = "mhlo.broadcast_in_dim"(%367) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %369 = mhlo.multiply %366, %368 : tensor<256xf32>
    %370 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %371 = mhlo.subtract %370, %367 : tensor<f32>
    %372 = "mhlo.broadcast_in_dim"(%371) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %373 = mhlo.multiply %arg97, %372 : tensor<256xf32>
    %374 = mhlo.add %369, %373 : tensor<256xf32>
    %375 = "mhlo.get_tuple_element"(%81) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %376 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %377 = "mhlo.broadcast_in_dim"(%376) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %378 = mhlo.multiply %375, %377 : tensor<256xf32>
    %379 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %380 = mhlo.subtract %379, %376 : tensor<f32>
    %381 = "mhlo.broadcast_in_dim"(%380) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %382 = mhlo.multiply %arg99, %381 : tensor<256xf32>
    %383 = mhlo.add %378, %382 : tensor<256xf32>
    %384 = "mhlo.get_tuple_element"(%81) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %385 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %386 = "mhlo.broadcast_in_dim"(%385) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %387 = mhlo.multiply %384, %386 : tensor<256xf32>
    %388 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %389 = mhlo.subtract %388, %385 : tensor<f32>
    %390 = "mhlo.broadcast_in_dim"(%389) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %391 = mhlo.multiply %arg100, %390 : tensor<256xf32>
    %392 = mhlo.add %387, %391 : tensor<256xf32>
    %393 = "mhlo.get_tuple_element"(%102) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %394 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %395 = "mhlo.broadcast_in_dim"(%394) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %396 = mhlo.multiply %393, %395 : tensor<256xf32>
    %397 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %398 = mhlo.subtract %397, %394 : tensor<f32>
    %399 = "mhlo.broadcast_in_dim"(%398) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %400 = mhlo.multiply %arg102, %399 : tensor<256xf32>
    %401 = mhlo.add %396, %400 : tensor<256xf32>
    %402 = "mhlo.get_tuple_element"(%102) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %403 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %404 = "mhlo.broadcast_in_dim"(%403) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %405 = mhlo.multiply %402, %404 : tensor<256xf32>
    %406 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %407 = mhlo.subtract %406, %403 : tensor<f32>
    %408 = "mhlo.broadcast_in_dim"(%407) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %409 = mhlo.multiply %arg103, %408 : tensor<256xf32>
    %410 = mhlo.add %405, %409 : tensor<256xf32>
    %411 = "mhlo.get_tuple_element"(%108) {index = 1 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %412 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %413 = "mhlo.broadcast_in_dim"(%412) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %414 = mhlo.multiply %411, %413 : tensor<256xf32>
    %415 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %416 = mhlo.subtract %415, %412 : tensor<f32>
    %417 = "mhlo.broadcast_in_dim"(%416) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %418 = mhlo.multiply %arg105, %417 : tensor<256xf32>
    %419 = mhlo.add %414, %418 : tensor<256xf32>
    %420 = "mhlo.get_tuple_element"(%108) {index = 2 : i32} : (tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %421 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %422 = "mhlo.broadcast_in_dim"(%421) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %423 = mhlo.multiply %420, %422 : tensor<256xf32>
    %424 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %425 = mhlo.subtract %424, %421 : tensor<f32>
    %426 = "mhlo.broadcast_in_dim"(%425) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %427 = mhlo.multiply %arg106, %426 : tensor<256xf32>
    %428 = mhlo.add %423, %427 : tensor<256xf32>
    %429 = "mhlo.get_tuple_element"(%122) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %430 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %431 = "mhlo.broadcast_in_dim"(%430) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %432 = mhlo.multiply %429, %431 : tensor<512xf32>
    %433 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %434 = mhlo.subtract %433, %430 : tensor<f32>
    %435 = "mhlo.broadcast_in_dim"(%434) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %436 = mhlo.multiply %arg108, %435 : tensor<512xf32>
    %437 = mhlo.add %432, %436 : tensor<512xf32>
    %438 = "mhlo.get_tuple_element"(%122) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %439 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %440 = "mhlo.broadcast_in_dim"(%439) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %441 = mhlo.multiply %438, %440 : tensor<512xf32>
    %442 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %443 = mhlo.subtract %442, %439 : tensor<f32>
    %444 = "mhlo.broadcast_in_dim"(%443) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %445 = mhlo.multiply %arg109, %444 : tensor<512xf32>
    %446 = mhlo.add %441, %445 : tensor<512xf32>
    %447 = "mhlo.get_tuple_element"(%128) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %448 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %449 = "mhlo.broadcast_in_dim"(%448) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %450 = mhlo.multiply %447, %449 : tensor<512xf32>
    %451 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %452 = mhlo.subtract %451, %448 : tensor<f32>
    %453 = "mhlo.broadcast_in_dim"(%452) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %454 = mhlo.multiply %arg111, %453 : tensor<512xf32>
    %455 = mhlo.add %450, %454 : tensor<512xf32>
    %456 = "mhlo.get_tuple_element"(%128) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %457 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %458 = "mhlo.broadcast_in_dim"(%457) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %459 = mhlo.multiply %456, %458 : tensor<512xf32>
    %460 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %461 = mhlo.subtract %460, %457 : tensor<f32>
    %462 = "mhlo.broadcast_in_dim"(%461) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %463 = mhlo.multiply %arg112, %462 : tensor<512xf32>
    %464 = mhlo.add %459, %463 : tensor<512xf32>
    %465 = "mhlo.get_tuple_element"(%118) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %466 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %467 = "mhlo.broadcast_in_dim"(%466) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %468 = mhlo.multiply %465, %467 : tensor<512xf32>
    %469 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %470 = mhlo.subtract %469, %466 : tensor<f32>
    %471 = "mhlo.broadcast_in_dim"(%470) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %472 = mhlo.multiply %arg114, %471 : tensor<512xf32>
    %473 = mhlo.add %468, %472 : tensor<512xf32>
    %474 = "mhlo.get_tuple_element"(%118) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %475 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %476 = "mhlo.broadcast_in_dim"(%475) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %477 = mhlo.multiply %474, %476 : tensor<512xf32>
    %478 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %479 = mhlo.subtract %478, %475 : tensor<f32>
    %480 = "mhlo.broadcast_in_dim"(%479) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %481 = mhlo.multiply %arg115, %480 : tensor<512xf32>
    %482 = mhlo.add %477, %481 : tensor<512xf32>
    %483 = "mhlo.get_tuple_element"(%139) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %484 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %485 = "mhlo.broadcast_in_dim"(%484) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %486 = mhlo.multiply %483, %485 : tensor<512xf32>
    %487 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %488 = mhlo.subtract %487, %484 : tensor<f32>
    %489 = "mhlo.broadcast_in_dim"(%488) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %490 = mhlo.multiply %arg117, %489 : tensor<512xf32>
    %491 = mhlo.add %486, %490 : tensor<512xf32>
    %492 = "mhlo.get_tuple_element"(%139) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %493 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %494 = "mhlo.broadcast_in_dim"(%493) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %495 = mhlo.multiply %492, %494 : tensor<512xf32>
    %496 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %497 = mhlo.subtract %496, %493 : tensor<f32>
    %498 = "mhlo.broadcast_in_dim"(%497) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %499 = mhlo.multiply %arg118, %498 : tensor<512xf32>
    %500 = mhlo.add %495, %499 : tensor<512xf32>
    %501 = "mhlo.get_tuple_element"(%145) {index = 1 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %502 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %503 = "mhlo.broadcast_in_dim"(%502) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %504 = mhlo.multiply %501, %503 : tensor<512xf32>
    %505 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %506 = mhlo.subtract %505, %502 : tensor<f32>
    %507 = "mhlo.broadcast_in_dim"(%506) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %508 = mhlo.multiply %arg120, %507 : tensor<512xf32>
    %509 = mhlo.add %504, %508 : tensor<512xf32>
    %510 = "mhlo.get_tuple_element"(%145) {index = 2 : i32} : (tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %511 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %512 = "mhlo.broadcast_in_dim"(%511) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %513 = mhlo.multiply %510, %512 : tensor<512xf32>
    %514 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %515 = mhlo.subtract %514, %511 : tensor<f32>
    %516 = "mhlo.broadcast_in_dim"(%515) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %517 = mhlo.multiply %arg121, %516 : tensor<512xf32>
    %518 = mhlo.add %513, %517 : tensor<512xf32>
    %519 = "mhlo.tuple"(%158, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %167, %176, %185, %194, %203, %212, %221, %230, %239, %248, %257, %266, %275, %284, %293, %302, %311, %320, %329, %338, %347, %356, %365, %374, %383, %392, %401, %410, %419, %428, %437, %446, %455, %464, %473, %482, %491, %500, %509, %518, %1, %0, %2, %6, %9, %10, %11, %15, %16, %17, %25, %26, %27, %31, %32, %33, %41, %46, %47, %51, %52, %53, %42, %43, %62, %63, %64, %68, %69, %70, %78, %83, %84, %88, %89, %90, %79, %80, %99, %100, %101, %105, %106, %107, %115, %120, %121, %125, %126, %127, %116, %117, %136, %137, %138, %142, %143, %144, %152, %154, %156) : (tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>) -> !tuple
    return %519 : !tuple
  }
  func.func private @aten.convolution_overrideable.175(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<64x3x7x7xf16>) -> tensor<1x64x112x112xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<1x64x112x112xf16>
    return %0 : tensor<1x64x112x112xf16>
  }
  func.func private @aten.native_batch_norm.180(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
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
  func.func private @aten.relu.202(%arg0: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x64x112x112xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<1x64x112x112xf16>
    return %2 : tensor<1x64x112x112xf16>
  }
  func.func private @aten.max_pool2d.277(%arg0: tensor<1x64x112x112xf16>) -> tuple<tensor<1x64x56x56xf16>, tensor<1x64x56x56xui32>> {
    %0 = mhlo.constant dense<0> : tensor<ui32>
    %1 = mhlo.constant dense<200704> : tensor<ui32>
    %2 = mhlo.constant dense<0xFC00> : tensor<f16>
    %3 = "mhlo.pad"(%arg0, %2) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x64x112x112xf16>, tensor<f16>) -> tensor<1x64x114x114xf16>
    %4 = mhlo.constant dense<0xFC00> : tensor<f16>
    %5 = "mhlo.reduce_window"(%3, %4) ( {
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):  // no predecessors
      %23 = mhlo.maximum %arg1, %arg2 : tensor<f16>
      "mhlo.return"(%23) : (tensor<f16>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x64x114x114xf16>, tensor<f16>) -> tensor<1x64x56x56xf16>
    %6 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<12544xui32>
    %7 = "mhlo.reshape"(%6) : (tensor<12544xui32>) -> tensor<112x112xui32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<112x112xui32>) -> tensor<1x64x112x112xui32>
    %9 = mhlo.constant dense<4294967295> : tensor<ui32>
    %10 = "mhlo.pad"(%8, %9) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x64x112x112xui32>, tensor<ui32>) -> tensor<1x64x114x114xui32>
    %11 = mhlo.constant dense<0> : tensor<ui32>
    %12 = "mhlo.broadcast_in_dim"(%11) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<ui32>) -> tensor<200704xui32>
    %13 = "mhlo.tuple"(%0, %1, %3, %5, %10, %12) : (tensor<ui32>, tensor<ui32>, tensor<1x64x114x114xf16>, tensor<1x64x56x56xf16>, tensor<1x64x114x114xui32>, tensor<200704xui32>) -> tuple<tensor<ui32>, tensor<ui32>, tensor<1x64x114x114xf16>, tensor<1x64x56x56xf16>, tensor<1x64x114x114xui32>, tensor<200704xui32>>
    %14:6 = "mhlo.while"(%0, %1, %3, %5, %10, %12) ( {
    ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>, %arg3: tensor<1x64x114x114xf16>, %arg4: tensor<1x64x56x56xf16>, %arg5: tensor<1x64x114x114xui32>, %arg6: tensor<200704xui32>):  // no predecessors
      %29 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
      "mhlo.return"(%29) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>, %arg3: tensor<1x64x114x114xf16>, %arg4: tensor<1x64x56x56xf16>, %arg5: tensor<1x64x114x114xui32>, %arg6: tensor<200704xui32>):  // no predecessors
      %24 = mhlo.constant dense<200704> : tensor<ui32>
      %25 = mhlo.remainder %arg1, %24 : tensor<ui32>
      %26 = mhlo.constant dense<3136> : tensor<ui32>
      %27 = mhlo.remainder %25, %26 : tensor<ui32>
      %28 = mhlo.constant dense<56> : tensor<ui32>
      %29 = mhlo.remainder %27, %28 : tensor<ui32>
      %30 = mhlo.constant dense<1> : tensor<ui32>
      %31 = mhlo.remainder %29, %30 : tensor<ui32>
      %32 = mhlo.constant dense<1> : tensor<ui32>
      %33 = mhlo.add %arg1, %32 : tensor<ui32>
      %39 = mhlo.constant dense<1> : tensor<ui32>
      %40 = mhlo.divide %arg1, %24 : tensor<ui32>
      %41 = mhlo.multiply %39, %40 : tensor<ui32>
      %42 = mhlo.constant dense<1> : tensor<ui32>
      %43 = mhlo.divide %25, %26 : tensor<ui32>
      %44 = mhlo.multiply %42, %43 : tensor<ui32>
      %45 = mhlo.constant dense<2> : tensor<ui32>
      %46 = mhlo.divide %27, %28 : tensor<ui32>
      %47 = mhlo.multiply %45, %46 : tensor<ui32>
      %48 = mhlo.constant dense<2> : tensor<ui32>
      %49 = mhlo.divide %29, %30 : tensor<ui32>
      %50 = mhlo.multiply %48, %49 : tensor<ui32>
      %51 = "mhlo.dynamic_slice"(%arg3, %41, %44, %47, %50) {slice_sizes = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<1x64x114x114xf16>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x3x3xf16>
      %52 = "mhlo.dynamic_slice"(%arg4, %40, %43, %46, %49) {slice_sizes = dense<1> : tensor<4xi64>} : (tensor<1x64x56x56xf16>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x1x1xf16>
      %53 = mhlo.constant dense<0xFC00> : tensor<f16>
      %54 = "mhlo.select_and_scatter"(%51, %52, %53) ( {
      ^bb0(%arg7: tensor<f16>, %arg8: tensor<f16>):  // no predecessors
        %65 = "mhlo.compare"(%arg7, %arg8) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f16>, tensor<f16>) -> tensor<i1>
        "mhlo.return"(%65) : (tensor<i1>) -> ()
      },  {
      ^bb0(%arg7: tensor<f16>, %arg8: tensor<f16>):  // no predecessors
        %65 = mhlo.maximum %arg7, %arg8 : tensor<f16>
        "mhlo.return"(%65) : (tensor<f16>) -> ()
      }) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<f16>) -> tensor<1x1x3x3xf16>
      %55 = "mhlo.broadcast_in_dim"(%53) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x1x3x3xf16>
      %56 = "mhlo.compare"(%54, %55) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<1x1x3x3xf16>, tensor<1x1x3x3xf16>) -> tensor<1x1x3x3xi1>
      %57 = "mhlo.dynamic_slice"(%arg5, %41, %44, %47, %50) {slice_sizes = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<1x64x114x114xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x3x3xui32>
      %58 = mhlo.constant dense<4294967295> : tensor<ui32>
      %59 = "mhlo.broadcast_in_dim"(%58) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<ui32>) -> tensor<1x1x3x3xui32>
      %60 = "mhlo.select"(%56, %57, %59) : (tensor<1x1x3x3xi1>, tensor<1x1x3x3xui32>, tensor<1x1x3x3xui32>) -> tensor<1x1x3x3xui32>
      %61 = "mhlo.reduce_window"(%60, %58) ( {
      ^bb0(%arg7: tensor<ui32>, %arg8: tensor<ui32>):  // no predecessors
        %65 = mhlo.minimum %arg7, %arg8 : tensor<ui32>
        "mhlo.return"(%65) : (tensor<ui32>) -> ()
      }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x1x3x3xui32>, tensor<ui32>) -> tensor<1x1x1x1xui32>
      %62 = "mhlo.reshape"(%61) : (tensor<1x1x1x1xui32>) -> tensor<1xui32>
      %63 = "mhlo.dynamic_update_slice"(%arg6, %62, %arg1) : (tensor<200704xui32>, tensor<1xui32>, tensor<ui32>) -> tensor<200704xui32>
      "mhlo.return"(%33, %arg2, %arg3, %arg4, %arg5, %63) : (tensor<ui32>, tensor<ui32>, tensor<1x64x114x114xf16>, tensor<1x64x56x56xf16>, tensor<1x64x114x114xui32>, tensor<200704xui32>) -> ()
    }) : (tensor<ui32>, tensor<ui32>, tensor<1x64x114x114xf16>, tensor<1x64x56x56xf16>, tensor<1x64x114x114xui32>, tensor<200704xui32>) -> (tensor<ui32>, tensor<ui32>, tensor<1x64x114x114xf16>, tensor<1x64x56x56xf16>, tensor<1x64x114x114xui32>, tensor<200704xui32>)
    %21 = "mhlo.reshape"(%14#5) : (tensor<200704xui32>) -> tensor<1x64x56x56xui32>
    %22 = "mhlo.tuple"(%5, %21) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xui32>) -> tuple<tensor<1x64x56x56xf16>, tensor<1x64x56x56xui32>>
    return %22 : tuple<tensor<1x64x56x56xf16>, tensor<1x64x56x56xui32>>
  }
  func.func private @aten.convolution_overrideable.312(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    return %0 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm.317(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
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
  func.func private @aten.relu.339(%arg0: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x64x56x56xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.expand.164(%arg0: tensor<f16>) -> tensor<1x64x56x56xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<1xf16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf16>) -> tensor<1x64x56x56xf16>
    return %3 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.mul.305(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x64x56x56xf16>
    return %0 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.add.351(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x64x56x56xf16>
    return %0 : tensor<1x64x56x56xf16>
  }
  func.func private @aten.convolution_overrideable.375(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<1x128x28x28xf16>
    return %0 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm.380(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
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
  func.func private @aten.convolution_overrideable.409(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<1x128x28x28xf16>
    return %0 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.relu.419(%arg0: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x128x28x28xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.convolution_overrideable.425(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    return %0 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.expand.153(%arg0: tensor<f16>) -> tensor<1x128x28x28xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<1xf16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf16>) -> tensor<1x128x28x28xf16>
    return %3 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.mul.402(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x128x28x28xf16>
    return %0 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.add.435(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x128x28x28xf16>
    return %0 : tensor<1x128x28x28xf16>
  }
  func.func private @aten.convolution_overrideable.459(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<1x256x14x14xf16>
    return %0 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm.464(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
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
  func.func private @aten.convolution_overrideable.493(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<1x256x14x14xf16>
    return %0 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.relu.503(%arg0: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x256x14x14xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.convolution_overrideable.509(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    return %0 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.expand.142(%arg0: tensor<f16>) -> tensor<1x256x14x14xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<1xf16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf16>) -> tensor<1x256x14x14xf16>
    return %3 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.mul.486(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x256x14x14xf16>
    return %0 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.add.519(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x256x14x14xf16>
    return %0 : tensor<1x256x14x14xf16>
  }
  func.func private @aten.convolution_overrideable.543(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<1x512x7x7xf16>
    return %0 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm.548(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
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
  func.func private @aten.convolution_overrideable.577(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<1x512x7x7xf16>
    return %0 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.relu.587(%arg0: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x512x7x7xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.convolution_overrideable.593(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    return %0 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.expand.131(%arg0: tensor<f16>) -> tensor<1x512x7x7xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<1xf16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf16>) -> tensor<1x512x7x7xf16>
    return %3 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.mul.570(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x512x7x7xf16>
    return %0 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.add.603(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x512x7x7xf16>
    return %0 : tensor<1x512x7x7xf16>
  }
  func.func private @aten.mean.631(%arg0: tensor<1x512x7x7xf16>) -> tensor<1x512x1x1xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.reduce"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):  // no predecessors
      %14 = mhlo.add %arg1, %arg2 : tensor<f16>
      "mhlo.return"(%14) : (tensor<f16>) -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (tensor<1x512x7x7xf16>, tensor<f16>) -> tensor<1x512xf16>
    %2 = mhlo.constant dense<49> : tensor<i64>
    %3 = mhlo.constant dense<0> : tensor<i64>
    %4 = "mhlo.compare"(%2, %3) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %5 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %6 = "mhlo.convert"(%2) : (tensor<i64>) -> tensor<f16>
    %7 = mhlo.divide %5, %6 : tensor<f16>
    %8 = mhlo.constant dense<0x7E00> : tensor<f16>
    %9 = "mhlo.select"(%4, %7, %8) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
    %10 = "mhlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x512xf16>
    %11 = mhlo.multiply %1, %10 : tensor<1x512xf16>
    %12 = "mhlo.reshape"(%11) : (tensor<1x512xf16>) -> tensor<1x512x1x1xf16>
    %13 = "mhlo.convert"(%12) : (tensor<1x512x1x1xf16>) -> tensor<1x512x1x1xf16>
    return %13 : tensor<1x512x1x1xf16>
  }
  func.func private @aten.view.648(%arg0: tensor<1x512x1x1xf16>) -> tensor<1x512xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x512x1x1xf16>) -> tensor<1x512xf16>
    return %0 : tensor<1x512xf16>
  }
  func.func private @aten.permute.126(%arg0: tensor<1000x512xf16>) -> tensor<512x1000xf16> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    return %0 : tensor<512x1000xf16>
  }
  func.func private @aten.addmm.652(%arg0: tensor<1x512xf16>, %arg1: tensor<512x1000xf16>, %arg2: tensor<1000xf16>) -> tensor<1x1000xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512xf16>, tensor<512x1000xf16>) -> tensor<1x1000xf16>
    %1 = "mhlo.reshape"(%arg2) : (tensor<1000xf16>) -> tensor<1x1000xf16>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1000xf16>) -> tensor<1x1000xf16>
    %3 = mhlo.add %0, %2 : tensor<1x1000xf16>
    return %3 : tensor<1x1000xf16>
  }
}""")], pipelines=[
    InputPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    HloOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    LinalgTensorOptPipeline(R"""
// CHECK-LABEL: func.func @main
"""),
    ByreTensorOptPipeline(R"""
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
    SetSpaceOptPipeline(R"""
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