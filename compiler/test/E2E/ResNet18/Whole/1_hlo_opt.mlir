// RUN: byteir-opt %s -hlo-opt="outline-single-elemwise-op" | FileCheck %s

// CHECK-LABEL: func.func @main
!tuple = tuple<tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>>
module @IrToMhlo.2452 {
  func.func @main(%arg0: tensor<4x3x224x224xf32>, %arg1: tensor<4x1000xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64x64x3x3xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64x64x3x3xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64x64x3x3xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64xf32>, %arg22: tensor<64x64x3x3xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<128x64x3x3xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128xf32>, %arg32: tensor<128x128x3x3xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128x64x1x1xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128x3x3xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128x128x3x3xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<256x128x3x3xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256xf32>, %arg57: tensor<256x256x3x3xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256x128x1x1xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256x256x3x3xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<256x256x3x3xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<256xf32>, %arg77: tensor<512x256x3x3xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512xf32>, %arg82: tensor<512x512x3x3xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512x256x1x1xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512xf32>, %arg91: tensor<512xf32>, %arg92: tensor<512x512x3x3xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512xf32>, %arg97: tensor<512x512x3x3xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<512xf32>, %arg102: tensor<1000x512xf32>, %arg103: tensor<1000xf32>) -> !tuple {
    %0 = mhlo.convert %arg0 : (tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16>
    %1 = mhlo.convert %arg2 : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %2 = call @aten.convolution_overrideable.181(%0, %1) : (tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<4x64x112x112xf16>
    %3 = call @aten.native_batch_norm.186(%2, %arg3, %arg4, %arg5, %arg6) {xla_shape = "(f16[4,64,112,112]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %4 = mhlo.get_tuple_element %3[2] : (tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %5 = mhlo.get_tuple_element %3[0] : (tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x112x112xf16>
    %6 = call @aten.relu.208(%5) : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %7 = call @aten.max_pool2d.283(%6) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, u32[4,64,56,56]{3,2,1,0})"} : (tensor<4x64x112x112xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<4x64x56x56xui32>>
    %8 = mhlo.get_tuple_element %7[1] : (tuple<tensor<4x64x56x56xf16>, tensor<4x64x56x56xui32>>) -> tensor<4x64x56x56xui32>
    %9 = mhlo.get_tuple_element %7[0] : (tuple<tensor<4x64x56x56xf16>, tensor<4x64x56x56xui32>>) -> tensor<4x64x56x56xf16>
    %10 = mhlo.convert %arg7 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %11 = call @aten.convolution_overrideable.318(%9, %10) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %12 = call @aten.native_batch_norm.323(%11, %arg8, %arg9, %arg10, %arg11) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %13 = mhlo.get_tuple_element %12[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %14 = mhlo.get_tuple_element %12[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x56x56xf16>
    %15 = call @aten.relu.345(%14) : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %16 = mhlo.convert %arg12 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %17 = call @aten.convolution_overrideable.351(%15, %16) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %18 = call @aten.native_batch_norm.356(%17, %arg13, %arg14, %arg15, %arg16) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %19 = mhlo.get_tuple_element %18[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %20 = mhlo.get_tuple_element %18[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x56x56xf16>
    %21 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %22 = call @aten.expand.172(%21) : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %23 = call @aten.mul.311(%9, %22) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %24 = call @aten.add.378(%20, %23) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %25 = call @aten.relu.383(%24) : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %26 = mhlo.convert %arg17 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %27 = call @aten.convolution_overrideable.396(%25, %26) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %28 = call @aten.native_batch_norm.401(%27, %arg18, %arg19, %arg20, %arg21) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %29 = mhlo.get_tuple_element %28[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %30 = mhlo.get_tuple_element %28[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x56x56xf16>
    %31 = call @aten.relu.423(%30) : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %32 = mhlo.convert %arg22 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %33 = call @aten.convolution_overrideable.429(%31, %32) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %34 = call @aten.native_batch_norm.434(%33, %arg23, %arg24, %arg25, %arg26) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %35 = mhlo.get_tuple_element %34[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %36 = mhlo.get_tuple_element %34[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x56x56xf16>
    %37 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %38 = call @aten.expand.164(%37) : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %39 = call @aten.mul.389(%25, %38) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %40 = call @aten.add.456(%36, %39) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %41 = call @aten.relu.461(%40) : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %42 = mhlo.convert %arg37 : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %43 = call @aten.convolution_overrideable.467(%41, %42) : (tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<4x128x28x28xf16>
    %44 = call @aten.native_batch_norm.472(%43, %arg38, %arg39, %arg40, %arg41) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %45 = mhlo.get_tuple_element %44[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %46 = mhlo.convert %arg27 : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %47 = call @aten.convolution_overrideable.501(%41, %46) : (tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<4x128x28x28xf16>
    %48 = call @aten.native_batch_norm.506(%47, %arg28, %arg29, %arg30, %arg31) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %49 = mhlo.get_tuple_element %48[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %50 = mhlo.get_tuple_element %48[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %51 = call @aten.relu.528(%50) : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %52 = mhlo.convert %arg32 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %53 = call @aten.convolution_overrideable.534(%51, %52) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %54 = call @aten.native_batch_norm.539(%53, %arg33, %arg34, %arg35, %arg36) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %55 = mhlo.get_tuple_element %54[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %56 = mhlo.get_tuple_element %54[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %57 = mhlo.get_tuple_element %44[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %58 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %59 = call @aten.expand.155(%58) : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %60 = call @aten.mul.494(%57, %59) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %61 = call @aten.add.561(%56, %60) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %62 = call @aten.relu.566(%61) : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %63 = mhlo.convert %arg42 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %64 = call @aten.convolution_overrideable.579(%62, %63) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %65 = call @aten.native_batch_norm.584(%64, %arg43, %arg44, %arg45, %arg46) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %66 = mhlo.get_tuple_element %65[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %67 = mhlo.get_tuple_element %65[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %68 = call @aten.relu.606(%67) : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %69 = mhlo.convert %arg47 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %70 = call @aten.convolution_overrideable.612(%68, %69) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %71 = call @aten.native_batch_norm.617(%70, %arg48, %arg49, %arg50, %arg51) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %72 = mhlo.get_tuple_element %71[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %73 = mhlo.get_tuple_element %71[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %74 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %75 = call @aten.expand.147(%74) : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %76 = call @aten.mul.572(%62, %75) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %77 = call @aten.add.639(%73, %76) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %78 = call @aten.relu.644(%77) : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %79 = mhlo.convert %arg62 : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %80 = call @aten.convolution_overrideable.650(%78, %79) : (tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<4x256x14x14xf16>
    %81 = call @aten.native_batch_norm.655(%80, %arg63, %arg64, %arg65, %arg66) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %82 = mhlo.get_tuple_element %81[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %83 = mhlo.convert %arg52 : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %84 = call @aten.convolution_overrideable.684(%78, %83) : (tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<4x256x14x14xf16>
    %85 = call @aten.native_batch_norm.689(%84, %arg53, %arg54, %arg55, %arg56) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %86 = mhlo.get_tuple_element %85[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %87 = mhlo.get_tuple_element %85[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %88 = call @aten.relu.711(%87) : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %89 = mhlo.convert %arg57 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %90 = call @aten.convolution_overrideable.717(%88, %89) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %91 = call @aten.native_batch_norm.722(%90, %arg58, %arg59, %arg60, %arg61) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %92 = mhlo.get_tuple_element %91[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %93 = mhlo.get_tuple_element %91[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %94 = mhlo.get_tuple_element %81[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %95 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %96 = call @aten.expand.138(%95) : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %97 = call @aten.mul.677(%94, %96) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %98 = call @aten.add.744(%93, %97) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %99 = call @aten.relu.749(%98) : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %100 = mhlo.convert %arg67 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %101 = call @aten.convolution_overrideable.762(%99, %100) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %102 = call @aten.native_batch_norm.767(%101, %arg68, %arg69, %arg70, %arg71) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %103 = mhlo.get_tuple_element %102[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %104 = mhlo.get_tuple_element %102[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %105 = call @aten.relu.789(%104) : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %106 = mhlo.convert %arg72 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %107 = call @aten.convolution_overrideable.795(%105, %106) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %108 = call @aten.native_batch_norm.800(%107, %arg73, %arg74, %arg75, %arg76) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %109 = mhlo.get_tuple_element %108[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %110 = mhlo.get_tuple_element %108[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %111 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %112 = call @aten.expand.130(%111) : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %113 = call @aten.mul.755(%99, %112) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %114 = call @aten.add.822(%110, %113) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %115 = call @aten.relu.827(%114) : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %116 = mhlo.convert %arg87 : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %117 = call @aten.convolution_overrideable.833(%115, %116) : (tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<4x512x7x7xf16>
    %118 = call @aten.native_batch_norm.838(%117, %arg88, %arg89, %arg90, %arg91) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %119 = mhlo.get_tuple_element %118[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %120 = mhlo.convert %arg77 : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %121 = call @aten.convolution_overrideable.867(%115, %120) : (tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<4x512x7x7xf16>
    %122 = call @aten.native_batch_norm.872(%121, %arg78, %arg79, %arg80, %arg81) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %123 = mhlo.get_tuple_element %122[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %124 = mhlo.get_tuple_element %122[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %125 = call @aten.relu.894(%124) : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %126 = mhlo.convert %arg82 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %127 = call @aten.convolution_overrideable.900(%125, %126) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %128 = call @aten.native_batch_norm.905(%127, %arg83, %arg84, %arg85, %arg86) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %129 = mhlo.get_tuple_element %128[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %130 = mhlo.get_tuple_element %128[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %131 = mhlo.get_tuple_element %118[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %132 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %133 = call @aten.expand.121(%132) : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %134 = call @aten.mul.860(%131, %133) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %135 = call @aten.add.927(%130, %134) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %136 = call @aten.relu.932(%135) : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %137 = mhlo.convert %arg92 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %138 = call @aten.convolution_overrideable.945(%136, %137) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %139 = call @aten.native_batch_norm.950(%138, %arg93, %arg94, %arg95, %arg96) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %140 = mhlo.get_tuple_element %139[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %141 = mhlo.get_tuple_element %139[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %142 = call @aten.relu.972(%141) : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %143 = mhlo.convert %arg97 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %144 = call @aten.convolution_overrideable.978(%142, %143) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %145 = call @aten.native_batch_norm.983(%144, %arg98, %arg99, %arg100, %arg101) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %146 = mhlo.get_tuple_element %145[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %147 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %148 = mhlo.constant dense<4> : tensor<i64>
    %149 = mhlo.convert %148 : (tensor<i64>) -> tensor<f32>
    %150 = call @aten.div.1174(%147, %149) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %151 = call @aten.neg.1179(%150) : (tensor<f32>) -> tensor<f32>
    %152 = call @aten.expand.1183(%151) : (tensor<f32>) -> tensor<4x1000xf32>
    %153 = call @aten.mul.1190(%152, %arg1) : (tensor<4x1000xf32>, tensor<4x1000xf32>) -> tensor<4x1000xf32>
    %154 = mhlo.convert %153 : (tensor<4x1000xf32>) -> tensor<4x1000xf16>
    %155 = mhlo.get_tuple_element %145[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %156 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %157 = call @aten.expand.113(%156) : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %158 = call @aten.mul.938(%136, %157) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %159 = call @aten.add.1005(%155, %158) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %160 = call @aten.relu.1010(%159) : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %161 = call @aten.mean.1020(%160) : (tensor<4x512x7x7xf16>) -> tensor<4x512x1x1xf16>
    %162 = call @aten.view.1037(%161) : (tensor<4x512x1x1xf16>) -> tensor<4x512xf16>
    %163 = mhlo.convert %arg102 : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %164 = call @aten.permute.108(%163) {xla_shape = "f16[512,1000]{0,1}"} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    %165 = mhlo.convert %arg103 : (tensor<1000xf32>) -> tensor<1000xf16>
    %166 = call @aten.addmm.1041(%162, %164, %165) : (tensor<4x512xf16>, tensor<512x1000xf16>, tensor<1000xf16>) -> tensor<4x1000xf16>
    %167 = call @aten.log_softmax.1060(%166) : (tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %168 = call @aten._log_softmax_backward_data.1200(%154, %167) : (tensor<4x1000xf16>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %169 = call @aten.permute.1163(%163) {xla_shape = "f16[512,1000]{0,1}"} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    %170 = call @aten.permute.1167(%169) : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    %171 = call @aten.mm.1210(%168, %170) : (tensor<4x1000xf16>, tensor<1000x512xf16>) -> tensor<4x512xf16>
    %172 = call @aten.view.1215(%171) : (tensor<4x512xf16>) -> tensor<4x512x1x1xf16>
    %173 = call @aten.expand.1219(%172) : (tensor<4x512x1x1xf16>) -> tensor<4x512x7x7xf16>
    %174 = mhlo.constant dense<4.900000e+01> : tensor<f16>
    %175 = call @aten.div.1225(%173, %174) : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<4x512x7x7xf16>
    %176 = call @aten.threshold_backward.1231(%175, %160) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %177 = mhlo.get_tuple_element %145[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %178 = mhlo.get_tuple_element %145[3] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %179 = call @aten.native_batch_norm_backward.1241(%176, %144, %arg98, %177, %178) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %180 = mhlo.get_tuple_element %179[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %181 = call @aten.convolution_backward_overrideable.1270(%180, %142, %143) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f16[512,512,3,3]{0,1,3,2}, f16[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %182 = mhlo.get_tuple_element %181[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %183 = mhlo.get_tuple_element %181[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<4x512x7x7xf16>
    %184 = call @aten.threshold_backward.1286(%183, %142) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %185 = mhlo.get_tuple_element %139[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %186 = mhlo.get_tuple_element %139[3] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %187 = call @aten.native_batch_norm_backward.1296(%184, %138, %arg93, %185, %186) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %188 = mhlo.get_tuple_element %187[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %189 = call @aten.convolution_backward_overrideable.1325(%188, %136, %137) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f16[512,512,3,3]{0,1,3,2}, f16[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %190 = mhlo.get_tuple_element %189[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %191 = mhlo.get_tuple_element %189[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<4x512x7x7xf16>
    %192 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %193 = call @aten.expand.1155(%192) : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %194 = call @aten.mul.1341(%191, %193) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %195 = call @aten.add.1346(%176, %194) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %196 = call @aten.threshold_backward.1351(%195, %136) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %197 = mhlo.get_tuple_element %128[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %198 = mhlo.get_tuple_element %128[3] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %199 = call @aten.native_batch_norm_backward.1361(%196, %127, %arg83, %197, %198) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %200 = mhlo.get_tuple_element %199[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %201 = call @aten.convolution_backward_overrideable.1390(%200, %125, %126) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f16[512,512,3,3]{0,1,3,2}, f16[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %202 = mhlo.get_tuple_element %201[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %203 = mhlo.get_tuple_element %201[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<4x512x7x7xf16>
    %204 = call @aten.threshold_backward.1406(%203, %125) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %205 = mhlo.get_tuple_element %122[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %206 = mhlo.get_tuple_element %122[3] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %207 = call @aten.native_batch_norm_backward.1416(%204, %121, %arg78, %205, %206) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %208 = mhlo.get_tuple_element %207[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %209 = call @aten.convolution_backward_overrideable.1445(%208, %115, %120) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[512,256,3,3]{0,1,3,2}, f16[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
    %210 = mhlo.get_tuple_element %209[2] : (tuple<tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %211 = mhlo.get_tuple_element %118[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %212 = mhlo.get_tuple_element %118[3] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %213 = call @aten.native_batch_norm_backward.1466(%196, %117, %arg88, %211, %212) {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %214 = mhlo.get_tuple_element %213[0] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<4x512x7x7xf16>
    %215 = call @aten.convolution_backward_overrideable.1495(%214, %115, %116) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[512,256,1,1]{0,1,3,2}, f16[512]{0})"} : (tensor<4x512x7x7xf16>, tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
    %216 = mhlo.get_tuple_element %215[2] : (tuple<tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %217 = mhlo.get_tuple_element %215[0] : (tuple<tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<4x256x14x14xf16>
    %218 = mhlo.get_tuple_element %209[0] : (tuple<tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<4x256x14x14xf16>
    %219 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %220 = call @aten.expand.1147(%219) : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %221 = call @aten.mul.1461(%218, %220) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %222 = call @aten.add.1511(%217, %221) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %223 = call @aten.threshold_backward.1516(%222, %115) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %224 = mhlo.get_tuple_element %108[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %225 = mhlo.get_tuple_element %108[3] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %226 = call @aten.native_batch_norm_backward.1526(%223, %107, %arg73, %224, %225) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %227 = mhlo.get_tuple_element %226[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %228 = call @aten.convolution_backward_overrideable.1555(%227, %105, %106) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[256,256,3,3]{0,1,3,2}, f16[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %229 = mhlo.get_tuple_element %228[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %230 = mhlo.get_tuple_element %228[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<4x256x14x14xf16>
    %231 = call @aten.threshold_backward.1571(%230, %105) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %232 = mhlo.get_tuple_element %102[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %233 = mhlo.get_tuple_element %102[3] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %234 = call @aten.native_batch_norm_backward.1581(%231, %101, %arg68, %232, %233) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %235 = mhlo.get_tuple_element %234[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %236 = call @aten.convolution_backward_overrideable.1610(%235, %99, %100) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[256,256,3,3]{0,1,3,2}, f16[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %237 = mhlo.get_tuple_element %236[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %238 = mhlo.get_tuple_element %236[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<4x256x14x14xf16>
    %239 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %240 = call @aten.expand.1139(%239) : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %241 = call @aten.mul.1626(%238, %240) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %242 = call @aten.add.1631(%223, %241) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %243 = call @aten.threshold_backward.1636(%242, %99) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %244 = mhlo.get_tuple_element %91[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %245 = mhlo.get_tuple_element %91[3] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %246 = call @aten.native_batch_norm_backward.1646(%243, %90, %arg58, %244, %245) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %247 = mhlo.get_tuple_element %246[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %248 = call @aten.convolution_backward_overrideable.1675(%247, %88, %89) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[256,256,3,3]{0,1,3,2}, f16[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %249 = mhlo.get_tuple_element %248[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %250 = mhlo.get_tuple_element %248[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<4x256x14x14xf16>
    %251 = call @aten.threshold_backward.1691(%250, %88) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %252 = mhlo.get_tuple_element %85[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %253 = mhlo.get_tuple_element %85[3] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %254 = call @aten.native_batch_norm_backward.1701(%251, %84, %arg53, %252, %253) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %255 = mhlo.get_tuple_element %254[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %256 = call @aten.convolution_backward_overrideable.1730(%255, %78, %83) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[256,128,3,3]{0,1,3,2}, f16[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
    %257 = mhlo.get_tuple_element %256[2] : (tuple<tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %258 = mhlo.get_tuple_element %81[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %259 = mhlo.get_tuple_element %81[3] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %260 = call @aten.native_batch_norm_backward.1751(%243, %80, %arg63, %258, %259) {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %261 = mhlo.get_tuple_element %260[0] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<4x256x14x14xf16>
    %262 = call @aten.convolution_backward_overrideable.1780(%261, %78, %79) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[256,128,1,1]{0,1,3,2}, f16[256]{0})"} : (tensor<4x256x14x14xf16>, tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
    %263 = mhlo.get_tuple_element %262[2] : (tuple<tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %264 = mhlo.get_tuple_element %262[0] : (tuple<tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<4x128x28x28xf16>
    %265 = mhlo.get_tuple_element %256[0] : (tuple<tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<4x128x28x28xf16>
    %266 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %267 = call @aten.expand.1131(%266) : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %268 = call @aten.mul.1746(%265, %267) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %269 = call @aten.add.1796(%264, %268) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %270 = call @aten.threshold_backward.1801(%269, %78) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %271 = mhlo.get_tuple_element %71[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %272 = mhlo.get_tuple_element %71[3] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %273 = call @aten.native_batch_norm_backward.1811(%270, %70, %arg48, %271, %272) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %274 = mhlo.get_tuple_element %273[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %275 = call @aten.convolution_backward_overrideable.1840(%274, %68, %69) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[128,128,3,3]{0,1,3,2}, f16[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %276 = mhlo.get_tuple_element %275[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %277 = mhlo.get_tuple_element %275[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<4x128x28x28xf16>
    %278 = call @aten.threshold_backward.1856(%277, %68) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %279 = mhlo.get_tuple_element %65[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %280 = mhlo.get_tuple_element %65[3] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %281 = call @aten.native_batch_norm_backward.1866(%278, %64, %arg43, %279, %280) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %282 = mhlo.get_tuple_element %281[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %283 = call @aten.convolution_backward_overrideable.1895(%282, %62, %63) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[128,128,3,3]{0,1,3,2}, f16[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %284 = mhlo.get_tuple_element %283[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %285 = mhlo.get_tuple_element %283[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<4x128x28x28xf16>
    %286 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %287 = call @aten.expand.1123(%286) : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %288 = call @aten.mul.1911(%285, %287) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %289 = call @aten.add.1916(%270, %288) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %290 = call @aten.threshold_backward.1921(%289, %62) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %291 = mhlo.get_tuple_element %54[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %292 = mhlo.get_tuple_element %54[3] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %293 = call @aten.native_batch_norm_backward.1931(%290, %53, %arg33, %291, %292) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %294 = mhlo.get_tuple_element %293[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %295 = call @aten.convolution_backward_overrideable.1960(%294, %51, %52) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[128,128,3,3]{0,1,3,2}, f16[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %296 = mhlo.get_tuple_element %295[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %297 = mhlo.get_tuple_element %295[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<4x128x28x28xf16>
    %298 = call @aten.threshold_backward.1976(%297, %51) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %299 = mhlo.get_tuple_element %48[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %300 = mhlo.get_tuple_element %48[3] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %301 = call @aten.native_batch_norm_backward.1986(%298, %47, %arg28, %299, %300) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %302 = mhlo.get_tuple_element %301[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %303 = call @aten.convolution_backward_overrideable.2015(%302, %41, %46) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[128,64,3,3]{0,1,3,2}, f16[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
    %304 = mhlo.get_tuple_element %303[2] : (tuple<tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %305 = mhlo.get_tuple_element %44[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %306 = mhlo.get_tuple_element %44[3] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %307 = call @aten.native_batch_norm_backward.2036(%290, %43, %arg38, %305, %306) {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %308 = mhlo.get_tuple_element %307[0] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<4x128x28x28xf16>
    %309 = call @aten.convolution_backward_overrideable.2065(%308, %41, %42) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[128,64,1,1]{0,1,3,2}, f16[128]{0})"} : (tensor<4x128x28x28xf16>, tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
    %310 = mhlo.get_tuple_element %309[2] : (tuple<tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %311 = mhlo.get_tuple_element %309[0] : (tuple<tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<4x64x56x56xf16>
    %312 = mhlo.get_tuple_element %303[0] : (tuple<tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<4x64x56x56xf16>
    %313 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %314 = call @aten.expand.1115(%313) : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %315 = call @aten.mul.2031(%312, %314) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %316 = call @aten.add.2081(%311, %315) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %317 = call @aten.threshold_backward.2086(%316, %41) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %318 = mhlo.get_tuple_element %34[1] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %319 = mhlo.get_tuple_element %34[3] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %320 = call @aten.native_batch_norm_backward.2096(%317, %33, %arg23, %318, %319) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %321 = mhlo.get_tuple_element %320[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x56x56xf16>
    %322 = call @aten.convolution_backward_overrideable.2125(%321, %31, %32) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[64,64,3,3]{0,1,3,2}, f16[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %323 = mhlo.get_tuple_element %322[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %324 = mhlo.get_tuple_element %322[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<4x64x56x56xf16>
    %325 = call @aten.threshold_backward.2141(%324, %31) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %326 = mhlo.get_tuple_element %28[1] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %327 = mhlo.get_tuple_element %28[3] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %328 = call @aten.native_batch_norm_backward.2151(%325, %27, %arg18, %326, %327) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %329 = mhlo.get_tuple_element %328[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x56x56xf16>
    %330 = call @aten.convolution_backward_overrideable.2180(%329, %25, %26) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[64,64,3,3]{0,1,3,2}, f16[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %331 = mhlo.get_tuple_element %330[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %332 = mhlo.get_tuple_element %330[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<4x64x56x56xf16>
    %333 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %334 = call @aten.expand.1107(%333) : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %335 = call @aten.mul.2196(%332, %334) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %336 = call @aten.add.2201(%317, %335) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %337 = call @aten.threshold_backward.2206(%336, %25) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %338 = mhlo.get_tuple_element %18[1] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %339 = mhlo.get_tuple_element %18[3] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %340 = call @aten.native_batch_norm_backward.2216(%337, %17, %arg13, %338, %339) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %341 = mhlo.get_tuple_element %340[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x56x56xf16>
    %342 = call @aten.convolution_backward_overrideable.2245(%341, %15, %16) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[64,64,3,3]{0,1,3,2}, f16[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %343 = mhlo.get_tuple_element %342[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %344 = mhlo.get_tuple_element %342[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<4x64x56x56xf16>
    %345 = call @aten.threshold_backward.2261(%344, %15) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %346 = mhlo.get_tuple_element %12[1] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %347 = mhlo.get_tuple_element %12[3] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %348 = call @aten.native_batch_norm_backward.2271(%345, %11, %arg8, %346, %347) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %349 = mhlo.get_tuple_element %348[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x56x56xf16>
    %350 = call @aten.convolution_backward_overrideable.2300(%349, %9, %10) {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[64,64,3,3]{0,1,3,2}, f16[64]{0})"} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %351 = mhlo.get_tuple_element %350[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %352 = mhlo.get_tuple_element %350[0] : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<4x64x56x56xf16>
    %353 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %354 = call @aten.expand.1099(%353) : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %355 = call @aten.mul.2316(%352, %354) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %356 = call @aten.add.2321(%337, %355) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %357 = call @aten.max_pool2d_with_indices_backward.2334(%356, %6) : (tensor<4x64x56x56xf16>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %358 = call @aten.threshold_backward.2340(%357, %6) : (tensor<4x64x112x112xf16>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %359 = mhlo.get_tuple_element %3[1] : (tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %360 = mhlo.get_tuple_element %3[3] : (tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %361 = call @aten.native_batch_norm_backward.2350(%358, %2, %arg3, %359, %360) {xla_shape = "(f16[4,64,112,112]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<4x64x112x112xf16>, tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
    %362 = mhlo.get_tuple_element %361[0] : (tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<4x64x112x112xf16>
    %363 = call @aten.convolution_backward_overrideable.2379(%362, %0, %1) {xla_shape = "(f16[4,3,224,224]{3,2,1,0}, f16[64,3,7,7]{0,1,3,2}, f16[64]{0})"} : (tensor<4x64x112x112xf16>, tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tuple<tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
    %364 = mhlo.get_tuple_element %363[0] : (tuple<tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<4x3x224x224xf16>
    %365 = mhlo.get_tuple_element %363[2] : (tuple<tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %366 = call @aten.mul.1073(%167, %arg1) : (tensor<4x1000xf16>, tensor<4x1000xf32>) -> tensor<4x1000xf32>
    %367 = call @aten.sum.1083(%366) : (tensor<4x1000xf32>) -> tensor<f32>
    %368 = call @aten.neg.1089(%367) : (tensor<f32>) -> tensor<f32>
    %369 = mhlo.constant dense<4.000000e+00> : tensor<f32>
    %370 = call @aten.div.1093(%368, %369) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %371 = mhlo.get_tuple_element %363[1] {xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (tuple<tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<64x3x7x7xf16>
    %372 = mhlo.convert %371 {xla_shape = "f32[64,3,7,7]{0,1,3,2}"} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %373 = mhlo.get_tuple_element %361[1] : (tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %374 = mhlo.get_tuple_element %361[2] : (tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %375 = mhlo.get_tuple_element %350[1] {xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %376 = mhlo.convert %375 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %377 = mhlo.get_tuple_element %348[1] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %378 = mhlo.get_tuple_element %348[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %379 = mhlo.get_tuple_element %342[1] {xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %380 = mhlo.convert %379 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %381 = mhlo.get_tuple_element %340[1] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %382 = mhlo.get_tuple_element %340[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %383 = mhlo.get_tuple_element %330[1] {xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %384 = mhlo.convert %383 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %385 = mhlo.get_tuple_element %328[1] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %386 = mhlo.get_tuple_element %328[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %387 = mhlo.get_tuple_element %322[1] {xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %388 = mhlo.convert %387 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %389 = mhlo.get_tuple_element %320[1] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %390 = mhlo.get_tuple_element %320[2] : (tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %391 = mhlo.get_tuple_element %303[1] {xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (tuple<tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<128x64x3x3xf16>
    %392 = mhlo.convert %391 {xla_shape = "f32[128,64,3,3]{0,1,3,2}"} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %393 = mhlo.get_tuple_element %301[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %394 = mhlo.get_tuple_element %301[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %395 = mhlo.get_tuple_element %295[1] {xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %396 = mhlo.convert %395 {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %397 = mhlo.get_tuple_element %293[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %398 = mhlo.get_tuple_element %293[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %399 = mhlo.get_tuple_element %309[1] {xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (tuple<tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<128x64x1x1xf16>
    %400 = mhlo.convert %399 {xla_shape = "f32[128,64,1,1]{0,1,3,2}"} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %401 = mhlo.get_tuple_element %307[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %402 = mhlo.get_tuple_element %307[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %403 = mhlo.get_tuple_element %283[1] {xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %404 = mhlo.convert %403 {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %405 = mhlo.get_tuple_element %281[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %406 = mhlo.get_tuple_element %281[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %407 = mhlo.get_tuple_element %275[1] {xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %408 = mhlo.convert %407 {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %409 = mhlo.get_tuple_element %273[1] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %410 = mhlo.get_tuple_element %273[2] : (tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %411 = mhlo.get_tuple_element %256[1] {xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (tuple<tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<256x128x3x3xf16>
    %412 = mhlo.convert %411 {xla_shape = "f32[256,128,3,3]{0,1,3,2}"} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %413 = mhlo.get_tuple_element %254[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %414 = mhlo.get_tuple_element %254[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %415 = mhlo.get_tuple_element %248[1] {xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %416 = mhlo.convert %415 {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %417 = mhlo.get_tuple_element %246[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %418 = mhlo.get_tuple_element %246[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %419 = mhlo.get_tuple_element %262[1] {xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (tuple<tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<256x128x1x1xf16>
    %420 = mhlo.convert %419 {xla_shape = "f32[256,128,1,1]{0,1,3,2}"} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %421 = mhlo.get_tuple_element %260[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %422 = mhlo.get_tuple_element %260[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %423 = mhlo.get_tuple_element %236[1] {xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %424 = mhlo.convert %423 {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %425 = mhlo.get_tuple_element %234[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %426 = mhlo.get_tuple_element %234[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %427 = mhlo.get_tuple_element %228[1] {xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %428 = mhlo.convert %427 {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %429 = mhlo.get_tuple_element %226[1] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %430 = mhlo.get_tuple_element %226[2] : (tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %431 = mhlo.get_tuple_element %209[1] {xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (tuple<tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<512x256x3x3xf16>
    %432 = mhlo.convert %431 {xla_shape = "f32[512,256,3,3]{0,1,3,2}"} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %433 = mhlo.get_tuple_element %207[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %434 = mhlo.get_tuple_element %207[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %435 = mhlo.get_tuple_element %201[1] {xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %436 = mhlo.convert %435 {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %437 = mhlo.get_tuple_element %199[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %438 = mhlo.get_tuple_element %199[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %439 = mhlo.get_tuple_element %215[1] {xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (tuple<tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<512x256x1x1xf16>
    %440 = mhlo.convert %439 {xla_shape = "f32[512,256,1,1]{0,1,3,2}"} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %441 = mhlo.get_tuple_element %213[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %442 = mhlo.get_tuple_element %213[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %443 = mhlo.get_tuple_element %189[1] {xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %444 = mhlo.convert %443 {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %445 = mhlo.get_tuple_element %187[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %446 = mhlo.get_tuple_element %187[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %447 = mhlo.get_tuple_element %181[1] {xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %448 = mhlo.convert %447 {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %449 = mhlo.get_tuple_element %179[1] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %450 = mhlo.get_tuple_element %179[2] : (tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %451 = call @aten.view.2415(%161) : (tensor<4x512x1x1xf16>) -> tensor<4x512xf16>
    %452 = call @aten.permute.2419(%451) {xla_shape = "f16[512,4]{0,1}"} : (tensor<4x512xf16>) -> tensor<512x4xf16>
    %453 = call @aten.mm.2423(%452, %168) : (tensor<512x4xf16>, tensor<4x1000xf16>) -> tensor<512x1000xf16>
    %454 = call @aten.permute.2428(%453) {xla_shape = "f16[1000,512]{0,1}"} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    %455 = mhlo.convert %454 {xla_shape = "f32[1000,512]{0,1}"} : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %456 = call @aten.sum.2437(%168) : (tensor<4x1000xf16>) -> tensor<1x1000xf32>
    %457 = call @aten.view.2445(%456) : (tensor<1x1000xf32>) -> tensor<1000xf32>
    %458 = mhlo.convert %457 : (tensor<1000xf32>) -> tensor<1000xf16>
    %459 = mhlo.convert %458 : (tensor<1000xf16>) -> tensor<1000xf32>
    %460 = mhlo.tuple %370, %372, %373, %374, %376, %377, %378, %380, %381, %382, %384, %385, %386, %388, %389, %390, %392, %393, %394, %396, %397, %398, %400, %401, %402, %404, %405, %406, %408, %409, %410, %412, %413, %414, %416, %417, %418, %420, %421, %422, %424, %425, %426, %428, %429, %430, %432, %433, %434, %436, %437, %438, %440, %441, %442, %444, %445, %446, %448, %449, %450, %455, %459 {xla_shape = "(f32[], f32[64,3,7,7]{0,1,3,2}, f32[64]{0}, f32[64]{0}, f32[64,64,3,3]{0,1,3,2}, /*index=5*/f32[64]{0}, f32[64]{0}, f32[64,64,3,3]{0,1,3,2}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[64,64,3,3]{0,1,3,2}, f32[64]{0}, f32[64]{0}, f32[64,64,3,3]{0,1,3,2}, f32[64]{0}, /*index=15*/f32[64]{0}, f32[128,64,3,3]{0,1,3,2}, f32[128]{0}, f32[128]{0}, f32[128,128,3,3]{0,1,3,2}, /*index=20*/f32[128]{0}, f32[128]{0}, f32[128,64,1,1]{0,1,3,2}, f32[128]{0}, f32[128]{0}, /*index=25*/f32[128,128,3,3]{0,1,3,2}, f32[128]{0}, f32[128]{0}, f32[128,128,3,3]{0,1,3,2}, f32[128]{0}, /*index=30*/f32[128]{0}, f32[256,128,3,3]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,256,3,3]{0,1,3,2}, /*index=35*/f32[256]{0}, f32[256]{0}, f32[256,128,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, /*index=40*/f32[256,256,3,3]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0}, /*index=45*/f32[256]{0}, f32[512,256,3,3]{0,1,3,2}, f32[512]{0}, f32[512]{0}, f32[512,512,3,3]{0,1,3,2}, /*index=50*/f32[512]{0}, f32[512]{0}, f32[512,256,1,1]{0,1,3,2}, f32[512]{0}, f32[512]{0}, /*index=55*/f32[512,512,3,3]{0,1,3,2}, f32[512]{0}, f32[512]{0}, f32[512,512,3,3]{0,1,3,2}, f32[512]{0}, /*index=60*/f32[512]{0}, f32[1000,512]{0,1}, f32[1000]{0})"} : !tuple
    return %460 : !tuple
  }
  func.func private @aten.convolution_overrideable.181(%arg0: tensor<4x3x224x224xf16>, %arg1: tensor<64x3x7x7xf16>) -> tensor<4x64x112x112xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<4x64x112x112xf16>
    return %0 : tensor<4x64x112x112xf16>
  }
  func.func private @aten.native_batch_norm.186(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<64xf32>
    %5 = mhlo.rsqrt %4 : tensor<64xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,64,112,112]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %6 : tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.relu.208(%arg0: tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x112x112xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x64x112x112xf16>
    return %2 : tensor<4x64x112x112xf16>
  }
  func.func private @aten.max_pool2d.283(%arg0: tensor<4x64x112x112xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<4x64x56x56xui32>> {
    %0 = mhlo.constant dense<0> : tensor<ui32>
    %1 = mhlo.constant dense<802816> : tensor<ui32>
    %2 = mhlo.constant dense<0xFC00> : tensor<f16>
    %3 = "mhlo.pad"(%arg0, %2) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<f16>) -> tensor<4x64x114x114xf16>
    %4 = mhlo.constant dense<0xFC00> : tensor<f16>
    %5 = "mhlo.reduce_window"(%3, %4) ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %16 = mhlo.maximum %arg1, %arg2 : tensor<f16>
      mhlo.return %16 : tensor<f16>
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x114x114xf16>, tensor<f16>) -> tensor<4x64x56x56xf16>
    %6 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<12544xui32>
    %7 = mhlo.reshape %6 : (tensor<12544xui32>) -> tensor<112x112xui32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<112x112xui32>) -> tensor<4x64x112x112xui32>
    %9 = mhlo.constant dense<4294967295> : tensor<ui32>
    %10 = "mhlo.pad"(%8, %9) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<4x64x112x112xui32>, tensor<ui32>) -> tensor<4x64x114x114xui32>
    %11 = mhlo.constant dense<0> : tensor<ui32>
    %12 = "mhlo.broadcast_in_dim"(%11) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<ui32>) -> tensor<802816xui32>
    %13:6 = mhlo.while(%iterArg = %0, %iterArg_0 = %1, %iterArg_1 = %3, %iterArg_2 = %5, %iterArg_3 = %10, %iterArg_4 = %12) : tensor<ui32>, tensor<ui32>, tensor<4x64x114x114xf16>, tensor<4x64x56x56xf16>, tensor<4x64x114x114xui32>, tensor<802816xui32>
     cond {
      %16 = mhlo.compare  LT, %iterArg, %iterArg_0 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
      mhlo.return %16 : tensor<i1>
    } do {
      %16 = mhlo.constant dense<200704> : tensor<ui32>
      %17 = mhlo.remainder %iterArg, %16 : tensor<ui32>
      %18 = mhlo.constant dense<3136> : tensor<ui32>
      %19 = mhlo.remainder %17, %18 : tensor<ui32>
      %20 = mhlo.constant dense<56> : tensor<ui32>
      %21 = mhlo.remainder %19, %20 : tensor<ui32>
      %22 = mhlo.constant dense<1> : tensor<ui32>
      %23 = mhlo.remainder %21, %22 : tensor<ui32>
      %24 = mhlo.constant dense<1> : tensor<ui32>
      %25 = mhlo.add %iterArg, %24 : tensor<ui32>
      %26 = mhlo.constant dense<1> : tensor<ui32>
      %27 = mhlo.divide %iterArg, %16 : tensor<ui32>
      %28 = mhlo.multiply %26, %27 : tensor<ui32>
      %29 = mhlo.constant dense<1> : tensor<ui32>
      %30 = mhlo.divide %17, %18 : tensor<ui32>
      %31 = mhlo.multiply %29, %30 : tensor<ui32>
      %32 = mhlo.constant dense<2> : tensor<ui32>
      %33 = mhlo.divide %19, %20 : tensor<ui32>
      %34 = mhlo.multiply %32, %33 : tensor<ui32>
      %35 = mhlo.constant dense<2> : tensor<ui32>
      %36 = mhlo.divide %21, %22 : tensor<ui32>
      %37 = mhlo.multiply %35, %36 : tensor<ui32>
      %38 = "mhlo.dynamic_slice"(%iterArg_1, %28, %31, %34, %37) {slice_sizes = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<4x64x114x114xf16>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x3x3xf16>
      %39 = "mhlo.dynamic_slice"(%iterArg_2, %27, %30, %33, %36) {slice_sizes = dense<1> : tensor<4xi64>} : (tensor<4x64x56x56xf16>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x1x1xf16>
      %40 = mhlo.constant dense<0xFC00> : tensor<f16>
      %41 = "mhlo.select_and_scatter"(%38, %39, %40) ({
      ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
        %51 = mhlo.compare  GE, %arg1, %arg2 : (tensor<f16>, tensor<f16>) -> tensor<i1>
        mhlo.return %51 : tensor<i1>
      }, {
      ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
        %51 = mhlo.maximum %arg1, %arg2 : tensor<f16>
        mhlo.return %51 : tensor<f16>
      }) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<f16>) -> tensor<1x1x3x3xf16>
      %42 = "mhlo.broadcast_in_dim"(%40) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x1x3x3xf16>
      %43 = mhlo.compare  NE, %41, %42 : (tensor<1x1x3x3xf16>, tensor<1x1x3x3xf16>) -> tensor<1x1x3x3xi1>
      %44 = "mhlo.dynamic_slice"(%iterArg_3, %28, %31, %34, %37) {slice_sizes = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<4x64x114x114xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x3x3xui32>
      %45 = mhlo.constant dense<4294967295> : tensor<ui32>
      %46 = "mhlo.broadcast_in_dim"(%45) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<ui32>) -> tensor<1x1x3x3xui32>
      %47 = mhlo.select %43, %44, %46 : tensor<1x1x3x3xi1>, tensor<1x1x3x3xui32>
      %48 = "mhlo.reduce_window"(%47, %45) ({
      ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>):
        %51 = mhlo.minimum %arg1, %arg2 : tensor<ui32>
        mhlo.return %51 : tensor<ui32>
      }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x1x3x3xui32>, tensor<ui32>) -> tensor<1x1x1x1xui32>
      %49 = mhlo.reshape %48 : (tensor<1x1x1x1xui32>) -> tensor<1xui32>
      %50 = mhlo.dynamic_update_slice %iterArg_4, %49, %iterArg : (tensor<802816xui32>, tensor<1xui32>, tensor<ui32>) -> tensor<802816xui32>
      mhlo.return %25, %iterArg_0, %iterArg_1, %iterArg_2, %iterArg_3, %50 : tensor<ui32>, tensor<ui32>, tensor<4x64x114x114xf16>, tensor<4x64x56x56xf16>, tensor<4x64x114x114xui32>, tensor<802816xui32>
    }
    %14 = mhlo.reshape %13#5 : (tensor<802816xui32>) -> tensor<4x64x56x56xui32>
    %15 = mhlo.tuple %5, %14 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, u32[4,64,56,56]{3,2,1,0})"} : tuple<tensor<4x64x56x56xf16>, tensor<4x64x56x56xui32>>
    return %15 : tuple<tensor<4x64x56x56xf16>, tensor<4x64x56x56xui32>>
  }
  func.func private @aten.convolution_overrideable.318(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm.323(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<64xf32>
    %5 = mhlo.rsqrt %4 : tensor<64xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %6 : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.relu.345(%arg0: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.convolution_overrideable.351(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm.356(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<64xf32>
    %5 = mhlo.rsqrt %4 : tensor<64xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %6 : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.expand.172(%arg0: tensor<f16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    return %3 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.mul.311(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.add.378(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.relu.383(%arg0: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.convolution_overrideable.396(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm.401(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<64xf32>
    %5 = mhlo.rsqrt %4 : tensor<64xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %6 : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.relu.423(%arg0: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.convolution_overrideable.429(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm.434(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<64xf32>
    %5 = mhlo.rsqrt %4 : tensor<64xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %6 : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.expand.164(%arg0: tensor<f16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    return %3 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.mul.389(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.add.456(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.relu.461(%arg0: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.convolution_overrideable.467(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm.472(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<128xf32>
    %5 = mhlo.rsqrt %4 : tensor<128xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %6 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_overrideable.501(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm.506(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<128xf32>
    %5 = mhlo.rsqrt %4 : tensor<128xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %6 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.relu.528(%arg0: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.convolution_overrideable.534(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm.539(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<128xf32>
    %5 = mhlo.rsqrt %4 : tensor<128xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %6 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.expand.155(%arg0: tensor<f16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    return %3 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.mul.494(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.add.561(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.relu.566(%arg0: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.convolution_overrideable.579(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm.584(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<128xf32>
    %5 = mhlo.rsqrt %4 : tensor<128xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %6 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.relu.606(%arg0: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.convolution_overrideable.612(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm.617(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<128xf32>
    %5 = mhlo.rsqrt %4 : tensor<128xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %6 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.expand.147(%arg0: tensor<f16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    return %3 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.mul.572(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.add.639(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.relu.644(%arg0: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.convolution_overrideable.650(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm.655(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<256xf32>
    %5 = mhlo.rsqrt %4 : tensor<256xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %6 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_overrideable.684(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm.689(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<256xf32>
    %5 = mhlo.rsqrt %4 : tensor<256xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %6 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.relu.711(%arg0: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.convolution_overrideable.717(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm.722(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<256xf32>
    %5 = mhlo.rsqrt %4 : tensor<256xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %6 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.expand.138(%arg0: tensor<f16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    return %3 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.mul.677(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.add.744(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.relu.749(%arg0: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.convolution_overrideable.762(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm.767(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<256xf32>
    %5 = mhlo.rsqrt %4 : tensor<256xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %6 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.relu.789(%arg0: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.convolution_overrideable.795(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm.800(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<256xf32>
    %5 = mhlo.rsqrt %4 : tensor<256xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %6 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.expand.130(%arg0: tensor<f16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    return %3 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.mul.755(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.add.822(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.relu.827(%arg0: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.convolution_overrideable.833(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm.838(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<512xf32>
    %5 = mhlo.rsqrt %4 : tensor<512xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %6 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_overrideable.867(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm.872(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<512xf32>
    %5 = mhlo.rsqrt %4 : tensor<512xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %6 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.relu.894(%arg0: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.convolution_overrideable.900(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm.905(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<512xf32>
    %5 = mhlo.rsqrt %4 : tensor<512xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %6 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.expand.121(%arg0: tensor<f16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    return %3 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.mul.860(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.add.927(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.relu.932(%arg0: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.convolution_overrideable.945(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm.950(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<512xf32>
    %5 = mhlo.rsqrt %4 : tensor<512xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %6 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.relu.972(%arg0: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.convolution_overrideable.978(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm.983(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %4 = mhlo.add %batch_var, %3 : tensor<512xf32>
    %5 = mhlo.rsqrt %4 : tensor<512xf32>
    %6 = mhlo.tuple %1, %batch_mean, %batch_var, %5 {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %6 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.div.1174(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.divide %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @aten.neg.1179(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.negate %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @aten.expand.1183(%arg0: tensor<f32>) -> tensor<4x1000xf32> {
    %0 = mhlo.reshape %arg0 : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = mhlo.reshape %1 : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x1000xf32>
    return %3 : tensor<4x1000xf32>
  }
  func.func private @aten.mul.1190(%arg0: tensor<4x1000xf32>, %arg1: tensor<4x1000xf32>) -> tensor<4x1000xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x1000xf32>
    return %0 : tensor<4x1000xf32>
  }
  func.func private @aten.expand.113(%arg0: tensor<f16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    return %3 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.mul.938(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.add.1005(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.relu.1010(%arg0: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.mean.1020(%arg0: tensor<4x512x7x7xf16>) -> tensor<4x512x1x1xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [3, 2] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<4x512xf16>
     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
      %14 = mhlo.add %arg1, %arg2 : tensor<f16>
      mhlo.return %14 : tensor<f16>
    }
    %2 = mhlo.constant dense<49> : tensor<i64>
    %3 = mhlo.constant dense<0> : tensor<i64>
    %4 = mhlo.compare  NE, %2, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %5 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %6 = mhlo.convert %2 : (tensor<i64>) -> tensor<f16>
    %7 = mhlo.divide %5, %6 : tensor<f16>
    %8 = mhlo.constant dense<0x7E00> : tensor<f16>
    %9 = mhlo.select %4, %7, %8 : tensor<i1>, tensor<f16>
    %10 = "mhlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512xf16>
    %11 = mhlo.multiply %1, %10 : tensor<4x512xf16>
    %12 = mhlo.reshape %11 : (tensor<4x512xf16>) -> tensor<4x512x1x1xf16>
    %13 = mhlo.convert %12 : tensor<4x512x1x1xf16>
    return %13 : tensor<4x512x1x1xf16>
  }
  func.func private @aten.view.1037(%arg0: tensor<4x512x1x1xf16>) -> tensor<4x512xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<4x512x1x1xf16>) -> tensor<4x512xf16>
    return %0 : tensor<4x512xf16>
  }
  func.func private @aten.permute.108(%arg0: tensor<1000x512xf16>) -> tensor<512x1000xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[512,1000]{0,1}"} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    return %0 : tensor<512x1000xf16>
  }
  func.func private @aten.addmm.1041(%arg0: tensor<4x512xf16>, %arg1: tensor<512x1000xf16>, %arg2: tensor<1000xf16>) -> tensor<4x1000xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<512x1000xf16>) -> tensor<4x1000xf16>
    %1 = mhlo.reshape %arg2 : (tensor<1000xf16>) -> tensor<1x1000xf16>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1000xf16>) -> tensor<1x1000xf16>
    %3 = mhlo.reshape %2 : (tensor<1x1000xf16>) -> tensor<1000xf16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf16>) -> tensor<4x1000xf16>
    %5 = mhlo.add %0, %4 : tensor<4x1000xf16>
    return %5 : tensor<4x1000xf16>
  }
  func.func private @aten.log_softmax.1060(%arg0: tensor<4x1000xf16>) -> tensor<4x1000xf16> {
    %0 = mhlo.constant dense<0xFC00> : tensor<f16>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
      %10 = mhlo.maximum %arg1, %arg2 : tensor<f16>
      mhlo.return %10 : tensor<f16>
    }
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %3 = mhlo.subtract %arg0, %2 : tensor<4x1000xf16>
    %4 = mhlo.exponential %3 : tensor<4x1000xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%4 init: %5) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
      %10 = mhlo.add %arg1, %arg2 : tensor<f16>
      mhlo.return %10 : tensor<f16>
    }
    %7 = mhlo.log %6 : tensor<4xf16>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %9 = mhlo.subtract %3, %8 : tensor<4x1000xf16>
    return %9 : tensor<4x1000xf16>
  }
  func.func private @aten._log_softmax_backward_data.1200(%arg0: tensor<4x1000xf16>, %arg1: tensor<4x1000xf16>) -> tensor<4x1000xf16> {
    %0 = mhlo.exponential %arg1 : tensor<4x1000xf16>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg2: tensor<f16>, %arg3: tensor<f16>)  {
      %6 = mhlo.add %arg2, %arg3 : tensor<f16>
      mhlo.return %6 : tensor<f16>
    }
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %4 = mhlo.multiply %0, %3 : tensor<4x1000xf16>
    %5 = mhlo.subtract %arg0, %4 : tensor<4x1000xf16>
    return %5 : tensor<4x1000xf16>
  }
  func.func private @aten.permute.1163(%arg0: tensor<1000x512xf16>) -> tensor<512x1000xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[512,1000]{0,1}"} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    return %0 : tensor<512x1000xf16>
  }
  func.func private @aten.permute.1167(%arg0: tensor<512x1000xf16>) -> tensor<1000x512xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func.func private @aten.mm.1210(%arg0: tensor<4x1000xf16>, %arg1: tensor<1000x512xf16>) -> tensor<4x512xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x1000xf16>, tensor<1000x512xf16>) -> tensor<4x512xf16>
    return %0 : tensor<4x512xf16>
  }
  func.func private @aten.view.1215(%arg0: tensor<4x512xf16>) -> tensor<4x512x1x1xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<4x512xf16>) -> tensor<4x512x1x1xf16>
    return %0 : tensor<4x512x1x1xf16>
  }
  func.func private @aten.expand.1219(%arg0: tensor<4x512x1x1xf16>) -> tensor<4x512x7x7xf16> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<4x512x1x1xf16>) -> tensor<4x512x1x1xf16>
    %1 = mhlo.reshape %0 : (tensor<4x512x1x1xf16>) -> tensor<4x512xf16>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x512xf16>) -> tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.div.1225(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<f16>) -> tensor<4x512x7x7xf16> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %1 = mhlo.divide %arg0, %0 : tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.threshold_backward.1231(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %5 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm_backward.1241(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<512xf32>
    %4 = mhlo.multiply %3, %3 : tensor<512xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %7 = mhlo.subtract %4, %6 : tensor<512xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    return %10 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1270(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<512x512x3x3xf16>) -> tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f16[512,512,3,3]{0,1,3,2}, f16[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
  }
  func.func private @aten.threshold_backward.1286(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %5 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm_backward.1296(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<512xf32>
    %4 = mhlo.multiply %3, %3 : tensor<512xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %7 = mhlo.subtract %4, %6 : tensor<512xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    return %10 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1325(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<512x512x3x3xf16>) -> tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f16[512,512,3,3]{0,1,3,2}, f16[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
  }
  func.func private @aten.expand.1155(%arg0: tensor<f16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    return %3 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.mul.1341(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.add.1346(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x512x7x7xf16>
    return %0 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.threshold_backward.1351(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %5 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm_backward.1361(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<512xf32>
    %4 = mhlo.multiply %3, %3 : tensor<512xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %7 = mhlo.subtract %4, %6 : tensor<512xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    return %10 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1390(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<512x512x3x3xf16>) -> tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f16[512,512,3,3]{0,1,3,2}, f16[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
  }
  func.func private @aten.threshold_backward.1406(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x512x7x7xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %5 : tensor<4x512x7x7xf16>
  }
  func.func private @aten.native_batch_norm_backward.1416(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<512xf32>
    %4 = mhlo.multiply %3, %3 : tensor<512xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %7 = mhlo.subtract %4, %6 : tensor<512xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    return %10 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1445(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<512x256x3x3xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<4x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[512,256,3,3]{0,1,3,2}, f16[512]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
  }
  func.func private @aten.native_batch_norm_backward.1466(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<512xf32>
    %4 = mhlo.multiply %3, %3 : tensor<512xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %7 = mhlo.subtract %4, %6 : tensor<512xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,512,7,7]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    return %10 : tuple<tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1495(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<512x256x1x1xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (tensor<1x1x256x512xf16>) -> tensor<1x1x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<4x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[512,256,1,1]{0,1,3,2}, f16[512]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
  }
  func.func private @aten.expand.1147(%arg0: tensor<f16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    return %3 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.mul.1461(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.add.1511(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.threshold_backward.1516(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %5 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm_backward.1526(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<256xf32>
    %4 = mhlo.multiply %3, %3 : tensor<256xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %7 = mhlo.subtract %4, %6 : tensor<256xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    return %10 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1555(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<256x256x3x3xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[256,256,3,3]{0,1,3,2}, f16[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
  }
  func.func private @aten.threshold_backward.1571(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %5 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm_backward.1581(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<256xf32>
    %4 = mhlo.multiply %3, %3 : tensor<256xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %7 = mhlo.subtract %4, %6 : tensor<256xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    return %10 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1610(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<256x256x3x3xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[256,256,3,3]{0,1,3,2}, f16[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
  }
  func.func private @aten.expand.1139(%arg0: tensor<f16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    return %3 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.mul.1626(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.add.1631(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    return %0 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.threshold_backward.1636(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %5 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm_backward.1646(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<256xf32>
    %4 = mhlo.multiply %3, %3 : tensor<256xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %7 = mhlo.subtract %4, %6 : tensor<256xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    return %10 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1675(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<256x256x3x3xf16>) -> tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f16[256,256,3,3]{0,1,3,2}, f16[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
  }
  func.func private @aten.threshold_backward.1691(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x256x14x14xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %5 : tensor<4x256x14x14xf16>
  }
  func.func private @aten.native_batch_norm_backward.1701(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<256xf32>
    %4 = mhlo.multiply %3, %3 : tensor<256xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %7 = mhlo.subtract %4, %6 : tensor<256xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    return %10 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1730(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<256x128x3x3xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<4x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[256,128,3,3]{0,1,3,2}, f16[256]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
  }
  func.func private @aten.native_batch_norm_backward.1751(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<256xf32>
    %4 = mhlo.multiply %3, %3 : tensor<256xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %7 = mhlo.subtract %4, %6 : tensor<256xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,256,14,14]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    return %10 : tuple<tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1780(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<256x128x1x1xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (tensor<1x1x128x256xf16>) -> tensor<1x1x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<4x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[256,128,1,1]{0,1,3,2}, f16[256]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
  }
  func.func private @aten.expand.1131(%arg0: tensor<f16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    return %3 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.mul.1746(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.add.1796(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.threshold_backward.1801(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %5 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm_backward.1811(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<128xf32>
    %4 = mhlo.multiply %3, %3 : tensor<128xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %7 = mhlo.subtract %4, %6 : tensor<128xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    return %10 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1840(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<128x128x3x3xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[128,128,3,3]{0,1,3,2}, f16[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
  }
  func.func private @aten.threshold_backward.1856(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %5 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm_backward.1866(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<128xf32>
    %4 = mhlo.multiply %3, %3 : tensor<128xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %7 = mhlo.subtract %4, %6 : tensor<128xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    return %10 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1895(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<128x128x3x3xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[128,128,3,3]{0,1,3,2}, f16[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
  }
  func.func private @aten.expand.1123(%arg0: tensor<f16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    return %3 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.mul.1911(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.add.1916(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    return %0 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.threshold_backward.1921(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %5 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm_backward.1931(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<128xf32>
    %4 = mhlo.multiply %3, %3 : tensor<128xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %7 = mhlo.subtract %4, %6 : tensor<128xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    return %10 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1960(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<128x128x3x3xf16>) -> tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f16[128,128,3,3]{0,1,3,2}, f16[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
  }
  func.func private @aten.threshold_backward.1976(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x128x28x28xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %5 : tensor<4x128x28x28xf16>
  }
  func.func private @aten.native_batch_norm_backward.1986(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<128xf32>
    %4 = mhlo.multiply %3, %3 : tensor<128xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %7 = mhlo.subtract %4, %6 : tensor<128xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    return %10 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2015(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<128x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<4x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[128,64,3,3]{0,1,3,2}, f16[128]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
  }
  func.func private @aten.native_batch_norm_backward.2036(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<128xf32>
    %4 = mhlo.multiply %3, %3 : tensor<128xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %7 = mhlo.subtract %4, %6 : tensor<128xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,128,28,28]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    return %10 : tuple<tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2065(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<128x64x1x1xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (tensor<1x1x64x128xf16>) -> tensor<1x1x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<4x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[128,64,1,1]{0,1,3,2}, f16[128]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
  }
  func.func private @aten.expand.1115(%arg0: tensor<f16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    return %3 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.mul.2031(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.add.2081(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.threshold_backward.2086(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %5 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm_backward.2096(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2125(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<64x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<64xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[64,64,3,3]{0,1,3,2}, f16[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
  }
  func.func private @aten.threshold_backward.2141(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %5 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm_backward.2151(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2180(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<64x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<64xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[64,64,3,3]{0,1,3,2}, f16[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
  }
  func.func private @aten.expand.1107(%arg0: tensor<f16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    return %3 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.mul.2196(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.add.2201(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.threshold_backward.2206(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %5 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm_backward.2216(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2245(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<64x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<64xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[64,64,3,3]{0,1,3,2}, f16[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
  }
  func.func private @aten.threshold_backward.2261(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %5 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.native_batch_norm_backward.2271(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2300(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<64x64x3x3xf16>) -> tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<64xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,64,56,56]{3,2,1,0}, f16[64,64,3,3]{0,1,3,2}, f16[64]{0})"} : tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
  }
  func.func private @aten.expand.1099(%arg0: tensor<f16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = mhlo.reshape %1 : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x56x56xf16>
    return %3 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.mul.2316(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.add.2321(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @aten.max_pool2d_with_indices_backward.2334(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.select_and_scatter"(%arg1, %arg0, %0) ({
    ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):
      %2 = mhlo.compare  GE, %arg2, %arg3 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %2 : tensor<i1>
    }, {
    ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):
      %2 = mhlo.add %arg2, %arg3 : tensor<f16>
      mhlo.return %2 : tensor<f16>
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<4x64x112x112xf16>
    return %1 : tensor<4x64x112x112xf16>
  }
  func.func private @aten.threshold_backward.2340(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x112x112xf16>
    %2 = mhlo.compare  GT, %arg1, %1 : (tensor<4x64x112x112xf16>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<4x64x112x112xf16>
    %5 = mhlo.select %2, %arg0, %4 : tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>
    return %5 : tensor<4x64x112x112xf16>
  }
  func.func private @aten.native_batch_norm_backward.2350(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<4x64x112x112xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.convert %arg1 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x112x112xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %9 = mhlo.convert %grad_operand : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    %10 = mhlo.tuple %9, %grad_scale, %grad_offset {xla_shape = "(f16[4,64,112,112]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2379(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<4x3x224x224xf16>, %arg2: tensor<64x3x7x7xf16>) -> tuple<tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[7,7,3,64]{1,0,2,3}"} : (tensor<64x3x7x7xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[7,7,3,64]{1,0,2,3}"} : (tensor<7x7x3x64xf16>) -> tensor<7x7x3x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[3, 4], [3, 4]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x112x112xf16>, tensor<7x7x3x64xf16>) -> tensor<4x3x224x224xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<4x64x112x112xf16>, tensor<f16>) -> tensor<64xf16>
     reducer(%arg3: tensor<f16>, %arg4: tensor<f16>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      mhlo.return %8 : tensor<f16>
    }
    %7 = mhlo.tuple %2, %4, %6 {xla_shape = "(f16[4,3,224,224]{3,2,1,0}, f16[64,3,7,7]{0,1,3,2}, f16[64]{0})"} : tuple<tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
  }
  func.func private @aten.mul.1073(%arg0: tensor<4x1000xf16>, %arg1: tensor<4x1000xf32>) -> tensor<4x1000xf32> {
    %0 = mhlo.convert %arg0 : (tensor<4x1000xf16>) -> tensor<4x1000xf32>
    %1 = mhlo.multiply %0, %arg1 : tensor<4x1000xf32>
    return %1 : tensor<4x1000xf32>
  }
  func.func private @aten.sum.1083(%arg0: tensor<4x1000xf32>) -> tensor<f32> {
    %0 = mhlo.constant dense<4000> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0, 1] : (tensor<4x1000xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %3 = mhlo.add %arg1, %arg2 : tensor<f32>
      mhlo.return %3 : tensor<f32>
    }
    return %2 : tensor<f32>
  }
  func.func private @aten.neg.1089(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.negate %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @aten.div.1093(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.divide %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @aten.view.2415(%arg0: tensor<4x512x1x1xf16>) -> tensor<4x512xf16> {
    %0 = mhlo.reshape %arg0 : (tensor<4x512x1x1xf16>) -> tensor<4x512xf16>
    return %0 : tensor<4x512xf16>
  }
  func.func private @aten.permute.2419(%arg0: tensor<4x512xf16>) -> tensor<512x4xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[512,4]{0,1}"} : (tensor<4x512xf16>) -> tensor<512x4xf16>
    return %0 : tensor<512x4xf16>
  }
  func.func private @aten.mm.2423(%arg0: tensor<512x4xf16>, %arg1: tensor<4x1000xf16>) -> tensor<512x1000xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<512x4xf16>, tensor<4x1000xf16>) -> tensor<512x1000xf16>
    return %0 : tensor<512x1000xf16>
  }
  func.func private @aten.permute.2428(%arg0: tensor<512x1000xf16>) -> tensor<1000x512xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func.func private @aten.sum.2437(%arg0: tensor<4x1000xf16>) -> tensor<1x1000xf32> {
    %0 = mhlo.constant dense<4> : tensor<i64>
    %1 = mhlo.convert %arg0 : (tensor<4x1000xf16>) -> tensor<4x1000xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = mhlo.reduce(%1 init: %2) across dimensions = [0] : (tensor<4x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %5 = mhlo.add %arg1, %arg2 : tensor<f32>
      mhlo.return %5 : tensor<f32>
    }
    %4 = mhlo.reshape %3 : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %4 : tensor<1x1000xf32>
  }
  func.func private @aten.view.2445(%arg0: tensor<1x1000xf32>) -> tensor<1000xf32> {
    %0 = mhlo.reshape %arg0 : (tensor<1x1000xf32>) -> tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
}

