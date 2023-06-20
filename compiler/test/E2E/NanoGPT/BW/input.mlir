// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @forward
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<16384x1536xf32>, %arg1: tensor<384x256x64xf32>, %arg2: tensor<64x256x384xf32>, %arg3: tensor<1x1x256x256xi1>, %arg4: tensor<384xf32>, %arg5: tensor<384x384xf32>, %arg6: tensor<64x256x1xf32>, %arg7: tensor<384x256x256xf32>, %arg8: tensor<64x256x1536xf32>, %arg9: tensor<384x256x64xf32>, %arg10: tensor<384x1536xf32>, %arg11: tensor<64x256x1536xf32>, %arg12: tensor<16384x384xf32>, %arg13: tensor<384xf32>, %arg14: tensor<384xf32>, %arg15: tensor<64x256x1xf32>, %arg16: tensor<384x1152xf32>, %arg17: tensor<1536x384xf32>, %arg18: tensor<384x1152xf32>, %arg19: tensor<384x256x64xf32>, %arg20: tensor<64x256x1xf32>, %arg21: tensor<384x384xf32>, %arg22: tensor<384x64x256xf32>, %arg23: tensor<16384x1536xf32>, %arg24: tensor<384x1536xf32>, %arg25: tensor<64x256x384xf32>, %arg26: tensor<16384x384xf32>, %arg27: tensor<64x6x256x256xf32>, %arg28: tensor<64x6x256x256xf32>, %arg29: tensor<64x256x1xf32>, %arg30: tensor<384xf32>, %arg31: tensor<384x256x64xf32>, %arg32: tensor<384x384xf32>, %arg33: tensor<64x256x1536xf32>, %arg34: tensor<16384x384xf32>, %arg35: tensor<384x64x256xf32>, %arg36: tensor<64x256x1536xf32>, %arg37: tensor<384x1152xf32>, %arg38: tensor<16384x1536xf32>, %arg39: tensor<16384x65xf32>, %arg40: tensor<64x256x1xf32>, %arg41: tensor<16384xi64>, %arg42: tensor<384xf32>, %arg43: tensor<64x256x1536xf32>, %arg44: tensor<64x256x384xf32>, %arg45: tensor<16384x384xf32>, %arg46: tensor<64x256x1xf32>, %arg47: tensor<64x256x1xf32>, %arg48: tensor<16384x384xf32>, %arg49: tensor<384x256x64xf32>, %arg50: tensor<64x6x256x256xf32>, %arg51: tensor<64x256x1536xf32>, %arg52: tensor<384xf32>, %arg53: tensor<64x256xi64>, %arg54: tensor<64x6x256x256xf32>, %arg55: tensor<64x6x256x256xf32>, %arg56: tensor<16384x384xf32>, %arg57: tensor<64x256x1536xf32>, %arg58: tensor<64x256x1xf32>, %arg59: tensor<f32>, %arg60: tensor<64x256x1xf32>, %arg61: tensor<1536x384xf32>, %arg62: tensor<1536x384xf32>, %arg63: tensor<64x256x1xf32>, %arg64: tensor<16384x384xf32>, %arg65: tensor<384x256x256xf32>, %arg66: tensor<64x256x1xf32>, %arg67: tensor<384xf32>, %arg68: tensor<64x256x1536xf32>, %arg69: tensor<64x256x1536xf32>, %arg70: tensor<64x256x384xf32>, %arg71: tensor<64x256x1536xf32>, %arg72: tensor<64x6x256x256xf32>, %arg73: tensor<64x256x1536xf32>, %arg74: tensor<64x256x384xf32>, %arg75: tensor<1536x384xf32>, %arg76: tensor<64x256x1536xf32>, %arg77: tensor<384xf32>, %arg78: tensor<384x384xf32>, %arg79: tensor<64x256x1xf32>, %arg80: tensor<1536x384xf32>, %arg81: tensor<64x256x1536xf32>, %arg82: tensor<1x1x256x256xi1>, %arg83: tensor<16384x384xf32>, %arg84: tensor<64x256x1xf32>, %arg85: tensor<16384x1536xf32>, %arg86: tensor<384x256x64xf32>, %arg87: tensor<64x256x384xf32>, %arg88: tensor<64x256x1xf32>, %arg89: tensor<64x256x1xf32>, %arg90: tensor<384x256x256xf32>, %arg91: tensor<64x256x384xf32>, %arg92: tensor<64x256x384xf32>, %arg93: tensor<64x256x1xf32>, %arg94: tensor<64x256x1536xf32>, %arg95: tensor<16384x384xf32>, %arg96: tensor<384x64x256xf32>, %arg97: tensor<64x256x1xf32>, %arg98: tensor<384xf32>, %arg99: tensor<64x256x1xf32>, %arg100: tensor<64x256x1536xf32>, %arg101: tensor<64x6x256x256xf32>, %arg102: tensor<64x256x1536xf32>, %arg103: tensor<384x256x64xf32>, %arg104: tensor<16384x384xf32>, %arg105: tensor<384x256x64xf32>, %arg106: tensor<64x256x384xf32>, %arg107: tensor<64x256x1536xf32>, %arg108: tensor<64x256x1xf32>, %arg109: tensor<16384x384xf32>, %arg110: tensor<384x256x64xf32>, %arg111: tensor<64x256x1xf32>, %arg112: tensor<1x256xi64>, %arg113: tensor<16384x384xf32>, %arg114: tensor<384x256x64xf32>, %arg115: tensor<64x256x1536xf32>, %arg116: tensor<64x256x384xf32>, %arg117: tensor<64x256x1xf32>, %arg118: tensor<64x256x384xf32>, %arg119: tensor<64x256x1536xf32>, %arg120: tensor<384x256x256xf32>, %arg121: tensor<64x256x1536xf32>, %arg122: tensor<384xf32>, %arg123: tensor<384x1152xf32>, %arg124: tensor<384x256x64xf32>, %arg125: tensor<384x1536xf32>, %arg126: tensor<64x256x1xf32>, %arg127: tensor<384x65xf32>, %arg128: tensor<64x256x1536xf32>, %arg129: tensor<64x256x384xf32>, %arg130: tensor<64x256x1xf32>, %arg131: tensor<64x256x384xf32>, %arg132: tensor<64x256x384xf32>, %arg133: tensor<64x256x384xf32>, %arg134: tensor<64x256x384xf32>, %arg135: tensor<16384x384xf32>, %arg136: tensor<16384x384xf32>, %arg137: tensor<64x256x1xf32>, %arg138: tensor<1x1x256x256xi1>, %arg139: tensor<384x256x256xf32>, %arg140: tensor<64x256x1xf32>, %arg141: tensor<384x64x256xf32>, %arg142: tensor<64x256x1xf32>, %arg143: tensor<16384x384xf32>, %arg144: tensor<16384x1536xf32>, %arg145: tensor<1x1x256x256xi1>, %arg146: tensor<384xf32>, %arg147: tensor<64x256x384xf32>, %arg148: tensor<64x256x384xf32>, %arg149: tensor<1536x384xf32>, %arg150: tensor<64x256x1536xf32>, %arg151: tensor<64x256x384xf32>, %arg152: tensor<64x256x384xf32>, %arg153: tensor<16384x384xf32>, %arg154: tensor<16384x1536xf32>, %arg155: tensor<16384x384xf32>, %arg156: tensor<1x1x256x256xi1>, %arg157: tensor<64x256x1536xf32>, %arg158: tensor<384x64x256xf32>, %arg159: tensor<64x6x256x256xf32>, %arg160: tensor<384xf32>, %arg161: tensor<384x1536xf32>, %arg162: tensor<64x256x384xf32>, %arg163: tensor<384x1152xf32>, %arg164: tensor<16384x384xf32>, %arg165: tensor<16384x384xf32>, %arg166: tensor<384x256x256xf32>, %arg167: tensor<384x1536xf32>, %arg168: tensor<64x6x256x256xf32>, %arg169: tensor<64x256x384xf32>, %arg170: tensor<64x6x256x256xf32>, %arg171: tensor<64x256x384xf32>, %arg172: tensor<64x6x256x256xf32>, %arg173: tensor<384x1152xf32>, %arg174: tensor<64x6x256x256xf32>, %arg175: tensor<64x256x384xf32>, %arg176: tensor<384xf32>, %arg177: tensor<64x256x384xf32>, %arg178: tensor<384x256x64xf32>, %arg179: tensor<384x384xf32>, %arg180: tensor<384x384xf32>, %arg181: tensor<64x256x384xf32>, %arg182: tensor<384x64x256xf32>, %arg183: tensor<1x1x256x256xi1>, %arg184: tensor<384x1536xf32>, %arg185: tensor<64x256x1536xf32>, %arg186: tensor<f32>, %arg187: tensor<16384x65xf32>):
    %0 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<256x384xf32>} : () -> tensor<256x384xf32>
    %1 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<64x256x1152xf32>} : () -> tensor<64x256x1152xf32>
    %2 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<65x384xf32>} : () -> tensor<65x384xf32>
    %3 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<64x256x384xf32>} : () -> tensor<64x256x384xf32>
    %4 = "mhlo.constant"() {value = dense<-1> : tensor<64x256xi64>} : () -> tensor<64x256xi64>
    %5 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<1x256x384xf32>} : () -> tensor<1x256x384xf32>
    %6 = "mhlo.constant"() {value = dense<-1> : tensor<1x256xi64>} : () -> tensor<1x256xi64>
    %7 = "mhlo.constant"() {value = dense<3.840000e+02> : tensor<64x256x1xf32>} : () -> tensor<64x256x1xf32>
    %8 = "mhlo.constant"() {value = dense<3.840000e+02> : tensor<64x256x384xf32>} : () -> tensor<64x256x384xf32>
    %9 = "mhlo.constant"() {value = dense<1.250000e-01> : tensor<64x6x256x256xf32>} : () -> tensor<64x6x256x256xf32>
    %10 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<64x6x256x256xf32>} : () -> tensor<64x6x256x256xf32>
    %11 = "mhlo.constant"() {value = dense<5.000000e-01> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %12 = "mhlo.constant"() {value = dense<3.000000e+00> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %13 = "mhlo.constant"() {value = dense<2.000000e+00> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %14 = "mhlo.constant"() {value = dense<4.471500e-02> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %15 = "mhlo.constant"() {value = dense<0.797884583> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %16 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<64x256x1536xf32>} : () -> tensor<64x256x1536xf32>
    %17 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // %18 = "mhlo.custom_call"(%arg186, %arg39, %arg41, %arg59) {api_version = 1 : i32, backend_config = "", byteir_attrs = {ignore_index = -1 : i64, reduction = 1 : i64}, call_target_name = "byteir.nll_loss_backward", called_computations = [], has_side_effect = false} : (tensor<f32>, tensor<16384x65xf32>, tensor<16384xi64>, tensor<f32>) -> tensor<16384x65xf32>
    %18 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<16384x65xf32>} : () -> tensor<16384x65xf32>
    %19 = "mhlo.exponential"(%arg39) : (tensor<16384x65xf32>) -> tensor<16384x65xf32>
    %20 = "mhlo.reduce"(%18, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<16384x65xf32>, tensor<f32>) -> tensor<16384xf32>
    %21 = "mhlo.reshape"(%20) : (tensor<16384xf32>) -> tensor<16384x1xf32>
    %22 = "mhlo.broadcast_in_dim"(%21) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<16384x1xf32>) -> tensor<16384x65xf32>
    %23 = "mhlo.multiply"(%19, %22) : (tensor<16384x65xf32>, tensor<16384x65xf32>) -> tensor<16384x65xf32>
    %24 = "mhlo.subtract"(%18, %23) : (tensor<16384x65xf32>, tensor<16384x65xf32>) -> tensor<16384x65xf32>
    %25 = "mhlo.add"(%arg187, %24) : (tensor<16384x65xf32>, tensor<16384x65xf32>) -> tensor<16384x65xf32>
    %26 = "mhlo.transpose"(%25) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x65xf32>) -> tensor<65x16384xf32>
    %27 = "mhlo.dot"(%26, %arg109) : (tensor<65x16384xf32>, tensor<16384x384xf32>) -> tensor<65x384xf32>
    %28 = "mhlo.transpose"(%arg127) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x65xf32>) -> tensor<65x384xf32>
    %29 = "mhlo.dot"(%25, %28) : (tensor<16384x65xf32>, tensor<65x384xf32>) -> tensor<16384x384xf32>
    %30 = "mhlo.reshape"(%29) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %31 = "mhlo.broadcast_in_dim"(%arg40) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %32 = "mhlo.subtract"(%arg134, %31) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %33 = "mhlo.broadcast_in_dim"(%arg108) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %34 = "mhlo.multiply"(%32, %33) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %35 = "mhlo.broadcast_in_dim"(%arg14) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %36 = "mhlo.multiply"(%30, %35) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %37 = "mhlo.multiply"(%36, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %38 = "mhlo.reduce"(%36, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %39 = "mhlo.reshape"(%38) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %40 = "mhlo.multiply"(%36, %34) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %41 = "mhlo.reduce"(%40, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %42 = "mhlo.reshape"(%41) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %43 = "mhlo.broadcast_in_dim"(%42) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %44 = "mhlo.multiply"(%34, %43) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %45 = "mhlo.broadcast_in_dim"(%39) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %46 = "mhlo.subtract"(%37, %45) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %47 = "mhlo.subtract"(%46, %44) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %48 = "mhlo.divide"(%arg108, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %49 = "mhlo.broadcast_in_dim"(%48) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %50 = "mhlo.multiply"(%49, %47) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %51 = "mhlo.multiply"(%30, %34) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %52 = "mhlo.reduce"(%51, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %53 = "mhlo.reduce"(%30, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %54 = "mhlo.multiply"(%50, %arg181) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %55 = "mhlo.reshape"(%54) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %56 = "mhlo.transpose"(%arg17) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %57 = "mhlo.dot"(%55, %56) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %58 = "mhlo.transpose"(%55) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %59 = "mhlo.dot"(%58, %arg23) : (tensor<384x16384xf32>, tensor<16384x1536xf32>) -> tensor<384x1536xf32>
    %60 = "mhlo.reduce"(%55, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %61 = "mhlo.reshape"(%57) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %62 = "mhlo.multiply"(%61, %arg128) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %63 = "mhlo.multiply"(%61, %arg107) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %64 = "mhlo.multiply"(%arg119, %arg119) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %65 = "mhlo.subtract"(%16, %64) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %66 = "mhlo.multiply"(%62, %65) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %67 = "mhlo.multiply"(%66, %15) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %68 = "mhlo.multiply"(%67, %14) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %69 = "mhlo.power"(%arg150, %13) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %70 = "mhlo.multiply"(%69, %12) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %71 = "mhlo.multiply"(%68, %70) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %72 = "mhlo.add"(%67, %71) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %73 = "mhlo.multiply"(%63, %11) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %74 = "mhlo.add"(%72, %73) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %75 = "mhlo.reshape"(%74) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %76 = "mhlo.transpose"(%arg125) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %77 = "mhlo.dot"(%75, %76) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %78 = "mhlo.transpose"(%75) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1536xf32>) -> tensor<1536x16384xf32>
    %79 = "mhlo.dot"(%78, %arg143) : (tensor<1536x16384xf32>, tensor<16384x384xf32>) -> tensor<1536x384xf32>
    %80 = "mhlo.reduce"(%75, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %81 = "mhlo.reshape"(%77) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %82 = "mhlo.broadcast_in_dim"(%arg58) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %83 = "mhlo.subtract"(%arg152, %82) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %84 = "mhlo.broadcast_in_dim"(%arg66) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %85 = "mhlo.multiply"(%83, %84) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %86 = "mhlo.broadcast_in_dim"(%arg77) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %87 = "mhlo.multiply"(%81, %86) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %88 = "mhlo.multiply"(%87, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %89 = "mhlo.reduce"(%87, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %90 = "mhlo.reshape"(%89) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %91 = "mhlo.multiply"(%87, %85) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %92 = "mhlo.reduce"(%91, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %93 = "mhlo.reshape"(%92) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %94 = "mhlo.broadcast_in_dim"(%93) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %95 = "mhlo.multiply"(%85, %94) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %96 = "mhlo.broadcast_in_dim"(%90) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %97 = "mhlo.subtract"(%88, %96) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %98 = "mhlo.subtract"(%97, %95) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %99 = "mhlo.divide"(%arg66, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %100 = "mhlo.broadcast_in_dim"(%99) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %101 = "mhlo.multiply"(%100, %98) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %102 = "mhlo.multiply"(%81, %85) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %103 = "mhlo.reduce"(%102, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %104 = "mhlo.reduce"(%81, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %105 = "mhlo.add"(%50, %101) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %106 = "mhlo.multiply"(%105, %arg131) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %107 = "mhlo.reshape"(%106) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %108 = "mhlo.transpose"(%arg180) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %109 = "mhlo.dot"(%107, %108) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %110 = "mhlo.transpose"(%107) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %111 = "mhlo.dot"(%110, %arg104) : (tensor<384x16384xf32>, tensor<16384x384xf32>) -> tensor<384x384xf32>
    %112 = "mhlo.reduce"(%107, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %113 = "mhlo.reshape"(%109) : (tensor<16384x384xf32>) -> tensor<64x256x6x64xf32>
    %114 = "mhlo.transpose"(%113) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %115 = "mhlo.reshape"(%114) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %116 = "mhlo.transpose"(%arg120) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x256xf32>) -> tensor<384x256x256xf32>
    %117 = "mhlo.dot_general"(%116, %115) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %118 = "mhlo.transpose"(%arg114) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %119 = "mhlo.dot_general"(%115, %118) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %120 = "mhlo.reshape"(%117) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %121 = "mhlo.reshape"(%119) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %122 = "mhlo.multiply"(%121, %arg72) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %123 = "mhlo.multiply"(%122, %arg159) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %124 = "mhlo.reduce"(%123, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<64x6x256x256xf32>, tensor<f32>) -> tensor<64x6x256xf32>
    %125 = "mhlo.reshape"(%124) : (tensor<64x6x256xf32>) -> tensor<64x6x256x1xf32>
    %126 = "mhlo.broadcast_in_dim"(%125) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<64x6x256x1xf32>) -> tensor<64x6x256x256xf32>
    %127 = "mhlo.multiply"(%arg159, %126) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %128 = "mhlo.subtract"(%123, %127) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %129 = "mhlo.broadcast_in_dim"(%arg156) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %130 = "mhlo.select"(%129, %10, %128) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %131 = "mhlo.multiply"(%130, %9) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %132 = "mhlo.reshape"(%131) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %133 = "mhlo.transpose"(%arg19) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %134 = "mhlo.dot_general"(%133, %132) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x64x256xf32>, tensor<384x256x256xf32>) -> tensor<384x64x256xf32>
    %135 = "mhlo.transpose"(%arg96) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x64x256xf32>) -> tensor<384x256x64xf32>
    %136 = "mhlo.dot_general"(%132, %135) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %137 = "mhlo.reshape"(%134) : (tensor<384x64x256xf32>) -> tensor<64x6x64x256xf32>
    %138 = "mhlo.reshape"(%136) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %139 = "mhlo.transpose"(%120) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %140 = "mhlo.reshape"(%139) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %141 = "mhlo.transpose"(%138) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %142 = "mhlo.reshape"(%141) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %143 = "mhlo.transpose"(%137) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<64x6x64x256xf32>) -> tensor<64x256x6x64xf32>
    %144 = "mhlo.reshape"(%143) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %145 = "tensor.insert_slice"(%140, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 768>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %146 = "tensor.insert_slice"(%144, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %147 = "mhlo.add"(%145, %146) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %148 = "tensor.insert_slice"(%142, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %149 = "mhlo.add"(%147, %148) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %150 = "mhlo.reshape"(%149) : (tensor<64x256x1152xf32>) -> tensor<16384x1152xf32>
    %151 = "mhlo.transpose"(%arg16) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1152xf32>) -> tensor<1152x384xf32>
    %152 = "mhlo.dot"(%150, %151) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    %153 = "mhlo.transpose"(%150) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1152xf32>) -> tensor<1152x16384xf32>
    %154 = "mhlo.dot"(%153, %arg113) : (tensor<1152x16384xf32>, tensor<16384x384xf32>) -> tensor<1152x384xf32>
    %155 = "mhlo.reduce"(%150, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %156 = "mhlo.reshape"(%152) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %157 = "mhlo.broadcast_in_dim"(%arg46) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %158 = "mhlo.subtract"(%arg106, %157) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %159 = "mhlo.broadcast_in_dim"(%arg29) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %160 = "mhlo.multiply"(%158, %159) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %161 = "mhlo.broadcast_in_dim"(%arg52) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %162 = "mhlo.multiply"(%156, %161) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %163 = "mhlo.multiply"(%162, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %164 = "mhlo.reduce"(%162, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %165 = "mhlo.reshape"(%164) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %166 = "mhlo.multiply"(%162, %160) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %167 = "mhlo.reduce"(%166, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %168 = "mhlo.reshape"(%167) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %169 = "mhlo.broadcast_in_dim"(%168) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %170 = "mhlo.multiply"(%160, %169) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %171 = "mhlo.broadcast_in_dim"(%165) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %172 = "mhlo.subtract"(%163, %171) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %173 = "mhlo.subtract"(%172, %170) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %174 = "mhlo.divide"(%arg29, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %175 = "mhlo.broadcast_in_dim"(%174) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %176 = "mhlo.multiply"(%175, %173) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %177 = "mhlo.multiply"(%156, %160) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %178 = "mhlo.reduce"(%177, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %179 = "mhlo.reduce"(%156, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %180 = "mhlo.add"(%105, %176) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %181 = "mhlo.multiply"(%180, %arg133) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %182 = "mhlo.reshape"(%181) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %183 = "mhlo.transpose"(%arg75) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %184 = "mhlo.dot"(%182, %183) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %185 = "mhlo.transpose"(%182) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %186 = "mhlo.dot"(%185, %arg85) : (tensor<384x16384xf32>, tensor<16384x1536xf32>) -> tensor<384x1536xf32>
    %187 = "mhlo.reduce"(%182, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %188 = "mhlo.reshape"(%184) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %189 = "mhlo.multiply"(%188, %arg100) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %190 = "mhlo.multiply"(%188, %arg43) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %191 = "mhlo.multiply"(%arg71, %arg71) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %192 = "mhlo.subtract"(%16, %191) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %193 = "mhlo.multiply"(%189, %192) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %194 = "mhlo.multiply"(%193, %15) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %195 = "mhlo.multiply"(%194, %14) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %196 = "mhlo.power"(%arg115, %13) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %197 = "mhlo.multiply"(%196, %12) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %198 = "mhlo.multiply"(%195, %197) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %199 = "mhlo.add"(%194, %198) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %200 = "mhlo.multiply"(%190, %11) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %201 = "mhlo.add"(%199, %200) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %202 = "mhlo.reshape"(%201) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %203 = "mhlo.transpose"(%arg161) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %204 = "mhlo.dot"(%202, %203) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %205 = "mhlo.transpose"(%202) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1536xf32>) -> tensor<1536x16384xf32>
    %206 = "mhlo.dot"(%205, %arg155) : (tensor<1536x16384xf32>, tensor<16384x384xf32>) -> tensor<1536x384xf32>
    %207 = "mhlo.reduce"(%202, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %208 = "mhlo.reshape"(%204) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %209 = "mhlo.broadcast_in_dim"(%arg137) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %210 = "mhlo.subtract"(%arg116, %209) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %211 = "mhlo.broadcast_in_dim"(%arg117) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %212 = "mhlo.multiply"(%210, %211) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %213 = "mhlo.broadcast_in_dim"(%arg30) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %214 = "mhlo.multiply"(%208, %213) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %215 = "mhlo.multiply"(%214, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %216 = "mhlo.reduce"(%214, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %217 = "mhlo.reshape"(%216) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %218 = "mhlo.multiply"(%214, %212) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %219 = "mhlo.reduce"(%218, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %220 = "mhlo.reshape"(%219) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %221 = "mhlo.broadcast_in_dim"(%220) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %222 = "mhlo.multiply"(%212, %221) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %223 = "mhlo.broadcast_in_dim"(%217) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %224 = "mhlo.subtract"(%215, %223) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %225 = "mhlo.subtract"(%224, %222) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %226 = "mhlo.divide"(%arg117, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %227 = "mhlo.broadcast_in_dim"(%226) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %228 = "mhlo.multiply"(%227, %225) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %229 = "mhlo.multiply"(%208, %212) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %230 = "mhlo.reduce"(%229, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %231 = "mhlo.reduce"(%208, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %232 = "mhlo.add"(%180, %228) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %233 = "mhlo.multiply"(%232, %arg147) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %234 = "mhlo.reshape"(%233) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %235 = "mhlo.transpose"(%arg5) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %236 = "mhlo.dot"(%234, %235) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %237 = "mhlo.transpose"(%234) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %238 = "mhlo.dot"(%237, %arg164) : (tensor<384x16384xf32>, tensor<16384x384xf32>) -> tensor<384x384xf32>
    %239 = "mhlo.reduce"(%234, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %240 = "mhlo.reshape"(%236) : (tensor<16384x384xf32>) -> tensor<64x256x6x64xf32>
    %241 = "mhlo.transpose"(%240) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %242 = "mhlo.reshape"(%241) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %243 = "mhlo.transpose"(%arg166) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x256xf32>) -> tensor<384x256x256xf32>
    %244 = "mhlo.dot_general"(%243, %242) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %245 = "mhlo.transpose"(%arg105) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %246 = "mhlo.dot_general"(%242, %245) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %247 = "mhlo.reshape"(%244) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %248 = "mhlo.reshape"(%246) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %249 = "mhlo.multiply"(%248, %arg174) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %250 = "mhlo.multiply"(%249, %arg168) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %251 = "mhlo.reduce"(%250, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<64x6x256x256xf32>, tensor<f32>) -> tensor<64x6x256xf32>
    %252 = "mhlo.reshape"(%251) : (tensor<64x6x256xf32>) -> tensor<64x6x256x1xf32>
    %253 = "mhlo.broadcast_in_dim"(%252) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<64x6x256x1xf32>) -> tensor<64x6x256x256xf32>
    %254 = "mhlo.multiply"(%arg168, %253) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %255 = "mhlo.subtract"(%250, %254) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %256 = "mhlo.broadcast_in_dim"(%arg145) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %257 = "mhlo.select"(%256, %10, %255) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %258 = "mhlo.multiply"(%257, %9) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %259 = "mhlo.reshape"(%258) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %260 = "mhlo.transpose"(%arg110) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %261 = "mhlo.dot_general"(%260, %259) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x64x256xf32>, tensor<384x256x256xf32>) -> tensor<384x64x256xf32>
    %262 = "mhlo.transpose"(%arg158) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x64x256xf32>) -> tensor<384x256x64xf32>
    %263 = "mhlo.dot_general"(%259, %262) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %264 = "mhlo.reshape"(%261) : (tensor<384x64x256xf32>) -> tensor<64x6x64x256xf32>
    %265 = "mhlo.reshape"(%263) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %266 = "mhlo.transpose"(%247) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %267 = "mhlo.reshape"(%266) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %268 = "mhlo.transpose"(%265) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %269 = "mhlo.reshape"(%268) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %270 = "mhlo.transpose"(%264) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<64x6x64x256xf32>) -> tensor<64x256x6x64xf32>
    %271 = "mhlo.reshape"(%270) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %272 = "tensor.insert_slice"(%267, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 768>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %273 = "tensor.insert_slice"(%271, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %274 = "mhlo.add"(%272, %273) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %275 = "tensor.insert_slice"(%269, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %276 = "mhlo.add"(%274, %275) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %277 = "mhlo.reshape"(%276) : (tensor<64x256x1152xf32>) -> tensor<16384x1152xf32>
    %278 = "mhlo.transpose"(%arg173) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1152xf32>) -> tensor<1152x384xf32>
    %279 = "mhlo.dot"(%277, %278) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    %280 = "mhlo.transpose"(%277) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1152xf32>) -> tensor<1152x16384xf32>
    %281 = "mhlo.dot"(%280, %arg136) : (tensor<1152x16384xf32>, tensor<16384x384xf32>) -> tensor<1152x384xf32>
    %282 = "mhlo.reduce"(%277, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %283 = "mhlo.reshape"(%279) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %284 = "mhlo.broadcast_in_dim"(%arg99) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %285 = "mhlo.subtract"(%arg169, %284) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %286 = "mhlo.broadcast_in_dim"(%arg130) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %287 = "mhlo.multiply"(%285, %286) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %288 = "mhlo.broadcast_in_dim"(%arg4) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %289 = "mhlo.multiply"(%283, %288) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %290 = "mhlo.multiply"(%289, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %291 = "mhlo.reduce"(%289, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %292 = "mhlo.reshape"(%291) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %293 = "mhlo.multiply"(%289, %287) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %294 = "mhlo.reduce"(%293, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %295 = "mhlo.reshape"(%294) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %296 = "mhlo.broadcast_in_dim"(%295) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %297 = "mhlo.multiply"(%287, %296) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %298 = "mhlo.broadcast_in_dim"(%292) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %299 = "mhlo.subtract"(%290, %298) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %300 = "mhlo.subtract"(%299, %297) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %301 = "mhlo.divide"(%arg130, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %302 = "mhlo.broadcast_in_dim"(%301) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %303 = "mhlo.multiply"(%302, %300) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %304 = "mhlo.multiply"(%283, %287) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %305 = "mhlo.reduce"(%304, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %306 = "mhlo.reduce"(%283, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %307 = "mhlo.add"(%232, %303) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %308 = "mhlo.multiply"(%307, %arg171) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %309 = "mhlo.reshape"(%308) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %310 = "mhlo.transpose"(%arg149) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %311 = "mhlo.dot"(%309, %310) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %312 = "mhlo.transpose"(%309) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %313 = "mhlo.dot"(%312, %arg154) : (tensor<384x16384xf32>, tensor<16384x1536xf32>) -> tensor<384x1536xf32>
    %314 = "mhlo.reduce"(%309, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %315 = "mhlo.reshape"(%311) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %316 = "mhlo.multiply"(%315, %arg33) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %317 = "mhlo.multiply"(%315, %arg51) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %318 = "mhlo.multiply"(%arg94, %arg94) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %319 = "mhlo.subtract"(%16, %318) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %320 = "mhlo.multiply"(%316, %319) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %321 = "mhlo.multiply"(%320, %15) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %322 = "mhlo.multiply"(%321, %14) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %323 = "mhlo.power"(%arg69, %13) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %324 = "mhlo.multiply"(%323, %12) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %325 = "mhlo.multiply"(%322, %324) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %326 = "mhlo.add"(%321, %325) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %327 = "mhlo.multiply"(%317, %11) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %328 = "mhlo.add"(%326, %327) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %329 = "mhlo.reshape"(%328) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %330 = "mhlo.transpose"(%arg24) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %331 = "mhlo.dot"(%329, %330) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %332 = "mhlo.transpose"(%329) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1536xf32>) -> tensor<1536x16384xf32>
    %333 = "mhlo.dot"(%332, %arg12) : (tensor<1536x16384xf32>, tensor<16384x384xf32>) -> tensor<1536x384xf32>
    %334 = "mhlo.reduce"(%329, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %335 = "mhlo.reshape"(%331) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %336 = "mhlo.broadcast_in_dim"(%arg84) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %337 = "mhlo.subtract"(%arg132, %336) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %338 = "mhlo.broadcast_in_dim"(%arg79) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %339 = "mhlo.multiply"(%337, %338) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %340 = "mhlo.broadcast_in_dim"(%arg67) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %341 = "mhlo.multiply"(%335, %340) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %342 = "mhlo.multiply"(%341, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %343 = "mhlo.reduce"(%341, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %344 = "mhlo.reshape"(%343) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %345 = "mhlo.multiply"(%341, %339) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %346 = "mhlo.reduce"(%345, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %347 = "mhlo.reshape"(%346) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %348 = "mhlo.broadcast_in_dim"(%347) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %349 = "mhlo.multiply"(%339, %348) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %350 = "mhlo.broadcast_in_dim"(%344) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %351 = "mhlo.subtract"(%342, %350) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %352 = "mhlo.subtract"(%351, %349) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %353 = "mhlo.divide"(%arg79, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %354 = "mhlo.broadcast_in_dim"(%353) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %355 = "mhlo.multiply"(%354, %352) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %356 = "mhlo.multiply"(%335, %339) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %357 = "mhlo.reduce"(%356, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %358 = "mhlo.reduce"(%335, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %359 = "mhlo.add"(%307, %355) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %360 = "mhlo.multiply"(%359, %arg87) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %361 = "mhlo.reshape"(%360) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %362 = "mhlo.transpose"(%arg32) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %363 = "mhlo.dot"(%361, %362) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %364 = "mhlo.transpose"(%361) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %365 = "mhlo.dot"(%364, %arg56) : (tensor<384x16384xf32>, tensor<16384x384xf32>) -> tensor<384x384xf32>
    %366 = "mhlo.reduce"(%361, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %367 = "mhlo.reshape"(%363) : (tensor<16384x384xf32>) -> tensor<64x256x6x64xf32>
    %368 = "mhlo.transpose"(%367) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %369 = "mhlo.reshape"(%368) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %370 = "mhlo.transpose"(%arg7) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x256xf32>) -> tensor<384x256x256xf32>
    %371 = "mhlo.dot_general"(%370, %369) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %372 = "mhlo.transpose"(%arg178) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %373 = "mhlo.dot_general"(%369, %372) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %374 = "mhlo.reshape"(%371) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %375 = "mhlo.reshape"(%373) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %376 = "mhlo.multiply"(%375, %arg172) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %377 = "mhlo.multiply"(%376, %arg170) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %378 = "mhlo.reduce"(%377, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<64x6x256x256xf32>, tensor<f32>) -> tensor<64x6x256xf32>
    %379 = "mhlo.reshape"(%378) : (tensor<64x6x256xf32>) -> tensor<64x6x256x1xf32>
    %380 = "mhlo.broadcast_in_dim"(%379) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<64x6x256x1xf32>) -> tensor<64x6x256x256xf32>
    %381 = "mhlo.multiply"(%arg170, %380) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %382 = "mhlo.subtract"(%377, %381) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %383 = "mhlo.broadcast_in_dim"(%arg183) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %384 = "mhlo.select"(%383, %10, %382) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %385 = "mhlo.multiply"(%384, %9) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %386 = "mhlo.reshape"(%385) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %387 = "mhlo.transpose"(%arg103) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %388 = "mhlo.dot_general"(%387, %386) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x64x256xf32>, tensor<384x256x256xf32>) -> tensor<384x64x256xf32>
    %389 = "mhlo.transpose"(%arg141) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x64x256xf32>) -> tensor<384x256x64xf32>
    %390 = "mhlo.dot_general"(%386, %389) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %391 = "mhlo.reshape"(%388) : (tensor<384x64x256xf32>) -> tensor<64x6x64x256xf32>
    %392 = "mhlo.reshape"(%390) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %393 = "mhlo.transpose"(%374) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %394 = "mhlo.reshape"(%393) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %395 = "mhlo.transpose"(%392) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %396 = "mhlo.reshape"(%395) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %397 = "mhlo.transpose"(%391) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<64x6x64x256xf32>) -> tensor<64x256x6x64xf32>
    %398 = "mhlo.reshape"(%397) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %399 = "tensor.insert_slice"(%394, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 768>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %400 = "tensor.insert_slice"(%398, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %401 = "mhlo.add"(%399, %400) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %402 = "tensor.insert_slice"(%396, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %403 = "mhlo.add"(%401, %402) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %404 = "mhlo.reshape"(%403) : (tensor<64x256x1152xf32>) -> tensor<16384x1152xf32>
    %405 = "mhlo.transpose"(%arg37) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1152xf32>) -> tensor<1152x384xf32>
    %406 = "mhlo.dot"(%404, %405) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    %407 = "mhlo.transpose"(%404) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1152xf32>) -> tensor<1152x16384xf32>
    %408 = "mhlo.dot"(%407, %arg153) : (tensor<1152x16384xf32>, tensor<16384x384xf32>) -> tensor<1152x384xf32>
    %409 = "mhlo.reduce"(%404, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %410 = "mhlo.reshape"(%406) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %411 = "mhlo.broadcast_in_dim"(%arg93) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %412 = "mhlo.subtract"(%arg74, %411) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %413 = "mhlo.broadcast_in_dim"(%arg97) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %414 = "mhlo.multiply"(%412, %413) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %415 = "mhlo.broadcast_in_dim"(%arg42) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %416 = "mhlo.multiply"(%410, %415) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %417 = "mhlo.multiply"(%416, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %418 = "mhlo.reduce"(%416, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %419 = "mhlo.reshape"(%418) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %420 = "mhlo.multiply"(%416, %414) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %421 = "mhlo.reduce"(%420, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %422 = "mhlo.reshape"(%421) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %423 = "mhlo.broadcast_in_dim"(%422) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %424 = "mhlo.multiply"(%414, %423) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %425 = "mhlo.broadcast_in_dim"(%419) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %426 = "mhlo.subtract"(%417, %425) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %427 = "mhlo.subtract"(%426, %424) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %428 = "mhlo.divide"(%arg97, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %429 = "mhlo.broadcast_in_dim"(%428) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %430 = "mhlo.multiply"(%429, %427) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %431 = "mhlo.multiply"(%410, %414) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %432 = "mhlo.reduce"(%431, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %433 = "mhlo.reduce"(%410, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %434 = "mhlo.add"(%359, %430) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %435 = "mhlo.multiply"(%434, %arg91) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %436 = "mhlo.reshape"(%435) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %437 = "mhlo.transpose"(%arg61) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %438 = "mhlo.dot"(%436, %437) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %439 = "mhlo.transpose"(%436) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %440 = "mhlo.dot"(%439, %arg0) : (tensor<384x16384xf32>, tensor<16384x1536xf32>) -> tensor<384x1536xf32>
    %441 = "mhlo.reduce"(%436, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %442 = "mhlo.reshape"(%438) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %443 = "mhlo.multiply"(%442, %arg121) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %444 = "mhlo.multiply"(%442, %arg102) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %445 = "mhlo.multiply"(%arg73, %arg73) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %446 = "mhlo.subtract"(%16, %445) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %447 = "mhlo.multiply"(%443, %446) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %448 = "mhlo.multiply"(%447, %15) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %449 = "mhlo.multiply"(%448, %14) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %450 = "mhlo.power"(%arg157, %13) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %451 = "mhlo.multiply"(%450, %12) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %452 = "mhlo.multiply"(%449, %451) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %453 = "mhlo.add"(%448, %452) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %454 = "mhlo.multiply"(%444, %11) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %455 = "mhlo.add"(%453, %454) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %456 = "mhlo.reshape"(%455) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %457 = "mhlo.transpose"(%arg184) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %458 = "mhlo.dot"(%456, %457) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %459 = "mhlo.transpose"(%456) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1536xf32>) -> tensor<1536x16384xf32>
    %460 = "mhlo.dot"(%459, %arg26) : (tensor<1536x16384xf32>, tensor<16384x384xf32>) -> tensor<1536x384xf32>
    %461 = "mhlo.reduce"(%456, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %462 = "mhlo.reshape"(%458) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %463 = "mhlo.broadcast_in_dim"(%arg126) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %464 = "mhlo.subtract"(%arg118, %463) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %465 = "mhlo.broadcast_in_dim"(%arg15) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %466 = "mhlo.multiply"(%464, %465) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %467 = "mhlo.broadcast_in_dim"(%arg13) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %468 = "mhlo.multiply"(%462, %467) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %469 = "mhlo.multiply"(%468, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %470 = "mhlo.reduce"(%468, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %471 = "mhlo.reshape"(%470) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %472 = "mhlo.multiply"(%468, %466) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %473 = "mhlo.reduce"(%472, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %474 = "mhlo.reshape"(%473) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %475 = "mhlo.broadcast_in_dim"(%474) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %476 = "mhlo.multiply"(%466, %475) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %477 = "mhlo.broadcast_in_dim"(%471) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %478 = "mhlo.subtract"(%469, %477) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %479 = "mhlo.subtract"(%478, %476) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %480 = "mhlo.divide"(%arg15, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %481 = "mhlo.broadcast_in_dim"(%480) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %482 = "mhlo.multiply"(%481, %479) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %483 = "mhlo.multiply"(%462, %466) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %484 = "mhlo.reduce"(%483, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %485 = "mhlo.reduce"(%462, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %486 = "mhlo.add"(%434, %482) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %487 = "mhlo.multiply"(%486, %arg175) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %488 = "mhlo.reshape"(%487) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %489 = "mhlo.transpose"(%arg21) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %490 = "mhlo.dot"(%488, %489) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %491 = "mhlo.transpose"(%488) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %492 = "mhlo.dot"(%491, %arg95) : (tensor<384x16384xf32>, tensor<16384x384xf32>) -> tensor<384x384xf32>
    %493 = "mhlo.reduce"(%488, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %494 = "mhlo.reshape"(%490) : (tensor<16384x384xf32>) -> tensor<64x256x6x64xf32>
    %495 = "mhlo.transpose"(%494) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %496 = "mhlo.reshape"(%495) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %497 = "mhlo.transpose"(%arg90) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x256xf32>) -> tensor<384x256x256xf32>
    %498 = "mhlo.dot_general"(%497, %496) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %499 = "mhlo.transpose"(%arg49) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %500 = "mhlo.dot_general"(%496, %499) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %501 = "mhlo.reshape"(%498) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %502 = "mhlo.reshape"(%500) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %503 = "mhlo.multiply"(%502, %arg55) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %504 = "mhlo.multiply"(%503, %arg54) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %505 = "mhlo.reduce"(%504, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<64x6x256x256xf32>, tensor<f32>) -> tensor<64x6x256xf32>
    %506 = "mhlo.reshape"(%505) : (tensor<64x6x256xf32>) -> tensor<64x6x256x1xf32>
    %507 = "mhlo.broadcast_in_dim"(%506) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<64x6x256x1xf32>) -> tensor<64x6x256x256xf32>
    %508 = "mhlo.multiply"(%arg54, %507) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %509 = "mhlo.subtract"(%504, %508) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %510 = "mhlo.broadcast_in_dim"(%arg82) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %511 = "mhlo.select"(%510, %10, %509) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %512 = "mhlo.multiply"(%511, %9) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %513 = "mhlo.reshape"(%512) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %514 = "mhlo.transpose"(%arg9) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %515 = "mhlo.dot_general"(%514, %513) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x64x256xf32>, tensor<384x256x256xf32>) -> tensor<384x64x256xf32>
    %516 = "mhlo.transpose"(%arg22) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x64x256xf32>) -> tensor<384x256x64xf32>
    %517 = "mhlo.dot_general"(%513, %516) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %518 = "mhlo.reshape"(%515) : (tensor<384x64x256xf32>) -> tensor<64x6x64x256xf32>
    %519 = "mhlo.reshape"(%517) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %520 = "mhlo.transpose"(%501) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %521 = "mhlo.reshape"(%520) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %522 = "mhlo.transpose"(%519) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %523 = "mhlo.reshape"(%522) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %524 = "mhlo.transpose"(%518) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<64x6x64x256xf32>) -> tensor<64x256x6x64xf32>
    %525 = "mhlo.reshape"(%524) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %526 = "tensor.insert_slice"(%521, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 768>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %527 = "tensor.insert_slice"(%525, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %528 = "mhlo.add"(%526, %527) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %529 = "tensor.insert_slice"(%523, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %530 = "mhlo.add"(%528, %529) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %531 = "mhlo.reshape"(%530) : (tensor<64x256x1152xf32>) -> tensor<16384x1152xf32>
    %532 = "mhlo.transpose"(%arg18) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1152xf32>) -> tensor<1152x384xf32>
    %533 = "mhlo.dot"(%531, %532) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    %534 = "mhlo.transpose"(%531) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1152xf32>) -> tensor<1152x16384xf32>
    %535 = "mhlo.dot"(%534, %arg34) : (tensor<1152x16384xf32>, tensor<16384x384xf32>) -> tensor<1152x384xf32>
    %536 = "mhlo.reduce"(%531, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %537 = "mhlo.reshape"(%533) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %538 = "mhlo.broadcast_in_dim"(%arg63) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %539 = "mhlo.subtract"(%arg70, %538) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %540 = "mhlo.broadcast_in_dim"(%arg89) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %541 = "mhlo.multiply"(%539, %540) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %542 = "mhlo.broadcast_in_dim"(%arg176) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %543 = "mhlo.multiply"(%537, %542) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %544 = "mhlo.multiply"(%543, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %545 = "mhlo.reduce"(%543, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %546 = "mhlo.reshape"(%545) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %547 = "mhlo.multiply"(%543, %541) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %548 = "mhlo.reduce"(%547, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %549 = "mhlo.reshape"(%548) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %550 = "mhlo.broadcast_in_dim"(%549) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %551 = "mhlo.multiply"(%541, %550) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %552 = "mhlo.broadcast_in_dim"(%546) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %553 = "mhlo.subtract"(%544, %552) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %554 = "mhlo.subtract"(%553, %551) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %555 = "mhlo.divide"(%arg89, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %556 = "mhlo.broadcast_in_dim"(%555) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %557 = "mhlo.multiply"(%556, %554) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %558 = "mhlo.multiply"(%537, %541) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %559 = "mhlo.reduce"(%558, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %560 = "mhlo.reduce"(%537, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %561 = "mhlo.add"(%486, %557) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %562 = "mhlo.multiply"(%561, %arg25) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %563 = "mhlo.reshape"(%562) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %564 = "mhlo.transpose"(%arg62) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %565 = "mhlo.dot"(%563, %564) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %566 = "mhlo.transpose"(%563) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %567 = "mhlo.dot"(%566, %arg144) : (tensor<384x16384xf32>, tensor<16384x1536xf32>) -> tensor<384x1536xf32>
    %568 = "mhlo.reduce"(%563, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %569 = "mhlo.reshape"(%565) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %570 = "mhlo.multiply"(%569, %arg81) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %571 = "mhlo.multiply"(%569, %arg36) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %572 = "mhlo.multiply"(%arg76, %arg76) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %573 = "mhlo.subtract"(%16, %572) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %574 = "mhlo.multiply"(%570, %573) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %575 = "mhlo.multiply"(%574, %15) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %576 = "mhlo.multiply"(%575, %14) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %577 = "mhlo.power"(%arg11, %13) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %578 = "mhlo.multiply"(%577, %12) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %579 = "mhlo.multiply"(%576, %578) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %580 = "mhlo.add"(%575, %579) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %581 = "mhlo.multiply"(%571, %11) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %582 = "mhlo.add"(%580, %581) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %583 = "mhlo.reshape"(%582) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %584 = "mhlo.transpose"(%arg10) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %585 = "mhlo.dot"(%583, %584) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %586 = "mhlo.transpose"(%583) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1536xf32>) -> tensor<1536x16384xf32>
    %587 = "mhlo.dot"(%586, %arg83) : (tensor<1536x16384xf32>, tensor<16384x384xf32>) -> tensor<1536x384xf32>
    %588 = "mhlo.reduce"(%583, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %589 = "mhlo.reshape"(%585) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %590 = "mhlo.broadcast_in_dim"(%arg88) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %591 = "mhlo.subtract"(%arg92, %590) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %592 = "mhlo.broadcast_in_dim"(%arg20) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %593 = "mhlo.multiply"(%591, %592) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %594 = "mhlo.broadcast_in_dim"(%arg146) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %595 = "mhlo.multiply"(%589, %594) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %596 = "mhlo.multiply"(%595, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %597 = "mhlo.reduce"(%595, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %598 = "mhlo.reshape"(%597) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %599 = "mhlo.multiply"(%595, %593) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %600 = "mhlo.reduce"(%599, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %601 = "mhlo.reshape"(%600) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %602 = "mhlo.broadcast_in_dim"(%601) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %603 = "mhlo.multiply"(%593, %602) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %604 = "mhlo.broadcast_in_dim"(%598) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %605 = "mhlo.subtract"(%596, %604) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %606 = "mhlo.subtract"(%605, %603) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %607 = "mhlo.divide"(%arg20, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %608 = "mhlo.broadcast_in_dim"(%607) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %609 = "mhlo.multiply"(%608, %606) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %610 = "mhlo.multiply"(%589, %593) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %611 = "mhlo.reduce"(%610, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %612 = "mhlo.reduce"(%589, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %613 = "mhlo.add"(%561, %609) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %614 = "mhlo.multiply"(%613, %arg44) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %615 = "mhlo.reshape"(%614) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %616 = "mhlo.transpose"(%arg78) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %617 = "mhlo.dot"(%615, %616) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %618 = "mhlo.transpose"(%615) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %619 = "mhlo.dot"(%618, %arg45) : (tensor<384x16384xf32>, tensor<16384x384xf32>) -> tensor<384x384xf32>
    %620 = "mhlo.reduce"(%615, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %621 = "mhlo.reshape"(%617) : (tensor<16384x384xf32>) -> tensor<64x256x6x64xf32>
    %622 = "mhlo.transpose"(%621) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %623 = "mhlo.reshape"(%622) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %624 = "mhlo.transpose"(%arg65) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x256xf32>) -> tensor<384x256x256xf32>
    %625 = "mhlo.dot_general"(%624, %623) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %626 = "mhlo.transpose"(%arg31) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %627 = "mhlo.dot_general"(%623, %626) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %628 = "mhlo.reshape"(%625) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %629 = "mhlo.reshape"(%627) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %630 = "mhlo.multiply"(%629, %arg50) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %631 = "mhlo.multiply"(%630, %arg28) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %632 = "mhlo.reduce"(%631, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<64x6x256x256xf32>, tensor<f32>) -> tensor<64x6x256xf32>
    %633 = "mhlo.reshape"(%632) : (tensor<64x6x256xf32>) -> tensor<64x6x256x1xf32>
    %634 = "mhlo.broadcast_in_dim"(%633) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<64x6x256x1xf32>) -> tensor<64x6x256x256xf32>
    %635 = "mhlo.multiply"(%arg28, %634) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %636 = "mhlo.subtract"(%631, %635) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %637 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %638 = "mhlo.select"(%637, %10, %636) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %639 = "mhlo.multiply"(%638, %9) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %640 = "mhlo.reshape"(%639) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %641 = "mhlo.transpose"(%arg124) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %642 = "mhlo.dot_general"(%641, %640) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x64x256xf32>, tensor<384x256x256xf32>) -> tensor<384x64x256xf32>
    %643 = "mhlo.transpose"(%arg35) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x64x256xf32>) -> tensor<384x256x64xf32>
    %644 = "mhlo.dot_general"(%640, %643) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %645 = "mhlo.reshape"(%642) : (tensor<384x64x256xf32>) -> tensor<64x6x64x256xf32>
    %646 = "mhlo.reshape"(%644) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %647 = "mhlo.transpose"(%628) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %648 = "mhlo.reshape"(%647) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %649 = "mhlo.transpose"(%646) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %650 = "mhlo.reshape"(%649) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %651 = "mhlo.transpose"(%645) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<64x6x64x256xf32>) -> tensor<64x256x6x64xf32>
    %652 = "mhlo.reshape"(%651) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %653 = "tensor.insert_slice"(%648, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 768>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %654 = "tensor.insert_slice"(%652, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %655 = "mhlo.add"(%653, %654) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %656 = "tensor.insert_slice"(%650, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %657 = "mhlo.add"(%655, %656) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %658 = "mhlo.reshape"(%657) : (tensor<64x256x1152xf32>) -> tensor<16384x1152xf32>
    %659 = "mhlo.transpose"(%arg163) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1152xf32>) -> tensor<1152x384xf32>
    %660 = "mhlo.dot"(%658, %659) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    %661 = "mhlo.transpose"(%658) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1152xf32>) -> tensor<1152x16384xf32>
    %662 = "mhlo.dot"(%661, %arg64) : (tensor<1152x16384xf32>, tensor<16384x384xf32>) -> tensor<1152x384xf32>
    %663 = "mhlo.reduce"(%658, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %664 = "mhlo.reshape"(%660) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %665 = "mhlo.broadcast_in_dim"(%arg60) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %666 = "mhlo.subtract"(%arg129, %665) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %667 = "mhlo.broadcast_in_dim"(%arg47) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %668 = "mhlo.multiply"(%666, %667) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %669 = "mhlo.broadcast_in_dim"(%arg122) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %670 = "mhlo.multiply"(%664, %669) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %671 = "mhlo.multiply"(%670, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %672 = "mhlo.reduce"(%670, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %673 = "mhlo.reshape"(%672) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %674 = "mhlo.multiply"(%670, %668) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %675 = "mhlo.reduce"(%674, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %676 = "mhlo.reshape"(%675) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %677 = "mhlo.broadcast_in_dim"(%676) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %678 = "mhlo.multiply"(%668, %677) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %679 = "mhlo.broadcast_in_dim"(%673) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %680 = "mhlo.subtract"(%671, %679) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %681 = "mhlo.subtract"(%680, %678) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %682 = "mhlo.divide"(%arg47, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %683 = "mhlo.broadcast_in_dim"(%682) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %684 = "mhlo.multiply"(%683, %681) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %685 = "mhlo.multiply"(%664, %668) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %686 = "mhlo.reduce"(%685, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %687 = "mhlo.reduce"(%664, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %688 = "mhlo.add"(%613, %684) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %689 = "mhlo.multiply"(%688, %arg177) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %690 = "mhlo.reshape"(%689) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %691 = "mhlo.transpose"(%arg80) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1536x384xf32>) -> tensor<384x1536xf32>
    %692 = "mhlo.dot"(%690, %691) : (tensor<16384x384xf32>, tensor<384x1536xf32>) -> tensor<16384x1536xf32>
    %693 = "mhlo.transpose"(%690) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %694 = "mhlo.dot"(%693, %arg38) : (tensor<384x16384xf32>, tensor<16384x1536xf32>) -> tensor<384x1536xf32>
    %695 = "mhlo.reduce"(%690, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %696 = "mhlo.reshape"(%692) : (tensor<16384x1536xf32>) -> tensor<64x256x1536xf32>
    %697 = "mhlo.multiply"(%696, %arg57) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %698 = "mhlo.multiply"(%696, %arg8) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %699 = "mhlo.multiply"(%arg185, %arg185) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %700 = "mhlo.subtract"(%16, %699) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %701 = "mhlo.multiply"(%697, %700) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %702 = "mhlo.multiply"(%701, %15) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %703 = "mhlo.multiply"(%702, %14) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %704 = "mhlo.power"(%arg68, %13) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %705 = "mhlo.multiply"(%704, %12) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %706 = "mhlo.multiply"(%703, %705) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %707 = "mhlo.add"(%702, %706) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %708 = "mhlo.multiply"(%698, %11) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %709 = "mhlo.add"(%707, %708) : (tensor<64x256x1536xf32>, tensor<64x256x1536xf32>) -> tensor<64x256x1536xf32>
    %710 = "mhlo.reshape"(%709) : (tensor<64x256x1536xf32>) -> tensor<16384x1536xf32>
    %711 = "mhlo.transpose"(%arg167) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1536xf32>) -> tensor<1536x384xf32>
    %712 = "mhlo.dot"(%710, %711) : (tensor<16384x1536xf32>, tensor<1536x384xf32>) -> tensor<16384x384xf32>
    %713 = "mhlo.transpose"(%710) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1536xf32>) -> tensor<1536x16384xf32>
    %714 = "mhlo.dot"(%713, %arg48) : (tensor<1536x16384xf32>, tensor<16384x384xf32>) -> tensor<1536x384xf32>
    %715 = "mhlo.reduce"(%710, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %716 = "mhlo.reshape"(%712) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %717 = "mhlo.broadcast_in_dim"(%arg6) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %718 = "mhlo.subtract"(%arg148, %717) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %719 = "mhlo.broadcast_in_dim"(%arg142) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %720 = "mhlo.multiply"(%718, %719) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %721 = "mhlo.broadcast_in_dim"(%arg98) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %722 = "mhlo.multiply"(%716, %721) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %723 = "mhlo.multiply"(%722, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %724 = "mhlo.reduce"(%722, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %725 = "mhlo.reshape"(%724) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %726 = "mhlo.multiply"(%722, %720) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %727 = "mhlo.reduce"(%726, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %728 = "mhlo.reshape"(%727) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %729 = "mhlo.broadcast_in_dim"(%728) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %730 = "mhlo.multiply"(%720, %729) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %731 = "mhlo.broadcast_in_dim"(%725) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %732 = "mhlo.subtract"(%723, %731) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %733 = "mhlo.subtract"(%732, %730) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %734 = "mhlo.divide"(%arg142, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %735 = "mhlo.broadcast_in_dim"(%734) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %736 = "mhlo.multiply"(%735, %733) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %737 = "mhlo.multiply"(%716, %720) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %738 = "mhlo.reduce"(%737, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %739 = "mhlo.reduce"(%716, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %740 = "mhlo.add"(%688, %736) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %741 = "mhlo.multiply"(%740, %arg2) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %742 = "mhlo.reshape"(%741) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %743 = "mhlo.transpose"(%arg179) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x384xf32>) -> tensor<384x384xf32>
    %744 = "mhlo.dot"(%742, %743) : (tensor<16384x384xf32>, tensor<384x384xf32>) -> tensor<16384x384xf32>
    %745 = "mhlo.transpose"(%742) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x384xf32>) -> tensor<384x16384xf32>
    %746 = "mhlo.dot"(%745, %arg165) : (tensor<384x16384xf32>, tensor<16384x384xf32>) -> tensor<384x384xf32>
    %747 = "mhlo.reduce"(%742, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x384xf32>, tensor<f32>) -> tensor<384xf32>
    %748 = "mhlo.reshape"(%744) : (tensor<16384x384xf32>) -> tensor<64x256x6x64xf32>
    %749 = "mhlo.transpose"(%748) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x256x6x64xf32>) -> tensor<64x6x256x64xf32>
    %750 = "mhlo.reshape"(%749) : (tensor<64x6x256x64xf32>) -> tensor<384x256x64xf32>
    %751 = "mhlo.transpose"(%arg139) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x256xf32>) -> tensor<384x256x256xf32>
    %752 = "mhlo.dot_general"(%751, %750) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %753 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %754 = "mhlo.dot_general"(%750, %753) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x64xf32>, tensor<384x64x256xf32>) -> tensor<384x256x256xf32>
    %755 = "mhlo.reshape"(%752) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %756 = "mhlo.reshape"(%754) : (tensor<384x256x256xf32>) -> tensor<64x6x256x256xf32>
    %757 = "mhlo.multiply"(%756, %arg101) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %758 = "mhlo.multiply"(%757, %arg27) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %759 = "mhlo.reduce"(%758, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<64x6x256x256xf32>, tensor<f32>) -> tensor<64x6x256xf32>
    %760 = "mhlo.reshape"(%759) : (tensor<64x6x256xf32>) -> tensor<64x6x256x1xf32>
    %761 = "mhlo.broadcast_in_dim"(%760) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<64x6x256x1xf32>) -> tensor<64x6x256x256xf32>
    %762 = "mhlo.multiply"(%arg27, %761) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %763 = "mhlo.subtract"(%758, %762) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %764 = "mhlo.broadcast_in_dim"(%arg138) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x256xi1>) -> tensor<64x6x256x256xi1>
    %765 = "mhlo.select"(%764, %10, %763) : (tensor<64x6x256x256xi1>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %766 = "mhlo.multiply"(%765, %9) : (tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    %767 = "mhlo.reshape"(%766) : (tensor<64x6x256x256xf32>) -> tensor<384x256x256xf32>
    %768 = "mhlo.transpose"(%arg86) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x64xf32>) -> tensor<384x64x256xf32>
    %769 = "mhlo.dot_general"(%768, %767) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x64x256xf32>, tensor<384x256x256xf32>) -> tensor<384x64x256xf32>
    %770 = "mhlo.transpose"(%arg182) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x64x256xf32>) -> tensor<384x256x64xf32>
    %771 = "mhlo.dot_general"(%767, %770) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %772 = "mhlo.reshape"(%769) : (tensor<384x64x256xf32>) -> tensor<64x6x64x256xf32>
    %773 = "mhlo.reshape"(%771) : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %774 = "mhlo.transpose"(%755) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %775 = "mhlo.reshape"(%774) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %776 = "mhlo.transpose"(%773) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    %777 = "mhlo.reshape"(%776) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %778 = "mhlo.transpose"(%772) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<64x6x64x256xf32>) -> tensor<64x256x6x64xf32>
    %779 = "mhlo.reshape"(%778) : (tensor<64x256x6x64xf32>) -> tensor<64x256x384xf32>
    %780 = "tensor.insert_slice"(%775, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 768>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %781 = "tensor.insert_slice"(%779, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %782 = "mhlo.add"(%780, %781) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %783 = "tensor.insert_slice"(%777, %1) {operand_segment_sizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 64, 256, 384>, static_strides = array<i64: 1, 1, 1>} : (tensor<64x256x384xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %784 = "mhlo.add"(%782, %783) : (tensor<64x256x1152xf32>, tensor<64x256x1152xf32>) -> tensor<64x256x1152xf32>
    %785 = "mhlo.reshape"(%784) : (tensor<64x256x1152xf32>) -> tensor<16384x1152xf32>
    %786 = "mhlo.transpose"(%arg123) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1152xf32>) -> tensor<1152x384xf32>
    %787 = "mhlo.dot"(%785, %786) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    %788 = "mhlo.transpose"(%785) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16384x1152xf32>) -> tensor<1152x16384xf32>
    %789 = "mhlo.dot"(%788, %arg135) : (tensor<1152x16384xf32>, tensor<16384x384xf32>) -> tensor<1152x384xf32>
    %790 = "mhlo.reduce"(%785, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16384x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %791 = "mhlo.reshape"(%787) : (tensor<16384x384xf32>) -> tensor<64x256x384xf32>
    %792 = "mhlo.broadcast_in_dim"(%arg140) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %793 = "mhlo.subtract"(%arg162, %792) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %794 = "mhlo.broadcast_in_dim"(%arg111) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %795 = "mhlo.multiply"(%793, %794) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %796 = "mhlo.broadcast_in_dim"(%arg160) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<64x256x384xf32>
    %797 = "mhlo.multiply"(%791, %796) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %798 = "mhlo.multiply"(%797, %8) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %799 = "mhlo.reduce"(%797, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %800 = "mhlo.reshape"(%799) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %801 = "mhlo.multiply"(%797, %795) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %802 = "mhlo.reduce"(%801, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<64x256xf32>
    %803 = "mhlo.reshape"(%802) : (tensor<64x256xf32>) -> tensor<64x256x1xf32>
    %804 = "mhlo.broadcast_in_dim"(%803) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %805 = "mhlo.multiply"(%795, %804) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %806 = "mhlo.broadcast_in_dim"(%800) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %807 = "mhlo.subtract"(%798, %806) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %808 = "mhlo.subtract"(%807, %805) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %809 = "mhlo.divide"(%arg111, %7) : (tensor<64x256x1xf32>, tensor<64x256x1xf32>) -> tensor<64x256x1xf32>
    %810 = "mhlo.broadcast_in_dim"(%809) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xf32>) -> tensor<64x256x384xf32>
    %811 = "mhlo.multiply"(%810, %808) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %812 = "mhlo.multiply"(%791, %795) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %813 = "mhlo.reduce"(%812, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %814 = "mhlo.reduce"(%791, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<384xf32>
    %815 = "mhlo.add"(%740, %811) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %816 = "mhlo.multiply"(%815, %arg151) : (tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %817 = "mhlo.reduce"(%816, %17) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<64x256x384xf32>, tensor<f32>) -> tensor<256x384xf32>
    %818 = "mhlo.reshape"(%817) : (tensor<256x384xf32>) -> tensor<1x256x384xf32>
    %819 = "mhlo.compare"(%arg112, %6) {compare_type = #mhlo<comparison_type SIGNED>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<1x256xi64>, tensor<1x256xi64>) -> tensor<1x256xi1>
    %820 = "mhlo.reshape"(%819) : (tensor<1x256xi1>) -> tensor<1x256x1xi1>
    %821 = "mhlo.broadcast_in_dim"(%820) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x256x1xi1>) -> tensor<1x256x384xi1>
    %822 = "mhlo.select"(%821, %5, %818) : (tensor<1x256x384xi1>, tensor<1x256x384xf32>, tensor<1x256x384xf32>) -> tensor<1x256x384xf32>
    %823 = "mhlo.reshape"(%arg112) : (tensor<1x256xi64>) -> tensor<256x1xi64>
    %824 = "mhlo.reshape"(%822) : (tensor<1x256x384xf32>) -> tensor<256x384xf32>
    %825 = "mhlo.scatter"(%0, %823, %824) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<256x384xf32>, tensor<256x1xi64>, tensor<256x384xf32>) -> tensor<256x384xf32>
    %826 = "mhlo.compare"(%arg53, %4) {compare_type = #mhlo<comparison_type SIGNED>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<64x256xi64>, tensor<64x256xi64>) -> tensor<64x256xi1>
    %827 = "mhlo.reshape"(%826) : (tensor<64x256xi1>) -> tensor<64x256x1xi1>
    %828 = "mhlo.broadcast_in_dim"(%827) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<64x256x1xi1>) -> tensor<64x256x384xi1>
    %829 = "mhlo.select"(%828, %3, %816) : (tensor<64x256x384xi1>, tensor<64x256x384xf32>, tensor<64x256x384xf32>) -> tensor<64x256x384xf32>
    %830 = "mhlo.reshape"(%arg53) : (tensor<64x256xi64>) -> tensor<16384x1xi64>
    %831 = "mhlo.reshape"(%829) : (tensor<64x256x384xf32>) -> tensor<16384x384xf32>
    %832 = "mhlo.scatter"(%2, %830, %831) ({
    ^bb0(%arg188: tensor<f32>, %arg189: tensor<f32>):
      %833 = "mhlo.add"(%arg188, %arg189) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%833) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<65x384xf32>, tensor<16384x1xi64>, tensor<16384x384xf32>) -> tensor<65x384xf32>
    "func.return"(%832, %825, %813, %814, %789, %790, %746, %747, %738, %739, %714, %715, %694, %695, %686, %687, %662, %663, %619, %620, %611, %612, %587, %588, %567, %568, %559, %560, %535, %536, %492, %493, %484, %485, %460, %461, %440, %441, %432, %433, %408, %409, %365, %366, %357, %358, %333, %334, %313, %314, %305, %306, %281, %282, %238, %239, %230, %231, %206, %207, %186, %187, %178, %179, %154, %155, %111, %112, %103, %104, %79, %80, %59, %60, %52, %53, %27) : (tensor<65x384xf32>, tensor<256x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<65x384xf32>) -> ()
  }) {function_type = (tensor<16384x1536xf32>, tensor<384x256x64xf32>, tensor<64x256x384xf32>, tensor<1x1x256x256xi1>, tensor<384xf32>, tensor<384x384xf32>, tensor<64x256x1xf32>, tensor<384x256x256xf32>, tensor<64x256x1536xf32>, tensor<384x256x64xf32>, tensor<384x1536xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x256x1xf32>, tensor<384x1152xf32>, tensor<1536x384xf32>, tensor<384x1152xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<384x384xf32>, tensor<384x64x256xf32>, tensor<16384x1536xf32>, tensor<384x1536xf32>, tensor<64x256x384xf32>, tensor<16384x384xf32>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<384x256x64xf32>, tensor<384x384xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<384x64x256xf32>, tensor<64x256x1536xf32>, tensor<384x1152xf32>, tensor<16384x1536xf32>, tensor<16384x65xf32>, tensor<64x256x1xf32>, tensor<16384xi64>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<384x256x64xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1536xf32>, tensor<384xf32>, tensor<64x256xi64>, tensor<64x6x256x256xf32>, tensor<64x6x256x256xf32>, tensor<16384x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x1xf32>, tensor<f32>, tensor<64x256x1xf32>, tensor<1536x384xf32>, tensor<1536x384xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<384x256x256xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<64x256x1536xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<1536x384xf32>, tensor<64x256x1536xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<64x256x1xf32>, tensor<1536x384xf32>, tensor<64x256x1536xf32>, tensor<1x1x256x256xi1>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<16384x1536xf32>, tensor<384x256x64xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1xf32>, tensor<384x256x256xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x1536xf32>, tensor<16384x384xf32>, tensor<384x64x256xf32>, tensor<64x256x1xf32>, tensor<384xf32>, tensor<64x256x1xf32>, tensor<64x256x1536xf32>, tensor<64x6x256x256xf32>, tensor<64x256x1536xf32>, tensor<384x256x64xf32>, tensor<16384x384xf32>, tensor<384x256x64xf32>, tensor<64x256x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1xf32>, tensor<1x256xi64>, tensor<16384x384xf32>, tensor<384x256x64xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<64x256x1536xf32>, tensor<384x256x256xf32>, tensor<64x256x1536xf32>, tensor<384xf32>, tensor<384x1152xf32>, tensor<384x256x64xf32>, tensor<384x1536xf32>, tensor<64x256x1xf32>, tensor<384x65xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<64x256x1xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<64x256x1xf32>, tensor<1x1x256x256xi1>, tensor<384x256x256xf32>, tensor<64x256x1xf32>, tensor<384x64x256xf32>, tensor<64x256x1xf32>, tensor<16384x384xf32>, tensor<16384x1536xf32>, tensor<1x1x256x256xi1>, tensor<384xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<1536x384xf32>, tensor<64x256x1536xf32>, tensor<64x256x384xf32>, tensor<64x256x384xf32>, tensor<16384x384xf32>, tensor<16384x1536xf32>, tensor<16384x384xf32>, tensor<1x1x256x256xi1>, tensor<64x256x1536xf32>, tensor<384x64x256xf32>, tensor<64x6x256x256xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<64x256x384xf32>, tensor<384x1152xf32>, tensor<16384x384xf32>, tensor<16384x384xf32>, tensor<384x256x256xf32>, tensor<384x1536xf32>, tensor<64x6x256x256xf32>, tensor<64x256x384xf32>, tensor<64x6x256x256xf32>, tensor<64x256x384xf32>, tensor<64x6x256x256xf32>, tensor<384x1152xf32>, tensor<64x6x256x256xf32>, tensor<64x256x384xf32>, tensor<384xf32>, tensor<64x256x384xf32>, tensor<384x256x64xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<64x256x384xf32>, tensor<384x64x256xf32>, tensor<1x1x256x256xi1>, tensor<384x1536xf32>, tensor<64x256x1536xf32>, tensor<f32>, tensor<16384x65xf32>) -> (tensor<65x384xf32>, tensor<256x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1152x384xf32>, tensor<1152xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>, tensor<1536xf32>, tensor<384x1536xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<65x384xf32>), sym_name = "forward"} : () -> ()
}) {torch.debug_module_name = "GraphModule"} : () -> ()

