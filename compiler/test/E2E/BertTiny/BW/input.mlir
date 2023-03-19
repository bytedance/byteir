// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<256x128xf32>, %arg3: tensor<2x128x1xf32>, %arg4: tensor<256x128xf32>, %arg5: tensor<2x128xi64>, %arg6: tensor<256x128xf32>, %arg7: tensor<128x128xf32>, %arg8: tensor<4x128x64xf32>, %arg9: tensor<256x512xf32>, %arg10: tensor<2x128x1xf32>, %arg11: tensor<4x64x128xf32>, %arg12: tensor<256x128xf32>, %arg13: tensor<256x128xf32>, %arg14: tensor<2x128x128xf32>, %arg15: tensor<2x128x128xf32>, %arg16: tensor<2x128x1xf32>, %arg17: tensor<2x128x1xf32>, %arg18: tensor<2x128xi64>, %arg19: tensor<2x128x1xf32>, %arg20: tensor<128x512xf32>, %arg21: tensor<4x128x64xf32>, %arg22: tensor<256x512xf32>, %arg23: tensor<2x128x512xf32>, %arg24: tensor<2x128x128xf32>, %arg25: tensor<4x128x64xf32>, %arg26: tensor<512x128xf32>, %arg27: tensor<256x128xf32>, %arg28: tensor<128x512xf32>, %arg29: tensor<4x128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<256x128xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<2x128x128xf32>, %arg35: tensor<128x128xf32>, %arg36: tensor<128x128xf32>, %arg37: tensor<256x128xf32>, %arg38: tensor<2x128x1xf32>, %arg39: tensor<256x128xf32>, %arg40: tensor<4x64x128xf32>, %arg41: tensor<2x2x128x128xf32>, %arg42: tensor<1x128xi64>, %arg43: tensor<256x128xf32>, %arg44: tensor<2x128x1xf32>, %arg45: tensor<128x128xf32>, %arg46: tensor<128x30522xf32>, %arg47: tensor<2x128x1xf32>, %arg48: tensor<512x128xf32>, %arg49: tensor<128x128xf32>, %arg50: tensor<2x128x1xf32>, %arg51: tensor<4x128x64xf32>, %arg52: tensor<4x128x128xf32>, %arg53: tensor<128x128xf32>, %arg54: tensor<128x128xf32>, %arg55: tensor<2x128x1xf32>, %arg56: tensor<2x128x1xf32>, %arg57: tensor<128x128xf32>, %arg58: tensor<2x128x128xf32>, %arg59: tensor<128x128xf32>, %arg60: tensor<2x128x512xf32>, %arg61: tensor<2x128x128xf32>, %arg62: tensor<2x2x128x128xf32>, %arg63: tensor<128xf32>, %arg64: tensor<256x128xf32>, %arg65: tensor<128xf32>, %arg66: tensor<2x128x1xf32>, %arg67: tensor<256x128xf32>, %arg68: tensor<2x128x30522xf32>):
    %0 = "mhlo.constant"() {value = dense<0> : tensor<2x128xi64>} : () -> tensor<2x128xi64>
    %1 = "mhlo.constant"() {value = dense<-1> : tensor<2x128xi64>} : () -> tensor<2x128xi64>
    %2 = "mhlo.constant"() {value = dense<-1> : tensor<1x128xi64>} : () -> tensor<1x128xi64>
    %3 = "mhlo.constant"() {value = dense<1.280000e+02> : tensor<2x128x1xf32>} : () -> tensor<2x128x1xf32>
    %4 = "mhlo.constant"() {value = dense<1.280000e+02> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %5 = "mhlo.constant"() {value = dense<8.000000e+00> : tensor<2x2x128x128xf32>} : () -> tensor<2x2x128x128xf32>
    %6 = "mhlo.constant"() {value = dense<-0.0142647391> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %7 = "mhlo.constant"() {value = dense<-0.00737332925> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %8 = "mhlo.constant"() {value = dense<-0.00168282702> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %9 = "mhlo.constant"() {value = dense<-2.13374049E-4> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %10 = "mhlo.constant"() {value = dense<-1.45660715E-5> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %11 = "mhlo.constant"() {value = dense<-0.0160960332> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %12 = "mhlo.constant"() {value = dense<-2.954600e-03> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %13 = "mhlo.constant"() {value = dense<-7.34990637E-4> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %14 = "mhlo.constant"() {value = dense<-5.69250624E-5> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %15 = "mhlo.constant"() {value = dense<-2.10102394E-6> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %16 = "mhlo.constant"() {value = dense<2.77068146E-8> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %17 = "mhlo.constant"() {value = dense<-2.72614237E-10> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %18 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %19 = "mhlo.constant"() {value = dense<4.000000e+00> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %20 = "mhlo.constant"() {value = dense<-4.000000e+00> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %21 = "mhlo.constant"() {value = dense<-0.0142647391> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %22 = "mhlo.constant"() {value = dense<-0.00737332925> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %23 = "mhlo.constant"() {value = dense<-0.00168282702> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %24 = "mhlo.constant"() {value = dense<-2.13374049E-4> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %25 = "mhlo.constant"() {value = dense<-1.45660715E-5> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %26 = "mhlo.constant"() {value = dense<-0.0160960332> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %27 = "mhlo.constant"() {value = dense<-2.954600e-03> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %28 = "mhlo.constant"() {value = dense<-7.34990637E-4> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %29 = "mhlo.constant"() {value = dense<-5.69250624E-5> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %30 = "mhlo.constant"() {value = dense<-2.10102394E-6> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %31 = "mhlo.constant"() {value = dense<2.77068146E-8> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %32 = "mhlo.constant"() {value = dense<-2.72614237E-10> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %33 = "mhlo.constant"() {value = dense<4.000000e+00> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %34 = "mhlo.constant"() {value = dense<-4.000000e+00> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %35 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %36 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %37 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<2x128xf32>} : () -> tensor<2x128xf32>
    %38 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<512x128xf32>} : () -> tensor<512x128xf32>
    %39 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<1x128x128xf32>} : () -> tensor<1x128x128xf32>
    %40 = "mhlo.constant"() {value = dense<0.398942292> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %41 = "mhlo.constant"() {value = dense<-5.000000e-01> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %42 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %43 = "mhlo.constant"() {value = dense<5.000000e-01> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %44 = "mhlo.constant"() {value = dense<0.707106769> : tensor<2x128x512xf32>} : () -> tensor<2x128x512xf32>
    %45 = "mhlo.constant"() {value = dense<0.398942292> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %46 = "mhlo.constant"() {value = dense<-5.000000e-01> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %47 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %48 = "mhlo.constant"() {value = dense<5.000000e-01> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %49 = "mhlo.constant"() {value = dense<0.707106769> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
    %50 = "mhlo.constant"() {value = dense<0> : tensor<30522x128xi32>} : () -> tensor<30522x128xi32>
    %51 = "mhlo.reshape"(%arg68) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %52 = "mhlo.transpose"(%arg46) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %53 = "mhlo.dot"(%51, %52) : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    %54 = "mhlo.transpose"(%51) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x30522xf32>) -> tensor<30522x256xf32>
    %55 = "mhlo.dot"(%54, %arg37) : (tensor<30522x256xf32>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %56 = "mhlo.reduce"(%51, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    %57 = "mhlo.reshape"(%53) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %58 = "mhlo.broadcast_in_dim"(%arg10) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %59 = "mhlo.subtract"(%arg15, %58) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %60 = "mhlo.broadcast_in_dim"(%arg16) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %61 = "mhlo.multiply"(%59, %60) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %62 = "mhlo.broadcast_in_dim"(%arg32) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<2x128x128xf32>
    %63 = "mhlo.multiply"(%57, %62) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %64 = "mhlo.multiply"(%63, %4) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %65 = "mhlo.reduce"(%63, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %66 = "mhlo.reshape"(%65) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %67 = "mhlo.multiply"(%63, %61) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %68 = "mhlo.reduce"(%67, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %69 = "mhlo.reshape"(%68) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %70 = "mhlo.broadcast_in_dim"(%69) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %71 = "mhlo.multiply"(%61, %70) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %72 = "mhlo.broadcast_in_dim"(%66) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %73 = "mhlo.subtract"(%64, %72) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %74 = "mhlo.subtract"(%73, %71) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %75 = "mhlo.divide"(%arg16, %3) : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %76 = "mhlo.broadcast_in_dim"(%75) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %77 = "mhlo.multiply"(%76, %74) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %78 = "mhlo.multiply"(%57, %61) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %79 = "mhlo.reduce"(%78, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %80 = "mhlo.reduce"(%57, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %81 = "mhlo.multiply"(%arg14, %49) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %82 = "mhlo.clamp"(%34, %81, %33) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %83 = "mhlo.multiply"(%82, %82) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %84 = "mhlo.multiply"(%83, %35) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %85 = "mhlo.add"(%84, %32) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %86 = "mhlo.multiply"(%85, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %87 = "mhlo.add"(%86, %31) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %88 = "mhlo.multiply"(%87, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %89 = "mhlo.add"(%88, %30) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %90 = "mhlo.multiply"(%89, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %91 = "mhlo.add"(%90, %29) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %92 = "mhlo.multiply"(%91, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %93 = "mhlo.add"(%92, %28) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %94 = "mhlo.multiply"(%93, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %95 = "mhlo.add"(%94, %27) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %96 = "mhlo.multiply"(%95, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %97 = "mhlo.add"(%96, %26) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %98 = "mhlo.add"(%84, %25) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %99 = "mhlo.multiply"(%98, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %100 = "mhlo.add"(%99, %24) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %101 = "mhlo.multiply"(%100, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %102 = "mhlo.add"(%101, %23) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %103 = "mhlo.multiply"(%102, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %104 = "mhlo.add"(%103, %22) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %105 = "mhlo.multiply"(%104, %83) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %106 = "mhlo.add"(%105, %21) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %107 = "mhlo.multiply"(%82, %97) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %108 = "mhlo.divide"(%107, %106) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %109 = "mhlo.add"(%108, %47) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %110 = "mhlo.multiply"(%109, %48) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %111 = "mhlo.multiply"(%arg14, %arg14) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %112 = "mhlo.multiply"(%111, %46) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %113 = "mhlo.exponential"(%112) : (tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %114 = "mhlo.multiply"(%113, %45) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %115 = "mhlo.multiply"(%114, %arg14) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %116 = "mhlo.add"(%115, %110) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %117 = "mhlo.multiply"(%77, %116) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %118 = "mhlo.reshape"(%117) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %119 = "mhlo.transpose"(%arg7) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %120 = "mhlo.dot"(%118, %119) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %121 = "mhlo.transpose"(%118) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %122 = "mhlo.dot"(%121, %arg13) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %123 = "mhlo.reduce"(%118, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %124 = "mhlo.reshape"(%120) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %125 = "mhlo.broadcast_in_dim"(%arg50) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %126 = "mhlo.subtract"(%arg1, %125) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %127 = "mhlo.broadcast_in_dim"(%arg56) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %128 = "mhlo.multiply"(%126, %127) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %129 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<2x128x128xf32>
    %130 = "mhlo.multiply"(%124, %129) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %131 = "mhlo.multiply"(%130, %4) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %132 = "mhlo.reduce"(%130, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %133 = "mhlo.reshape"(%132) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %134 = "mhlo.multiply"(%130, %128) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %135 = "mhlo.reduce"(%134, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %136 = "mhlo.reshape"(%135) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %137 = "mhlo.broadcast_in_dim"(%136) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %138 = "mhlo.multiply"(%128, %137) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %139 = "mhlo.broadcast_in_dim"(%133) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %140 = "mhlo.subtract"(%131, %139) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %141 = "mhlo.subtract"(%140, %138) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %142 = "mhlo.divide"(%arg56, %3) : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %143 = "mhlo.broadcast_in_dim"(%142) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %144 = "mhlo.multiply"(%143, %141) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %145 = "mhlo.multiply"(%124, %128) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %146 = "mhlo.reduce"(%145, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %147 = "mhlo.reduce"(%124, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %148 = "mhlo.reshape"(%144) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %149 = "mhlo.transpose"(%arg48) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %150 = "mhlo.dot"(%148, %149) : (tensor<256x128xf32>, tensor<128x512xf32>) -> tensor<256x512xf32>
    %151 = "mhlo.transpose"(%148) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %152 = "mhlo.dot"(%151, %arg9) : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
    %153 = "mhlo.reduce"(%148, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %154 = "mhlo.reshape"(%150) : (tensor<256x512xf32>) -> tensor<2x128x512xf32>
    %155 = "mhlo.multiply"(%arg23, %44) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %156 = "mhlo.clamp"(%20, %155, %19) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %157 = "mhlo.multiply"(%156, %156) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %158 = "mhlo.multiply"(%157, %18) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %159 = "mhlo.add"(%158, %17) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %160 = "mhlo.multiply"(%159, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %161 = "mhlo.add"(%160, %16) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %162 = "mhlo.multiply"(%161, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %163 = "mhlo.add"(%162, %15) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %164 = "mhlo.multiply"(%163, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %165 = "mhlo.add"(%164, %14) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %166 = "mhlo.multiply"(%165, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %167 = "mhlo.add"(%166, %13) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %168 = "mhlo.multiply"(%167, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %169 = "mhlo.add"(%168, %12) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %170 = "mhlo.multiply"(%169, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %171 = "mhlo.add"(%170, %11) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %172 = "mhlo.add"(%158, %10) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %173 = "mhlo.multiply"(%172, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %174 = "mhlo.add"(%173, %9) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %175 = "mhlo.multiply"(%174, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %176 = "mhlo.add"(%175, %8) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %177 = "mhlo.multiply"(%176, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %178 = "mhlo.add"(%177, %7) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %179 = "mhlo.multiply"(%178, %157) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %180 = "mhlo.add"(%179, %6) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %181 = "mhlo.multiply"(%156, %171) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %182 = "mhlo.divide"(%181, %180) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %183 = "mhlo.add"(%182, %42) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %184 = "mhlo.multiply"(%183, %43) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %185 = "mhlo.multiply"(%arg23, %arg23) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %186 = "mhlo.multiply"(%185, %41) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %187 = "mhlo.exponential"(%186) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %188 = "mhlo.multiply"(%187, %40) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %189 = "mhlo.multiply"(%188, %arg23) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %190 = "mhlo.add"(%189, %184) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %191 = "mhlo.multiply"(%154, %190) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %192 = "mhlo.reshape"(%191) : (tensor<2x128x512xf32>) -> tensor<256x512xf32>
    %193 = "mhlo.transpose"(%arg28) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %194 = "mhlo.dot"(%192, %193) : (tensor<256x512xf32>, tensor<512x128xf32>) -> tensor<256x128xf32>
    %195 = "mhlo.transpose"(%192) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x512xf32>) -> tensor<512x256xf32>
    %196 = "mhlo.dot"(%195, %arg6) : (tensor<512x256xf32>, tensor<256x128xf32>) -> tensor<512x128xf32>
    %197 = "mhlo.reduce"(%192, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x512xf32>, tensor<f32>) -> tensor<512xf32>
    %198 = "mhlo.reshape"(%194) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %199 = "mhlo.add"(%144, %198) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %200 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %201 = "mhlo.subtract"(%arg61, %200) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %202 = "mhlo.broadcast_in_dim"(%arg55) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %203 = "mhlo.multiply"(%201, %202) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %204 = "mhlo.broadcast_in_dim"(%arg30) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<2x128x128xf32>
    %205 = "mhlo.multiply"(%199, %204) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %206 = "mhlo.multiply"(%205, %4) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %207 = "mhlo.reduce"(%205, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %208 = "mhlo.reshape"(%207) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %209 = "mhlo.multiply"(%205, %203) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %210 = "mhlo.reduce"(%209, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %211 = "mhlo.reshape"(%210) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %212 = "mhlo.broadcast_in_dim"(%211) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %213 = "mhlo.multiply"(%203, %212) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %214 = "mhlo.broadcast_in_dim"(%208) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %215 = "mhlo.subtract"(%206, %214) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %216 = "mhlo.subtract"(%215, %213) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %217 = "mhlo.divide"(%arg55, %3) : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %218 = "mhlo.broadcast_in_dim"(%217) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %219 = "mhlo.multiply"(%218, %216) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %220 = "mhlo.multiply"(%199, %203) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %221 = "mhlo.reduce"(%220, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %222 = "mhlo.reduce"(%199, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %223 = "mhlo.reshape"(%219) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %224 = "mhlo.transpose"(%arg59) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %225 = "mhlo.dot"(%223, %224) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %226 = "mhlo.transpose"(%223) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %227 = "mhlo.dot"(%226, %arg2) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %228 = "mhlo.reduce"(%223, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %229 = "mhlo.reshape"(%225) : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
    %230 = "mhlo.transpose"(%229) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %231 = "mhlo.reshape"(%230) : (tensor<2x2x128x64xf32>) -> tensor<4x128x64xf32>
    %232 = "mhlo.transpose"(%arg52) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %233 = "mhlo.dot_general"(%232, %231) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %234 = "mhlo.transpose"(%arg8) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x128x64xf32>) -> tensor<4x64x128xf32>
    %235 = "mhlo.dot_general"(%231, %234) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x64xf32>, tensor<4x64x128xf32>) -> tensor<4x128x128xf32>
    %236 = "mhlo.reshape"(%233) : (tensor<4x128x64xf32>) -> tensor<2x2x128x64xf32>
    %237 = "mhlo.reshape"(%235) : (tensor<4x128x128xf32>) -> tensor<2x2x128x128xf32>
    %238 = "mhlo.multiply"(%237, %arg41) : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %239 = "mhlo.reduce"(%238, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<2x2x128x128xf32>, tensor<f32>) -> tensor<2x2x128xf32>
    %240 = "mhlo.reshape"(%239) : (tensor<2x2x128xf32>) -> tensor<2x2x128x1xf32>
    %241 = "mhlo.broadcast_in_dim"(%240) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x2x128x1xf32>) -> tensor<2x2x128x128xf32>
    %242 = "mhlo.multiply"(%arg41, %241) : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %243 = "mhlo.subtract"(%238, %242) : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %244 = "mhlo.divide"(%243, %5) : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %245 = "mhlo.reshape"(%244) : (tensor<2x2x128x128xf32>) -> tensor<4x128x128xf32>
    %246 = "mhlo.transpose"(%arg51) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x128x64xf32>) -> tensor<4x64x128xf32>
    %247 = "mhlo.dot_general"(%246, %245) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x64x128xf32>, tensor<4x128x128xf32>) -> tensor<4x64x128xf32>
    %248 = "mhlo.transpose"(%arg40) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x64x128xf32>) -> tensor<4x128x64xf32>
    %249 = "mhlo.dot_general"(%245, %248) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %250 = "mhlo.reshape"(%247) : (tensor<4x64x128xf32>) -> tensor<2x2x64x128xf32>
    %251 = "mhlo.reshape"(%249) : (tensor<4x128x64xf32>) -> tensor<2x2x128x64xf32>
    %252 = "mhlo.transpose"(%251) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %253 = "mhlo.transpose"(%236) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %254 = "mhlo.reshape"(%253) : (tensor<2x128x2x64xf32>) -> tensor<256x128xf32>
    %255 = "mhlo.transpose"(%arg36) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %256 = "mhlo.dot"(%254, %255) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %257 = "mhlo.transpose"(%254) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %258 = "mhlo.dot"(%257, %arg64) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %259 = "mhlo.reduce"(%254, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %260 = "mhlo.reshape"(%256) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %261 = "mhlo.add"(%219, %260) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %262 = "mhlo.transpose"(%250) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<2x2x64x128xf32>) -> tensor<2x128x2x64xf32>
    %263 = "mhlo.reshape"(%262) : (tensor<2x128x2x64xf32>) -> tensor<256x128xf32>
    %264 = "mhlo.transpose"(%arg45) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %265 = "mhlo.dot"(%263, %264) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %266 = "mhlo.transpose"(%263) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %267 = "mhlo.dot"(%266, %arg4) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %268 = "mhlo.reduce"(%263, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %269 = "mhlo.reshape"(%265) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %270 = "mhlo.add"(%261, %269) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %271 = "mhlo.reshape"(%252) : (tensor<2x128x2x64xf32>) -> tensor<256x128xf32>
    %272 = "mhlo.transpose"(%arg53) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %273 = "mhlo.dot"(%271, %272) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %274 = "mhlo.transpose"(%271) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %275 = "mhlo.dot"(%274, %arg27) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %276 = "mhlo.reduce"(%271, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %277 = "mhlo.reshape"(%273) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %278 = "mhlo.add"(%270, %277) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %279 = "mhlo.broadcast_in_dim"(%arg44) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %280 = "mhlo.subtract"(%arg34, %279) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %281 = "mhlo.broadcast_in_dim"(%arg47) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %282 = "mhlo.multiply"(%280, %281) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %283 = "mhlo.broadcast_in_dim"(%arg33) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<2x128x128xf32>
    %284 = "mhlo.multiply"(%278, %283) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %285 = "mhlo.multiply"(%284, %4) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %286 = "mhlo.reduce"(%284, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %287 = "mhlo.reshape"(%286) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %288 = "mhlo.multiply"(%284, %282) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %289 = "mhlo.reduce"(%288, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %290 = "mhlo.reshape"(%289) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %291 = "mhlo.broadcast_in_dim"(%290) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %292 = "mhlo.multiply"(%282, %291) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %293 = "mhlo.broadcast_in_dim"(%287) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %294 = "mhlo.subtract"(%285, %293) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %295 = "mhlo.subtract"(%294, %292) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %296 = "mhlo.divide"(%arg47, %3) : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %297 = "mhlo.broadcast_in_dim"(%296) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %298 = "mhlo.multiply"(%297, %295) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %299 = "mhlo.multiply"(%278, %282) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %300 = "mhlo.reduce"(%299, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %301 = "mhlo.reduce"(%278, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %302 = "mhlo.reshape"(%298) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %303 = "mhlo.transpose"(%arg26) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %304 = "mhlo.dot"(%302, %303) : (tensor<256x128xf32>, tensor<128x512xf32>) -> tensor<256x512xf32>
    %305 = "mhlo.transpose"(%302) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %306 = "mhlo.dot"(%305, %arg22) : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
    %307 = "mhlo.reduce"(%302, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %308 = "mhlo.reshape"(%304) : (tensor<256x512xf32>) -> tensor<2x128x512xf32>
    %309 = "mhlo.multiply"(%arg60, %44) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %310 = "mhlo.clamp"(%20, %309, %19) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %311 = "mhlo.multiply"(%310, %310) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %312 = "mhlo.multiply"(%311, %18) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %313 = "mhlo.add"(%312, %17) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %314 = "mhlo.multiply"(%313, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %315 = "mhlo.add"(%314, %16) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %316 = "mhlo.multiply"(%315, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %317 = "mhlo.add"(%316, %15) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %318 = "mhlo.multiply"(%317, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %319 = "mhlo.add"(%318, %14) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %320 = "mhlo.multiply"(%319, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %321 = "mhlo.add"(%320, %13) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %322 = "mhlo.multiply"(%321, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %323 = "mhlo.add"(%322, %12) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %324 = "mhlo.multiply"(%323, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %325 = "mhlo.add"(%324, %11) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %326 = "mhlo.add"(%312, %10) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %327 = "mhlo.multiply"(%326, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %328 = "mhlo.add"(%327, %9) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %329 = "mhlo.multiply"(%328, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %330 = "mhlo.add"(%329, %8) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %331 = "mhlo.multiply"(%330, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %332 = "mhlo.add"(%331, %7) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %333 = "mhlo.multiply"(%332, %311) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %334 = "mhlo.add"(%333, %6) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %335 = "mhlo.multiply"(%310, %325) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %336 = "mhlo.divide"(%335, %334) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %337 = "mhlo.add"(%336, %42) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %338 = "mhlo.multiply"(%337, %43) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %339 = "mhlo.multiply"(%arg60, %arg60) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %340 = "mhlo.multiply"(%339, %41) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %341 = "mhlo.exponential"(%340) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %342 = "mhlo.multiply"(%341, %40) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %343 = "mhlo.multiply"(%342, %arg60) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %344 = "mhlo.add"(%343, %338) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %345 = "mhlo.multiply"(%308, %344) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %346 = "mhlo.reshape"(%345) : (tensor<2x128x512xf32>) -> tensor<256x512xf32>
    %347 = "mhlo.transpose"(%arg20) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %348 = "mhlo.dot"(%346, %347) : (tensor<256x512xf32>, tensor<512x128xf32>) -> tensor<256x128xf32>
    %349 = "mhlo.transpose"(%346) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x512xf32>) -> tensor<512x256xf32>
    %350 = "mhlo.dot"(%349, %arg31) : (tensor<512x256xf32>, tensor<256x128xf32>) -> tensor<512x128xf32>
    %351 = "mhlo.reduce"(%346, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x512xf32>, tensor<f32>) -> tensor<512xf32>
    %352 = "mhlo.reshape"(%348) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %353 = "mhlo.add"(%298, %352) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %354 = "mhlo.broadcast_in_dim"(%arg66) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %355 = "mhlo.subtract"(%arg24, %354) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %356 = "mhlo.broadcast_in_dim"(%arg38) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %357 = "mhlo.multiply"(%355, %356) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %358 = "mhlo.broadcast_in_dim"(%arg63) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<2x128x128xf32>
    %359 = "mhlo.multiply"(%353, %358) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %360 = "mhlo.multiply"(%359, %4) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %361 = "mhlo.reduce"(%359, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %362 = "mhlo.reshape"(%361) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %363 = "mhlo.multiply"(%359, %357) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %364 = "mhlo.reduce"(%363, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %365 = "mhlo.reshape"(%364) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %366 = "mhlo.broadcast_in_dim"(%365) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %367 = "mhlo.multiply"(%357, %366) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %368 = "mhlo.broadcast_in_dim"(%362) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %369 = "mhlo.subtract"(%360, %368) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %370 = "mhlo.subtract"(%369, %367) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %371 = "mhlo.divide"(%arg38, %3) : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %372 = "mhlo.broadcast_in_dim"(%371) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %373 = "mhlo.multiply"(%372, %370) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %374 = "mhlo.multiply"(%353, %357) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %375 = "mhlo.reduce"(%374, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %376 = "mhlo.reduce"(%353, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %377 = "mhlo.reshape"(%373) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %378 = "mhlo.transpose"(%arg57) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %379 = "mhlo.dot"(%377, %378) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %380 = "mhlo.transpose"(%377) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %381 = "mhlo.dot"(%380, %arg12) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %382 = "mhlo.reduce"(%377, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %383 = "mhlo.reshape"(%379) : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
    %384 = "mhlo.transpose"(%383) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %385 = "mhlo.reshape"(%384) : (tensor<2x2x128x64xf32>) -> tensor<4x128x64xf32>
    %386 = "mhlo.transpose"(%arg29) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %387 = "mhlo.dot_general"(%386, %385) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %388 = "mhlo.transpose"(%arg25) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x128x64xf32>) -> tensor<4x64x128xf32>
    %389 = "mhlo.dot_general"(%385, %388) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x64xf32>, tensor<4x64x128xf32>) -> tensor<4x128x128xf32>
    %390 = "mhlo.reshape"(%387) : (tensor<4x128x64xf32>) -> tensor<2x2x128x64xf32>
    %391 = "mhlo.reshape"(%389) : (tensor<4x128x128xf32>) -> tensor<2x2x128x128xf32>
    %392 = "mhlo.multiply"(%391, %arg62) : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %393 = "mhlo.reduce"(%392, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<2x2x128x128xf32>, tensor<f32>) -> tensor<2x2x128xf32>
    %394 = "mhlo.reshape"(%393) : (tensor<2x2x128xf32>) -> tensor<2x2x128x1xf32>
    %395 = "mhlo.broadcast_in_dim"(%394) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x2x128x1xf32>) -> tensor<2x2x128x128xf32>
    %396 = "mhlo.multiply"(%arg62, %395) : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %397 = "mhlo.subtract"(%392, %396) : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %398 = "mhlo.divide"(%397, %5) : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %399 = "mhlo.reshape"(%398) : (tensor<2x2x128x128xf32>) -> tensor<4x128x128xf32>
    %400 = "mhlo.transpose"(%arg21) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x128x64xf32>) -> tensor<4x64x128xf32>
    %401 = "mhlo.dot_general"(%400, %399) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x64x128xf32>, tensor<4x128x128xf32>) -> tensor<4x64x128xf32>
    %402 = "mhlo.transpose"(%arg11) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x64x128xf32>) -> tensor<4x128x64xf32>
    %403 = "mhlo.dot_general"(%399, %402) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %404 = "mhlo.reshape"(%401) : (tensor<4x64x128xf32>) -> tensor<2x2x64x128xf32>
    %405 = "mhlo.reshape"(%403) : (tensor<4x128x64xf32>) -> tensor<2x2x128x64xf32>
    %406 = "mhlo.transpose"(%405) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %407 = "mhlo.transpose"(%390) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %408 = "mhlo.reshape"(%407) : (tensor<2x128x2x64xf32>) -> tensor<256x128xf32>
    %409 = "mhlo.transpose"(%arg49) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %410 = "mhlo.dot"(%408, %409) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %411 = "mhlo.transpose"(%408) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %412 = "mhlo.dot"(%411, %arg39) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %413 = "mhlo.reduce"(%408, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %414 = "mhlo.reshape"(%410) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %415 = "mhlo.add"(%373, %414) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %416 = "mhlo.transpose"(%404) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<2x2x64x128xf32>) -> tensor<2x128x2x64xf32>
    %417 = "mhlo.reshape"(%416) : (tensor<2x128x2x64xf32>) -> tensor<256x128xf32>
    %418 = "mhlo.transpose"(%arg54) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %419 = "mhlo.dot"(%417, %418) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %420 = "mhlo.transpose"(%417) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %421 = "mhlo.dot"(%420, %arg43) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %422 = "mhlo.reduce"(%417, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %423 = "mhlo.reshape"(%419) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %424 = "mhlo.add"(%415, %423) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %425 = "mhlo.reshape"(%406) : (tensor<2x128x2x64xf32>) -> tensor<256x128xf32>
    %426 = "mhlo.transpose"(%arg35) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %427 = "mhlo.dot"(%425, %426) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %428 = "mhlo.transpose"(%425) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %429 = "mhlo.dot"(%428, %arg67) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    %430 = "mhlo.reduce"(%425, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<256x128xf32>, tensor<f32>) -> tensor<128xf32>
    %431 = "mhlo.reshape"(%427) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %432 = "mhlo.add"(%424, %431) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %433 = "mhlo.broadcast_in_dim"(%arg19) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %434 = "mhlo.subtract"(%arg58, %433) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %435 = "mhlo.broadcast_in_dim"(%arg17) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %436 = "mhlo.multiply"(%434, %435) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %437 = "mhlo.broadcast_in_dim"(%arg65) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<2x128x128xf32>
    %438 = "mhlo.multiply"(%432, %437) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %439 = "mhlo.multiply"(%438, %4) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %440 = "mhlo.reduce"(%438, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %441 = "mhlo.reshape"(%440) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %442 = "mhlo.multiply"(%438, %436) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %443 = "mhlo.reduce"(%442, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %444 = "mhlo.reshape"(%443) : (tensor<2x128xf32>) -> tensor<2x128x1xf32>
    %445 = "mhlo.broadcast_in_dim"(%444) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %446 = "mhlo.multiply"(%436, %445) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %447 = "mhlo.broadcast_in_dim"(%441) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %448 = "mhlo.subtract"(%439, %447) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %449 = "mhlo.subtract"(%448, %446) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %450 = "mhlo.divide"(%arg17, %3) : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %451 = "mhlo.broadcast_in_dim"(%450) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %452 = "mhlo.multiply"(%451, %449) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %453 = "mhlo.multiply"(%432, %436) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %454 = "mhlo.reduce"(%453, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %455 = "mhlo.reduce"(%432, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %456 = "mhlo.reduce"(%452, %36) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    %457 = "mhlo.reshape"(%456) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %458 = "mhlo.convert"(%arg42) : (tensor<1x128xi64>) -> tensor<1x128xf32>
    %459 = "mhlo.convert"(%458) : (tensor<1x128xf32>) -> tensor<1x128xi64>
    %460 = "mhlo.compare"(%459, %2) {compare_type = #mhlo<comparison_type SIGNED>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<1x128xi64>, tensor<1x128xi64>) -> tensor<1x128xi1>
    %461 = "mhlo.reshape"(%460) : (tensor<1x128xi1>) -> tensor<1x128x1xi1>
    %462 = "mhlo.broadcast_in_dim"(%461) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x128x1xi1>) -> tensor<1x128x128xi1>
    %463 = "mhlo.select"(%462, %39, %457) : (tensor<1x128x128xi1>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %464 = "mhlo.reshape"(%459) : (tensor<1x128xi64>) -> tensor<128x1xi64>
    %465 = "mhlo.reshape"(%463) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %466 = "mhlo.scatter"(%38, %464, %465) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %467 = "mhlo.convert"(%arg18) : (tensor<2x128xi64>) -> tensor<2x128xf32>
    %468 = "mhlo.convert"(%467) : (tensor<2x128xf32>) -> tensor<2x128xi64>
    %469 = "mhlo.compare"(%468, %1) {compare_type = #mhlo<comparison_type SIGNED>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<2x128xi64>, tensor<2x128xi64>) -> tensor<2x128xi1>
    %470 = "mhlo.reshape"(%469) : (tensor<2x128xi1>) -> tensor<2x128x1xi1>
    %471 = "mhlo.broadcast_in_dim"(%470) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xi1>) -> tensor<2x128x128xi1>
    %472 = "mhlo.select"(%471, %35, %452) : (tensor<2x128x128xi1>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %473 = "mhlo.reshape"(%468) : (tensor<2x128xi64>) -> tensor<256x1xi64>
    %474 = "mhlo.reshape"(%472) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %475 = "mhlo.scatter"(%37, %473, %474) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %476 = "mhlo.convert"(%arg5) : (tensor<2x128xi64>) -> tensor<2x128xf32>
    %477 = "mhlo.convert"(%476) : (tensor<2x128xf32>) -> tensor<2x128xi64>
    %478 = "mhlo.compare"(%477, %0) {compare_type = #mhlo<comparison_type SIGNED>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<2x128xi64>, tensor<2x128xi64>) -> tensor<2x128xi1>
    %479 = "mhlo.reshape"(%478) : (tensor<2x128xi1>) -> tensor<2x128x1xi1>
    %480 = "mhlo.broadcast_in_dim"(%479) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x128x1xi1>) -> tensor<2x128x128xi1>
    %481 = "mhlo.select"(%480, %35, %452) : (tensor<2x128x128xi1>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %482 = "mhlo.convert"(%50) : (tensor<30522x128xi32>) -> tensor<30522x128xf32>
    %483 = "mhlo.reshape"(%477) : (tensor<2x128xi64>) -> tensor<256x1xi64>
    %484 = "mhlo.reshape"(%481) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %485 = "mhlo.scatter"(%482, %483, %484) ({
    ^bb0(%arg69: tensor<f32>, %arg70: tensor<f32>):
      %486 = "mhlo.add"(%arg69, %arg70) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%486) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    "func.return"(%485, %466, %475, %454, %455, %429, %430, %421, %422, %412, %413, %381, %382, %375, %376, %350, %351, %306, %307, %300, %301, %275, %276, %267, %268, %258, %259, %227, %228, %221, %222, %196, %197, %152, %153, %146, %147, %122, %123, %79, %80, %55, %56) : (tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) -> ()
  }) {function_type = (tensor<128xf32>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<256x128xf32>, tensor<2x128xi64>, tensor<256x128xf32>, tensor<128x128xf32>, tensor<4x128x64xf32>, tensor<256x512xf32>, tensor<2x128x1xf32>, tensor<4x64x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<2x128xi64>, tensor<2x128x1xf32>, tensor<128x512xf32>, tensor<4x128x64xf32>, tensor<256x512xf32>, tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<4x128x64xf32>, tensor<512x128xf32>, tensor<256x128xf32>, tensor<128x512xf32>, tensor<4x128x128xf32>, tensor<128xf32>, tensor<256x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<256x128xf32>, tensor<4x64x128xf32>, tensor<2x2x128x128xf32>, tensor<1x128xi64>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<128x128xf32>, tensor<128x30522xf32>, tensor<2x128x1xf32>, tensor<512x128xf32>, tensor<128x128xf32>, tensor<2x128x1xf32>, tensor<4x128x64xf32>, tensor<4x128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<128xf32>, tensor<256x128xf32>, tensor<128xf32>, tensor<2x128x1xf32>, tensor<256x128xf32>, tensor<2x128x30522xf32>) -> (tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>), sym_name = "main"} : () -> ()
}) {torch.debug_module_name = "GraphModule"} : () -> ()
