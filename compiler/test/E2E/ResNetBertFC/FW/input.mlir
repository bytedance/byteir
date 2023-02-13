// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
!tuple = tuple<tensor<2x19xf32>, tensor<2x19xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<19xi64>, tensor<19xi64>, tensor<19xi64>, tensor<19xi64>, tensor<0xf32>, tensor<16x512x8x8xf32>, tensor<24x240x240xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<480x768xf32>, tensor<16x2048x8x8xf32>, tensor<2x2048xf32>, tensor<2x2816xf32>, tensor<2816x128xf32>, tensor<768x3072xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<2x128xf32>, tensor<480x768xf32>, tensor<16x2048x8x8xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<24x240x240xf32>, tensor<16x1024x16x16xf32>, tensor<16x512x8x8xf32>, tensor<480x768xf32>, tensor<16x1024x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<16x256x16x16xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x16x16xf32>, tensor<3072x768xf32>, tensor<480x3072xf32>, tensor<2x240x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768xf32>, tensor<2048xf32>, tensor<768xf32>, tensor<2048xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<16x2048x8x8xf32>, tensor<3072x768xf32>, tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<768xf32>, tensor<480x3072xf32>, tensor<768xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2x240x768xf32>, tensor<16x512x8x8xf32>, tensor<768xf32>, tensor<768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<16x512x8x8xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<768x768xf32>, tensor<512xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<2x240x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>, tensor<480x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<2x240x768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<2x240x3072xf32>, tensor<2x240xi64>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<16x128x32x32xf32>, tensor<2x12x240x240xf32>, tensor<480x768xf32>, tensor<3072x768xf32>, tensor<24x64x240xf32>, tensor<480x768xf32>, tensor<2x240x768xf32>, tensor<480x3072xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<2x240x768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<2x240x768xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<2x12x240x240xf32>, tensor<24x240x240xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<768x768xf32>, tensor<24x240x64xf32>, tensor<2x240x768xf32>, tensor<24x64x240xf32>, tensor<24x240x64xf32>, tensor<24x240x64xf32>, tensor<2x12x240x240xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<16x1024x16x16xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<480x768xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x3x7x7xf32>, tensor<16x256x16x16xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<24x64x240xf32>, tensor<16x256x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x16x16xf32>, tensor<16x256x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2x240x768xf32>, tensor<16x512x32x32xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<16x1024x16x16xf32>, tensor<128xf32>, tensor<16x1024x16x16xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<24x240x240xf32>, tensor<16x128x32x32xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<16x256x16x16xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<480x768xf32>, tensor<16x128x32x32xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<256xf32>, tensor<16x512x32x32xf32>, tensor<2x240x1xf32>, tensor<512xf32>, tensor<16x256x16x16xf32>, tensor<2x240x1xf32>, tensor<512xf32>, tensor<16x512x32x32xf32>, tensor<480x768xf32>, tensor<2x240x3072xf32>, tensor<128xf32>, tensor<2x240x1xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<2x240x1xf32>, tensor<512x128x1x1xf32>, tensor<128xf32>, tensor<768x768xf32>, tensor<128x128x3x3xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<256xf32>, tensor<768x768xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<768x768xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<256x1024x1x1xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<2x12x240x240xf32>, tensor<1024x256x1x1xf32>, tensor<24x64x240xf32>, tensor<256xf32>, tensor<2x12x240x240xf32>, tensor<16x256x64x64xf32>, tensor<24x240x64xf32>, tensor<256xf32>, tensor<64xf32>, tensor<2x12x240x240xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>, tensor<24x64x240xf32>, tensor<256xf32>, tensor<24x240x64xf32>, tensor<16x256x64x64xf32>, tensor<2x12x240x240xf32>, tensor<2x240x768xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<768x3072xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<3072x768xf32>, tensor<480x3072xf32>, tensor<2x12x240x240xf32>, tensor<2x240x768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<2x240x768xf32>, tensor<768x768xf32>, tensor<24x64x240xf32>, tensor<2x240x768xf32>, tensor<24x240x240xf32>, tensor<24x240x64xf32>, tensor<768x768xf32>, tensor<256xf32>, tensor<512xf32>, tensor<768x768xf32>, tensor<128x19xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2816x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<256x19xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048x256xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<768x768xf32>, tensor<2x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x2xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x3072xf32>, tensor<64xf32>, tensor<64xf32>, tensor<2x240x1xf32>, tensor<16x64x64x64xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<16x64x64x64xf32>, tensor<768x768xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<16x256x64x64xf32>, tensor<768x768xf32>, tensor<2x240x768xf32>, tensor<16x256x64x64xf32>, tensor<480x768xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>, tensor<24x240x64xf32>, tensor<16x512x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<16x2048x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512x8x8xf32>, tensor<16x2048x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<24x240x240xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<768x768xf32>, tensor<16x512x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<768xf32>, tensor<16x2048x8x8xf32>, tensor<480x768xf32>, tensor<480x768xf32>, tensor<16x256x16x16xf32>, tensor<768xf32>, tensor<16x512x8x8xf32>, tensor<768xf32>, tensor<256xf32>, tensor<24x240x240xf32>, tensor<256xf32>, tensor<2048xf32>, tensor<16x2048x8x8xf32>, tensor<768x3072xf32>, tensor<16x256x16x16xf32>, tensor<2048xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<24x64x240xf32>, tensor<512xf32>, tensor<16x512x16x16xf32>, tensor<768xf32>, tensor<768xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x16x16xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<480x3072xf32>, tensor<2x12x240x240xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<2x240x768xf32>, tensor<512xf32>, tensor<64xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<2x240x768xf32>, tensor<512xf32>, tensor<64xf32>, tensor<16x512x8x8xf32>, tensor<512x1024x1x1xf32>, tensor<2x240x3072xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<24x240x240xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<3072x768xf32>, tensor<256xf32>, tensor<768xf32>, tensor<480x3072xf32>, tensor<256x1024x1x1xf32>, tensor<768xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<768xf32>, tensor<2x240x768xf32>, tensor<768xf32>, tensor<1024xf32>, tensor<2x240x1xf32>, tensor<1024x256x1x1xf32>, tensor<480x768xf32>, tensor<2x240x1xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<2x768xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<256x256x3x3xf32>, tensor<2x240x1xf32>, tensor<1024xf32>, tensor<768x768xf32>, tensor<2x240x1xf32>, tensor<1024x256x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x64x128x128xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<256x1024x1x1xf32>, tensor<768xf32>, tensor<2x768xf32>, tensor<256xf32>, tensor<768xf32>, tensor<256x256x3x3xf32>, tensor<2x240x3072xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<512xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<768x256xf32>, tensor<768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<2x256xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<2x240x768xf32>, tensor<24x64x240xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<256x2xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<24x240x240xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<24x64x240xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<768xf32>, tensor<2x240x768xf32>, tensor<24x240x64xf32>, tensor<768xf32>, tensor<24x64x240xf32>, tensor<24x240x64xf32>, tensor<24x240x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<768x3072xf32>, tensor<16x1024x16x16xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<480x3072xf32>, tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x32x32xf32>, tensor<16x256x16x16xf32>, tensor<3072x768xf32>, tensor<16x256x16x16xf32>, tensor<2x240x768xf32>, tensor<2x240x1xf32>, tensor<256xf32>, tensor<16x1024x16x16xf32>, tensor<256xf32>, tensor<2x240x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<768x768xf32>, tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<480x768xf32>, tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>, tensor<768x768xf32>, tensor<16x256x16x16xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<64xf32>, tensor<3072x768xf32>, tensor<256x64x1x1xf32>, tensor<64x64x3x3xf32>, tensor<480x3072xf32>, tensor<128xf32>, tensor<128x256x1x1xf32>, tensor<2x240x768xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<2x240x1xf32>, tensor<512xf32>, tensor<2x240x1xf32>, tensor<512x128x1x1xf32>, tensor<768x768xf32>, tensor<512xf32>, tensor<480x768xf32>, tensor<512x256x1x1xf32>, tensor<480x768xf32>, tensor<2x240x768xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<768x768xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<768x768xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<128x128x3x3xf32>, tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>, tensor<64x64x3x3xf32>, tensor<64x256x1x1xf32>, tensor<24x240x64xf32>, tensor<64xf32>, tensor<24x240x240xf32>, tensor<24x240x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<480x768xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<768x3072xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<128xf32>, tensor<128xf32>, tensor<480x768xf32>, tensor<16x128x64x64xf32>, tensor<768x768xf32>, tensor<16x128x32x32xf32>, tensor<480x768xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<16x128x32x32xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<24x64x240xf32>, tensor<2x12x240x240xf32>, tensor<512xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<512xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<16x3x256x256xf32>, tensor<2x240xi64>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>, tensor<16x128x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>, tensor<16x128x32x32xf32>, tensor<16x512x32x32xf32>, tensor<64xf32>, tensor<24x240x64xf32>, tensor<64xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<2x240x3072xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3072x768xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x3072xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<24x240x64xf32>, tensor<256xf32>, tensor<2x240x768xf32>, tensor<256xf32>, tensor<24x64x240xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<24x240x240xf32>, tensor<24x240x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x256x64x64xf32>, tensor<16x64x128x128xf32>, tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>, tensor<1x240xi64>, tensor<768x768xf32>, tensor<24x240x64xf32>, tensor<2x240x768xf32>, tensor<24x240x64xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<512xf32>, tensor<3072x768xf32>, tensor<24x240x240xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x3072xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x240x768xf32>, tensor<64xf32>, tensor<768xf32>, tensor<64xf32>, tensor<2x240x1xf32>, tensor<768xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<768x768xf32>, tensor<64xf32>, tensor<480x768xf32>, tensor<768xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<768xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<480x768xf32>, tensor<64xf32>, tensor<64xf32>, tensor<768x768xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<768xf32>, tensor<128xf32>, tensor<768x768xf32>, tensor<2x240x3072xf32>, tensor<768xf32>, tensor<128xf32>, tensor<2x240x768xf32>, tensor<128xf32>, tensor<768xf32>, tensor<128xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<16x256x16x16xf32>, tensor<512xf32>, tensor<480x768xf32>, tensor<3072x768xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<512xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<512xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>>
module @IrToMhlo.4326 {
  func.func @main(%arg0: tensor<19xf32>, %arg1: tensor<19x256xf32>, %arg2: tensor<19xf32>, %arg3: tensor<19x128xf32>, %arg4: tensor<2xf32>, %arg5: tensor<2x256xf32>, %arg6: tensor<2xf32>, %arg7: tensor<2x256xf32>, %arg8: tensor<256xf32>, %arg9: tensor<256x2816xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x2816xf32>, %arg12: tensor<256xf32>, %arg13: tensor<256x2048xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256x768xf32>, %arg16: tensor<1000xf32>, %arg17: tensor<1000x2048xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64x3x7x7xf32>, %arg21: tensor<64xf32>, %arg22: tensor<64xf32>, %arg23: tensor<64x64x1x1xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64x64x3x3xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<256x64x1x1xf32>, %arg30: tensor<256xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256x64x1x1xf32>, %arg33: tensor<64xf32>, %arg34: tensor<64xf32>, %arg35: tensor<64x256x1x1xf32>, %arg36: tensor<64xf32>, %arg37: tensor<64xf32>, %arg38: tensor<64x64x3x3xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256x64x1x1xf32>, %arg42: tensor<64xf32>, %arg43: tensor<64xf32>, %arg44: tensor<64x256x1x1xf32>, %arg45: tensor<64xf32>, %arg46: tensor<64xf32>, %arg47: tensor<64x64x3x3xf32>, %arg48: tensor<256xf32>, %arg49: tensor<256xf32>, %arg50: tensor<256x64x1x1xf32>, %arg51: tensor<128xf32>, %arg52: tensor<128xf32>, %arg53: tensor<128x256x1x1xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128x128x3x3xf32>, %arg57: tensor<512xf32>, %arg58: tensor<512xf32>, %arg59: tensor<512x128x1x1xf32>, %arg60: tensor<512xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512x256x1x1xf32>, %arg63: tensor<128xf32>, %arg64: tensor<128xf32>, %arg65: tensor<128x512x1x1xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128x128x3x3xf32>, %arg69: tensor<512xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512x128x1x1xf32>, %arg72: tensor<128xf32>, %arg73: tensor<128xf32>, %arg74: tensor<128x512x1x1xf32>, %arg75: tensor<128xf32>, %arg76: tensor<128xf32>, %arg77: tensor<128x128x3x3xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512x128x1x1xf32>, %arg81: tensor<128xf32>, %arg82: tensor<128xf32>, %arg83: tensor<128x512x1x1xf32>, %arg84: tensor<128xf32>, %arg85: tensor<128xf32>, %arg86: tensor<128x128x3x3xf32>, %arg87: tensor<512xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512x128x1x1xf32>, %arg90: tensor<256xf32>, %arg91: tensor<256xf32>, %arg92: tensor<256x512x1x1xf32>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256x256x3x3xf32>, %arg96: tensor<1024xf32>, %arg97: tensor<1024xf32>, %arg98: tensor<1024x256x1x1xf32>, %arg99: tensor<1024xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<1024x512x1x1xf32>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256x1024x1x1xf32>, %arg105: tensor<256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<256x256x3x3xf32>, %arg108: tensor<1024xf32>, %arg109: tensor<1024xf32>, %arg110: tensor<1024x256x1x1xf32>, %arg111: tensor<256xf32>, %arg112: tensor<256xf32>, %arg113: tensor<256x1024x1x1xf32>, %arg114: tensor<256xf32>, %arg115: tensor<256xf32>, %arg116: tensor<256x256x3x3xf32>, %arg117: tensor<1024xf32>, %arg118: tensor<1024xf32>, %arg119: tensor<1024x256x1x1xf32>, %arg120: tensor<256xf32>, %arg121: tensor<256xf32>, %arg122: tensor<256x1024x1x1xf32>, %arg123: tensor<256xf32>, %arg124: tensor<256xf32>, %arg125: tensor<256x256x3x3xf32>, %arg126: tensor<1024xf32>, %arg127: tensor<1024xf32>, %arg128: tensor<1024x256x1x1xf32>, %arg129: tensor<256xf32>, %arg130: tensor<256xf32>, %arg131: tensor<256x1024x1x1xf32>, %arg132: tensor<256xf32>, %arg133: tensor<256xf32>, %arg134: tensor<256x256x3x3xf32>, %arg135: tensor<1024xf32>, %arg136: tensor<1024xf32>, %arg137: tensor<1024x256x1x1xf32>, %arg138: tensor<256xf32>, %arg139: tensor<256xf32>, %arg140: tensor<256x1024x1x1xf32>, %arg141: tensor<256xf32>, %arg142: tensor<256xf32>, %arg143: tensor<256x256x3x3xf32>, %arg144: tensor<1024xf32>, %arg145: tensor<1024xf32>, %arg146: tensor<1024x256x1x1xf32>, %arg147: tensor<512xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512x1024x1x1xf32>, %arg150: tensor<512xf32>, %arg151: tensor<512xf32>, %arg152: tensor<512x512x3x3xf32>, %arg153: tensor<2048xf32>, %arg154: tensor<2048xf32>, %arg155: tensor<2048x512x1x1xf32>, %arg156: tensor<2048xf32>, %arg157: tensor<2048xf32>, %arg158: tensor<2048x1024x1x1xf32>, %arg159: tensor<512xf32>, %arg160: tensor<512xf32>, %arg161: tensor<512x2048x1x1xf32>, %arg162: tensor<512xf32>, %arg163: tensor<512xf32>, %arg164: tensor<512x512x3x3xf32>, %arg165: tensor<2048xf32>, %arg166: tensor<2048xf32>, %arg167: tensor<2048x512x1x1xf32>, %arg168: tensor<512xf32>, %arg169: tensor<512xf32>, %arg170: tensor<512x2048x1x1xf32>, %arg171: tensor<512xf32>, %arg172: tensor<512xf32>, %arg173: tensor<512x512x3x3xf32>, %arg174: tensor<2048xf32>, %arg175: tensor<2048xf32>, %arg176: tensor<2048x512x1x1xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<512x768xf32>, %arg180: tensor<2x768xf32>, %arg181: tensor<21128x768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768x768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<768x768xf32>, %arg192: tensor<3072xf32>, %arg193: tensor<3072x768xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768x3072xf32>, %arg198: tensor<768xf32>, %arg199: tensor<768xf32>, %arg200: tensor<768xf32>, %arg201: tensor<768x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<768x768xf32>, %arg208: tensor<3072xf32>, %arg209: tensor<3072x768xf32>, %arg210: tensor<768xf32>, %arg211: tensor<768xf32>, %arg212: tensor<768xf32>, %arg213: tensor<768x3072xf32>, %arg214: tensor<768xf32>, %arg215: tensor<768xf32>, %arg216: tensor<768xf32>, %arg217: tensor<768x768xf32>, %arg218: tensor<768xf32>, %arg219: tensor<768x768xf32>, %arg220: tensor<768xf32>, %arg221: tensor<768x768xf32>, %arg222: tensor<768xf32>, %arg223: tensor<768x768xf32>, %arg224: tensor<3072xf32>, %arg225: tensor<3072x768xf32>, %arg226: tensor<768xf32>, %arg227: tensor<768xf32>, %arg228: tensor<768xf32>, %arg229: tensor<768x3072xf32>, %arg230: tensor<768xf32>, %arg231: tensor<768xf32>, %arg232: tensor<768xf32>, %arg233: tensor<768x768xf32>, %arg234: tensor<768xf32>, %arg235: tensor<768x768xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768x768xf32>, %arg238: tensor<768xf32>, %arg239: tensor<768x768xf32>, %arg240: tensor<3072xf32>, %arg241: tensor<3072x768xf32>, %arg242: tensor<768xf32>, %arg243: tensor<768xf32>, %arg244: tensor<768xf32>, %arg245: tensor<768x3072xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<768xf32>, %arg249: tensor<768x768xf32>, %arg250: tensor<768xf32>, %arg251: tensor<768x768xf32>, %arg252: tensor<768xf32>, %arg253: tensor<768x768xf32>, %arg254: tensor<768xf32>, %arg255: tensor<768x768xf32>, %arg256: tensor<3072xf32>, %arg257: tensor<3072x768xf32>, %arg258: tensor<768xf32>, %arg259: tensor<768xf32>, %arg260: tensor<768xf32>, %arg261: tensor<768x3072xf32>, %arg262: tensor<768xf32>, %arg263: tensor<768xf32>, %arg264: tensor<768xf32>, %arg265: tensor<768x768xf32>, %arg266: tensor<768xf32>, %arg267: tensor<768x768xf32>, %arg268: tensor<768xf32>, %arg269: tensor<768x768xf32>, %arg270: tensor<768xf32>, %arg271: tensor<768x768xf32>, %arg272: tensor<3072xf32>, %arg273: tensor<3072x768xf32>, %arg274: tensor<768xf32>, %arg275: tensor<768xf32>, %arg276: tensor<768xf32>, %arg277: tensor<768x3072xf32>, %arg278: tensor<768xf32>, %arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768x768xf32>, %arg282: tensor<768xf32>, %arg283: tensor<768x768xf32>, %arg284: tensor<768xf32>, %arg285: tensor<768x768xf32>, %arg286: tensor<768xf32>, %arg287: tensor<768x768xf32>, %arg288: tensor<3072xf32>, %arg289: tensor<3072x768xf32>, %arg290: tensor<768xf32>, %arg291: tensor<768xf32>, %arg292: tensor<768xf32>, %arg293: tensor<768x3072xf32>, %arg294: tensor<768xf32>, %arg295: tensor<768xf32>, %arg296: tensor<768xf32>, %arg297: tensor<768x768xf32>, %arg298: tensor<768xf32>, %arg299: tensor<768x768xf32>, %arg300: tensor<768xf32>, %arg301: tensor<768x768xf32>, %arg302: tensor<768xf32>, %arg303: tensor<768x768xf32>, %arg304: tensor<3072xf32>, %arg305: tensor<3072x768xf32>, %arg306: tensor<768xf32>, %arg307: tensor<768xf32>, %arg308: tensor<768xf32>, %arg309: tensor<768x3072xf32>, %arg310: tensor<768xf32>, %arg311: tensor<768xf32>, %arg312: tensor<768xf32>, %arg313: tensor<768x768xf32>, %arg314: tensor<768xf32>, %arg315: tensor<768x768xf32>, %arg316: tensor<768xf32>, %arg317: tensor<768x768xf32>, %arg318: tensor<768xf32>, %arg319: tensor<768x768xf32>, %arg320: tensor<3072xf32>, %arg321: tensor<3072x768xf32>, %arg322: tensor<768xf32>, %arg323: tensor<768xf32>, %arg324: tensor<768xf32>, %arg325: tensor<768x3072xf32>, %arg326: tensor<768xf32>, %arg327: tensor<768xf32>, %arg328: tensor<768xf32>, %arg329: tensor<768x768xf32>, %arg330: tensor<768xf32>, %arg331: tensor<768x768xf32>, %arg332: tensor<768xf32>, %arg333: tensor<768x768xf32>, %arg334: tensor<768xf32>, %arg335: tensor<768x768xf32>, %arg336: tensor<3072xf32>, %arg337: tensor<3072x768xf32>, %arg338: tensor<768xf32>, %arg339: tensor<768xf32>, %arg340: tensor<768xf32>, %arg341: tensor<768x3072xf32>, %arg342: tensor<768xf32>, %arg343: tensor<768xf32>, %arg344: tensor<768xf32>, %arg345: tensor<768x768xf32>, %arg346: tensor<768xf32>, %arg347: tensor<768x768xf32>, %arg348: tensor<768xf32>, %arg349: tensor<768x768xf32>, %arg350: tensor<768xf32>, %arg351: tensor<768x768xf32>, %arg352: tensor<3072xf32>, %arg353: tensor<3072x768xf32>, %arg354: tensor<768xf32>, %arg355: tensor<768xf32>, %arg356: tensor<768xf32>, %arg357: tensor<768x3072xf32>, %arg358: tensor<768xf32>, %arg359: tensor<768xf32>, %arg360: tensor<768xf32>, %arg361: tensor<768x768xf32>, %arg362: tensor<768xf32>, %arg363: tensor<768x768xf32>, %arg364: tensor<768xf32>, %arg365: tensor<768x768xf32>, %arg366: tensor<768xf32>, %arg367: tensor<768x768xf32>, %arg368: tensor<3072xf32>, %arg369: tensor<3072x768xf32>, %arg370: tensor<768xf32>, %arg371: tensor<768xf32>, %arg372: tensor<768xf32>, %arg373: tensor<768x3072xf32>, %arg374: tensor<768xf32>, %arg375: tensor<768x768xf32>, %arg376: tensor<768xf32>, %arg377: tensor<768xf32>, %arg378: tensor<512x768xf32>, %arg379: tensor<2x768xf32>, %arg380: tensor<21128x768xf32>, %arg381: tensor<768xf32>, %arg382: tensor<768xf32>, %arg383: tensor<768xf32>, %arg384: tensor<768x768xf32>, %arg385: tensor<768xf32>, %arg386: tensor<768x768xf32>, %arg387: tensor<768xf32>, %arg388: tensor<768x768xf32>, %arg389: tensor<768xf32>, %arg390: tensor<768x768xf32>, %arg391: tensor<3072xf32>, %arg392: tensor<3072x768xf32>, %arg393: tensor<768xf32>, %arg394: tensor<768xf32>, %arg395: tensor<768xf32>, %arg396: tensor<768x3072xf32>, %arg397: tensor<768xf32>, %arg398: tensor<768xf32>, %arg399: tensor<768xf32>, %arg400: tensor<768x768xf32>, %arg401: tensor<768xf32>, %arg402: tensor<768x768xf32>, %arg403: tensor<768xf32>, %arg404: tensor<768x768xf32>, %arg405: tensor<768xf32>, %arg406: tensor<768x768xf32>, %arg407: tensor<3072xf32>, %arg408: tensor<3072x768xf32>, %arg409: tensor<768xf32>, %arg410: tensor<768xf32>, %arg411: tensor<768xf32>, %arg412: tensor<768x3072xf32>, %arg413: tensor<768xf32>, %arg414: tensor<768xf32>, %arg415: tensor<768xf32>, %arg416: tensor<768x768xf32>, %arg417: tensor<768xf32>, %arg418: tensor<768x768xf32>, %arg419: tensor<768xf32>, %arg420: tensor<768x768xf32>, %arg421: tensor<768xf32>, %arg422: tensor<768x768xf32>, %arg423: tensor<3072xf32>, %arg424: tensor<3072x768xf32>, %arg425: tensor<768xf32>, %arg426: tensor<768xf32>, %arg427: tensor<768xf32>, %arg428: tensor<768x3072xf32>, %arg429: tensor<768xf32>, %arg430: tensor<768xf32>, %arg431: tensor<768xf32>, %arg432: tensor<768x768xf32>, %arg433: tensor<768xf32>, %arg434: tensor<768x768xf32>, %arg435: tensor<768xf32>, %arg436: tensor<768x768xf32>, %arg437: tensor<768xf32>, %arg438: tensor<768x768xf32>, %arg439: tensor<3072xf32>, %arg440: tensor<3072x768xf32>, %arg441: tensor<768xf32>, %arg442: tensor<768xf32>, %arg443: tensor<768xf32>, %arg444: tensor<768x3072xf32>, %arg445: tensor<768xf32>, %arg446: tensor<768xf32>, %arg447: tensor<768xf32>, %arg448: tensor<768x768xf32>, %arg449: tensor<768xf32>, %arg450: tensor<768x768xf32>, %arg451: tensor<768xf32>, %arg452: tensor<768x768xf32>, %arg453: tensor<768xf32>, %arg454: tensor<768x768xf32>, %arg455: tensor<3072xf32>, %arg456: tensor<3072x768xf32>, %arg457: tensor<768xf32>, %arg458: tensor<768xf32>, %arg459: tensor<768xf32>, %arg460: tensor<768x3072xf32>, %arg461: tensor<768xf32>, %arg462: tensor<768xf32>, %arg463: tensor<768xf32>, %arg464: tensor<768x768xf32>, %arg465: tensor<768xf32>, %arg466: tensor<768x768xf32>, %arg467: tensor<768xf32>, %arg468: tensor<768x768xf32>, %arg469: tensor<768xf32>, %arg470: tensor<768x768xf32>, %arg471: tensor<3072xf32>, %arg472: tensor<3072x768xf32>, %arg473: tensor<768xf32>, %arg474: tensor<768xf32>, %arg475: tensor<768xf32>, %arg476: tensor<768x3072xf32>, %arg477: tensor<768xf32>, %arg478: tensor<768xf32>, %arg479: tensor<768xf32>, %arg480: tensor<768x768xf32>, %arg481: tensor<768xf32>, %arg482: tensor<768x768xf32>, %arg483: tensor<768xf32>, %arg484: tensor<768x768xf32>, %arg485: tensor<768xf32>, %arg486: tensor<768x768xf32>, %arg487: tensor<3072xf32>, %arg488: tensor<3072x768xf32>, %arg489: tensor<768xf32>, %arg490: tensor<768xf32>, %arg491: tensor<768xf32>, %arg492: tensor<768x3072xf32>, %arg493: tensor<768xf32>, %arg494: tensor<768xf32>, %arg495: tensor<768xf32>, %arg496: tensor<768x768xf32>, %arg497: tensor<768xf32>, %arg498: tensor<768x768xf32>, %arg499: tensor<768xf32>, %arg500: tensor<768x768xf32>, %arg501: tensor<768xf32>, %arg502: tensor<768x768xf32>, %arg503: tensor<3072xf32>, %arg504: tensor<3072x768xf32>, %arg505: tensor<768xf32>, %arg506: tensor<768xf32>, %arg507: tensor<768xf32>, %arg508: tensor<768x3072xf32>, %arg509: tensor<768xf32>, %arg510: tensor<768xf32>, %arg511: tensor<768xf32>, %arg512: tensor<768x768xf32>, %arg513: tensor<768xf32>, %arg514: tensor<768x768xf32>, %arg515: tensor<768xf32>, %arg516: tensor<768x768xf32>, %arg517: tensor<768xf32>, %arg518: tensor<768x768xf32>, %arg519: tensor<3072xf32>, %arg520: tensor<3072x768xf32>, %arg521: tensor<768xf32>, %arg522: tensor<768xf32>, %arg523: tensor<768xf32>, %arg524: tensor<768x3072xf32>, %arg525: tensor<768xf32>, %arg526: tensor<768xf32>, %arg527: tensor<768xf32>, %arg528: tensor<768x768xf32>, %arg529: tensor<768xf32>, %arg530: tensor<768x768xf32>, %arg531: tensor<768xf32>, %arg532: tensor<768x768xf32>, %arg533: tensor<768xf32>, %arg534: tensor<768x768xf32>, %arg535: tensor<3072xf32>, %arg536: tensor<3072x768xf32>, %arg537: tensor<768xf32>, %arg538: tensor<768xf32>, %arg539: tensor<768xf32>, %arg540: tensor<768x3072xf32>, %arg541: tensor<768xf32>, %arg542: tensor<768xf32>, %arg543: tensor<768xf32>, %arg544: tensor<768x768xf32>, %arg545: tensor<768xf32>, %arg546: tensor<768x768xf32>, %arg547: tensor<768xf32>, %arg548: tensor<768x768xf32>, %arg549: tensor<768xf32>, %arg550: tensor<768x768xf32>, %arg551: tensor<3072xf32>, %arg552: tensor<3072x768xf32>, %arg553: tensor<768xf32>, %arg554: tensor<768xf32>, %arg555: tensor<768xf32>, %arg556: tensor<768x3072xf32>, %arg557: tensor<768xf32>, %arg558: tensor<768xf32>, %arg559: tensor<768xf32>, %arg560: tensor<768x768xf32>, %arg561: tensor<768xf32>, %arg562: tensor<768x768xf32>, %arg563: tensor<768xf32>, %arg564: tensor<768x768xf32>, %arg565: tensor<768xf32>, %arg566: tensor<768x768xf32>, %arg567: tensor<3072xf32>, %arg568: tensor<3072x768xf32>, %arg569: tensor<768xf32>, %arg570: tensor<768xf32>, %arg571: tensor<768xf32>, %arg572: tensor<768x3072xf32>, %arg573: tensor<768xf32>, %arg574: tensor<768x768xf32>, %arg575: tensor<i64>, %arg576: tensor<64xf32>, %arg577: tensor<64xf32>, %arg578: tensor<i64>, %arg579: tensor<64xf32>, %arg580: tensor<64xf32>, %arg581: tensor<i64>, %arg582: tensor<64xf32>, %arg583: tensor<64xf32>, %arg584: tensor<i64>, %arg585: tensor<256xf32>, %arg586: tensor<256xf32>, %arg587: tensor<i64>, %arg588: tensor<256xf32>, %arg589: tensor<256xf32>, %arg590: tensor<i64>, %arg591: tensor<64xf32>, %arg592: tensor<64xf32>, %arg593: tensor<i64>, %arg594: tensor<64xf32>, %arg595: tensor<64xf32>, %arg596: tensor<i64>, %arg597: tensor<256xf32>, %arg598: tensor<256xf32>, %arg599: tensor<i64>, %arg600: tensor<64xf32>, %arg601: tensor<64xf32>, %arg602: tensor<i64>, %arg603: tensor<64xf32>, %arg604: tensor<64xf32>, %arg605: tensor<i64>, %arg606: tensor<256xf32>, %arg607: tensor<256xf32>, %arg608: tensor<i64>, %arg609: tensor<128xf32>, %arg610: tensor<128xf32>, %arg611: tensor<i64>, %arg612: tensor<128xf32>, %arg613: tensor<128xf32>, %arg614: tensor<i64>, %arg615: tensor<512xf32>, %arg616: tensor<512xf32>, %arg617: tensor<i64>, %arg618: tensor<512xf32>, %arg619: tensor<512xf32>, %arg620: tensor<i64>, %arg621: tensor<128xf32>, %arg622: tensor<128xf32>, %arg623: tensor<i64>, %arg624: tensor<128xf32>, %arg625: tensor<128xf32>, %arg626: tensor<i64>, %arg627: tensor<512xf32>, %arg628: tensor<512xf32>, %arg629: tensor<i64>, %arg630: tensor<128xf32>, %arg631: tensor<128xf32>, %arg632: tensor<i64>, %arg633: tensor<128xf32>, %arg634: tensor<128xf32>, %arg635: tensor<i64>, %arg636: tensor<512xf32>, %arg637: tensor<512xf32>, %arg638: tensor<i64>, %arg639: tensor<128xf32>, %arg640: tensor<128xf32>, %arg641: tensor<i64>, %arg642: tensor<128xf32>, %arg643: tensor<128xf32>, %arg644: tensor<i64>, %arg645: tensor<512xf32>, %arg646: tensor<512xf32>, %arg647: tensor<i64>, %arg648: tensor<256xf32>, %arg649: tensor<256xf32>, %arg650: tensor<i64>, %arg651: tensor<256xf32>, %arg652: tensor<256xf32>, %arg653: tensor<i64>, %arg654: tensor<1024xf32>, %arg655: tensor<1024xf32>, %arg656: tensor<i64>, %arg657: tensor<1024xf32>, %arg658: tensor<1024xf32>, %arg659: tensor<i64>, %arg660: tensor<256xf32>, %arg661: tensor<256xf32>, %arg662: tensor<i64>, %arg663: tensor<256xf32>, %arg664: tensor<256xf32>, %arg665: tensor<i64>, %arg666: tensor<1024xf32>, %arg667: tensor<1024xf32>, %arg668: tensor<i64>, %arg669: tensor<256xf32>, %arg670: tensor<256xf32>, %arg671: tensor<i64>, %arg672: tensor<256xf32>, %arg673: tensor<256xf32>, %arg674: tensor<i64>, %arg675: tensor<1024xf32>, %arg676: tensor<1024xf32>, %arg677: tensor<i64>, %arg678: tensor<256xf32>, %arg679: tensor<256xf32>, %arg680: tensor<i64>, %arg681: tensor<256xf32>, %arg682: tensor<256xf32>, %arg683: tensor<i64>, %arg684: tensor<1024xf32>, %arg685: tensor<1024xf32>, %arg686: tensor<i64>, %arg687: tensor<256xf32>, %arg688: tensor<256xf32>, %arg689: tensor<i64>, %arg690: tensor<256xf32>, %arg691: tensor<256xf32>, %arg692: tensor<i64>, %arg693: tensor<1024xf32>, %arg694: tensor<1024xf32>, %arg695: tensor<i64>, %arg696: tensor<256xf32>, %arg697: tensor<256xf32>, %arg698: tensor<i64>, %arg699: tensor<256xf32>, %arg700: tensor<256xf32>, %arg701: tensor<i64>, %arg702: tensor<1024xf32>, %arg703: tensor<1024xf32>, %arg704: tensor<i64>, %arg705: tensor<512xf32>, %arg706: tensor<512xf32>, %arg707: tensor<i64>, %arg708: tensor<512xf32>, %arg709: tensor<512xf32>, %arg710: tensor<i64>, %arg711: tensor<2048xf32>, %arg712: tensor<2048xf32>, %arg713: tensor<i64>, %arg714: tensor<2048xf32>, %arg715: tensor<2048xf32>, %arg716: tensor<i64>, %arg717: tensor<512xf32>, %arg718: tensor<512xf32>, %arg719: tensor<i64>, %arg720: tensor<512xf32>, %arg721: tensor<512xf32>, %arg722: tensor<i64>, %arg723: tensor<2048xf32>, %arg724: tensor<2048xf32>, %arg725: tensor<i64>, %arg726: tensor<512xf32>, %arg727: tensor<512xf32>, %arg728: tensor<i64>, %arg729: tensor<512xf32>, %arg730: tensor<512xf32>, %arg731: tensor<i64>, %arg732: tensor<2048xf32>, %arg733: tensor<2048xf32>, %arg734: tensor<1x512xi64>, %arg735: tensor<1x512xi64>, %arg736: tensor<1x512xi64>, %arg737: tensor<1x512xi64>, %arg738: tensor<16x3x256x256xf32>, %arg739: tensor<2x240xi1>, %arg740: tensor<2x240xi64>, %arg741: tensor<0xf32>, %arg742: tensor<2x19xi64>) -> !tuple {
    %0 = call @aten.view.1080(%arg740) : (tensor<2x240xi64>) -> tensor<480xi64>
    %1 = call @aten.index_select.1100(%arg380, %0) : (tensor<21128x768xf32>, tensor<480xi64>) -> tensor<480x768xf32>
    %2 = call @aten.view.1090(%1) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %3 = "mhlo.slice"(%arg737) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %4 = "mhlo.slice"(%3) {limit_indices = dense<[1, 240]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x240xi64>
    %5 = call @aten.expand.1074(%4) : (tensor<1x240xi64>) -> tensor<2x240xi64>
    %6 = call @aten.view.1080(%5) : (tensor<2x240xi64>) -> tensor<480xi64>
    %7 = call @aten.index_select.1084(%arg379, %6) : (tensor<2x768xf32>, tensor<480xi64>) -> tensor<480x768xf32>
    %8 = call @aten.view.1090(%7) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %9 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = call @aten.expand.772(%9) : (tensor<f32>) -> tensor<2x240x768xf32>
    %11 = call @aten.mul.1094(%8, %10) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %12 = call @aten.add.1107(%2, %11) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %13 = "mhlo.slice"(%arg736) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %14 = "mhlo.slice"(%13) {limit_indices = dense<[1, 240]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x240xi64>
    %15 = call @aten.view.1051(%14) : (tensor<1x240xi64>) -> tensor<240xi64>
    %16 = call @aten.index_select.1055(%arg378, %15) : (tensor<512x768xf32>, tensor<240xi64>) -> tensor<240x768xf32>
    %17 = call @aten.view.1061(%16) : (tensor<240x768xf32>) -> tensor<1x240x768xf32>
    %18 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %19 = call @aten.expand.1042(%18) : (tensor<f32>) -> tensor<1x240x768xf32>
    %20 = call @aten.mul.1065(%17, %19) : (tensor<1x240x768xf32>, tensor<1x240x768xf32>) -> tensor<1x240x768xf32>
    %21 = call @aten.add.1112(%12, %20) : (tensor<2x240x768xf32>, tensor<1x240x768xf32>) -> tensor<2x240x768xf32>
    %22 = call @aten.view.1120(%21) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %23 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %24 = call @aten.expand.758(%23) : (tensor<f32>) -> tensor<480xf32>
    %25 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = call @aten.expand.758(%25) : (tensor<f32>) -> tensor<480xf32>
    %27 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = call @aten.expand.758(%27) : (tensor<f32>) -> tensor<480xf32>
    %29 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %30 = call @aten.expand.758(%29) : (tensor<f32>) -> tensor<480xf32>
    %31 = call @aten.native_batch_norm.1124(%22, %24, %26, %28, %30) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %32 = "mhlo.get_tuple_element"(%31) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %33 = "mhlo.get_tuple_element"(%31) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %34 = call @aten.view.1144(%33) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %35 = call @aten.mul.1148(%34, %arg377) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %36 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %37 = call @aten.mul.1154(%35, %36) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %38 = call @aten.add.1160(%arg376, %37) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %39 = call @aten.view.1169(%38) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %40 = call @aten.permute.752(%arg388) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %41 = call @aten.addmm.1173(%39, %40, %arg387) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %42 = call @aten.view.1090(%41) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %43 = call @aten.view.1185(%42) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %44 = call @aten.permute.1189(%43) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %45 = call @aten.expand.1193(%44) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %46 = call @aten.view.1197(%45) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %47 = call @aten.view.1169(%38) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %48 = call @aten.permute.752(%arg386) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %49 = call @aten.addmm.1173(%47, %48, %arg385) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %50 = call @aten.view.1090(%49) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %51 = call @aten.view.1185(%50) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %52 = call @aten.permute.1189(%51) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %53 = call @aten.permute.1249(%52) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %54 = call @aten.expand.1253(%53) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %55 = call @aten.view.1257(%54) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %56 = call @aten.matmul.1269(%46, %55) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %57 = call @aten.view.1274(%56) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %58 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %59 = call @aten.div.1278(%57, %58) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %60 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %61 = call @aten.expand.1202(%60) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %62 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %63 = call @aten.expand.1202(%62) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %64 = "mhlo.slice"(%arg739) {limit_indices = dense<[2, 240]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x240xi1>) -> tensor<2x240xi1>
    %65 = call @aten.view.1211(%64) : (tensor<2x240xi1>) -> tensor<2x1x240xi1>
    %66 = call @aten.view.1215(%65) : (tensor<2x1x240xi1>) -> tensor<2x1x1x240xi1>
    %67 = "mhlo.slice"(%66) {limit_indices = dense<[2, 1, 1, 240]> : tensor<4xi64>, start_indices = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<2x1x1x240xi1>) -> tensor<2x1x1x240xi1>
    %68 = "mhlo.convert"(%67) : (tensor<2x1x1x240xi1>) -> tensor<2x1x1x240xf32>
    %69 = call @aten.mul.1223(%63, %68) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %70 = call @aten.sub.1230(%61, %69) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %71 = mhlo.constant dense<-1.000000e+04> : tensor<f32>
    %72 = call @aten.mul.1235(%70, %71) : (tensor<2x1x1x240xf32>, tensor<f32>) -> tensor<2x1x1x240xf32>
    %73 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %74 = call @aten.expand.1202(%73) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %75 = call @aten.mul.1223(%72, %74) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %76 = call @aten.add.1284(%59, %75) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %77 = call @aten.softmax.1300(%76) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %78 = call @aten.expand.1312(%77) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %79 = call @aten.view.1316(%78) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %80 = call @aten.view.1169(%38) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %81 = call @aten.permute.752(%arg390) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %82 = call @aten.addmm.1173(%80, %81, %arg389) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %83 = call @aten.view.1090(%82) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %84 = call @aten.view.1185(%83) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %85 = call @aten.permute.1189(%84) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %86 = call @aten.expand.1193(%85) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %87 = call @aten.view.1197(%86) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %88 = call @aten.matmul.1320(%79, %87) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %89 = call @aten.view.1325(%88) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %90 = call @aten.permute.1329(%89) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %91 = call @aten.view.1333(%90) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %92 = call @aten.view.1169(%91) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %93 = call @aten.permute.752(%arg384) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %94 = call @aten.addmm.1173(%92, %93, %arg383) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %95 = call @aten.view.1090(%94) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %96 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %97 = call @aten.expand.772(%96) : (tensor<f32>) -> tensor<2x240x768xf32>
    %98 = call @aten.mul.1094(%38, %97) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %99 = call @aten.add.1107(%95, %98) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %100 = call @aten.view.1120(%99) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %101 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %102 = call @aten.expand.758(%101) : (tensor<f32>) -> tensor<480xf32>
    %103 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %104 = call @aten.expand.758(%103) : (tensor<f32>) -> tensor<480xf32>
    %105 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %106 = call @aten.expand.758(%105) : (tensor<f32>) -> tensor<480xf32>
    %107 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %108 = call @aten.expand.758(%107) : (tensor<f32>) -> tensor<480xf32>
    %109 = call @aten.native_batch_norm.1124(%100, %102, %104, %106, %108) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %110 = "mhlo.get_tuple_element"(%109) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %111 = "mhlo.get_tuple_element"(%109) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %112 = call @aten.view.1144(%111) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %113 = call @aten.mul.1148(%112, %arg382) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %114 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %115 = call @aten.mul.1154(%113, %114) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %116 = call @aten.add.1160(%arg381, %115) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %117 = call @aten.view.1169(%116) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %118 = call @aten.permute.1356(%arg392) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %119 = call @aten.addmm.1361(%117, %118, %arg391) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %120 = call @aten.view.1372(%119) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %121 = call @aten.gelu.1376(%120) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %122 = call @aten.view.1450(%121) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %123 = call @aten.permute.1352(%arg396) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %124 = call @aten.addmm.1454(%122, %123, %arg395) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %125 = call @aten.view.1090(%124) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %126 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %127 = call @aten.expand.772(%126) : (tensor<f32>) -> tensor<2x240x768xf32>
    %128 = call @aten.mul.1094(%116, %127) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %129 = call @aten.add.1107(%125, %128) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %130 = call @aten.view.1120(%129) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %131 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %132 = call @aten.expand.758(%131) : (tensor<f32>) -> tensor<480xf32>
    %133 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %134 = call @aten.expand.758(%133) : (tensor<f32>) -> tensor<480xf32>
    %135 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %136 = call @aten.expand.758(%135) : (tensor<f32>) -> tensor<480xf32>
    %137 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %138 = call @aten.expand.758(%137) : (tensor<f32>) -> tensor<480xf32>
    %139 = call @aten.native_batch_norm.1124(%130, %132, %134, %136, %138) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %140 = "mhlo.get_tuple_element"(%139) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %141 = "mhlo.get_tuple_element"(%139) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %142 = call @aten.view.1144(%141) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %143 = call @aten.mul.1148(%142, %arg394) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %144 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %145 = call @aten.mul.1154(%143, %144) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %146 = call @aten.add.1160(%arg393, %145) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %147 = call @aten.view.1169(%146) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %148 = call @aten.permute.752(%arg404) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %149 = call @aten.addmm.1173(%147, %148, %arg403) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %150 = call @aten.view.1090(%149) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %151 = call @aten.view.1185(%150) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %152 = call @aten.permute.1189(%151) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %153 = call @aten.expand.1193(%152) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %154 = call @aten.view.1197(%153) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %155 = call @aten.view.1169(%146) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %156 = call @aten.permute.752(%arg402) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %157 = call @aten.addmm.1173(%155, %156, %arg401) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %158 = call @aten.view.1090(%157) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %159 = call @aten.view.1185(%158) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %160 = call @aten.permute.1189(%159) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %161 = call @aten.permute.1249(%160) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %162 = call @aten.expand.1253(%161) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %163 = call @aten.view.1257(%162) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %164 = call @aten.matmul.1269(%154, %163) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %165 = call @aten.view.1274(%164) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %166 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %167 = call @aten.div.1278(%165, %166) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %168 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %169 = call @aten.expand.1202(%168) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %170 = call @aten.mul.1223(%72, %169) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %171 = call @aten.add.1284(%167, %170) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %172 = call @aten.softmax.1300(%171) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %173 = call @aten.expand.1312(%172) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %174 = call @aten.view.1316(%173) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %175 = call @aten.view.1169(%146) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %176 = call @aten.permute.752(%arg406) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %177 = call @aten.addmm.1173(%175, %176, %arg405) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %178 = call @aten.view.1090(%177) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %179 = call @aten.view.1185(%178) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %180 = call @aten.permute.1189(%179) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %181 = call @aten.expand.1193(%180) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %182 = call @aten.view.1197(%181) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %183 = call @aten.matmul.1320(%174, %182) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %184 = call @aten.view.1325(%183) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %185 = call @aten.permute.1329(%184) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %186 = call @aten.view.1333(%185) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %187 = call @aten.view.1169(%186) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %188 = call @aten.permute.752(%arg400) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %189 = call @aten.addmm.1173(%187, %188, %arg399) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %190 = call @aten.view.1090(%189) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %191 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %192 = call @aten.expand.772(%191) : (tensor<f32>) -> tensor<2x240x768xf32>
    %193 = call @aten.mul.1094(%146, %192) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %194 = call @aten.add.1107(%190, %193) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %195 = call @aten.view.1120(%194) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %196 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %197 = call @aten.expand.758(%196) : (tensor<f32>) -> tensor<480xf32>
    %198 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %199 = call @aten.expand.758(%198) : (tensor<f32>) -> tensor<480xf32>
    %200 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %201 = call @aten.expand.758(%200) : (tensor<f32>) -> tensor<480xf32>
    %202 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %203 = call @aten.expand.758(%202) : (tensor<f32>) -> tensor<480xf32>
    %204 = call @aten.native_batch_norm.1124(%195, %197, %199, %201, %203) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %205 = "mhlo.get_tuple_element"(%204) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %206 = "mhlo.get_tuple_element"(%204) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %207 = call @aten.view.1144(%206) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %208 = call @aten.mul.1148(%207, %arg398) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %209 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %210 = call @aten.mul.1154(%208, %209) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %211 = call @aten.add.1160(%arg397, %210) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %212 = call @aten.view.1169(%211) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %213 = call @aten.permute.1356(%arg408) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %214 = call @aten.addmm.1361(%212, %213, %arg407) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %215 = call @aten.view.1372(%214) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %216 = call @aten.gelu.1376(%215) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %217 = call @aten.view.1450(%216) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %218 = call @aten.permute.1352(%arg412) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %219 = call @aten.addmm.1454(%217, %218, %arg411) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %220 = call @aten.view.1090(%219) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %221 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %222 = call @aten.expand.772(%221) : (tensor<f32>) -> tensor<2x240x768xf32>
    %223 = call @aten.mul.1094(%211, %222) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %224 = call @aten.add.1107(%220, %223) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %225 = call @aten.view.1120(%224) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %226 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %227 = call @aten.expand.758(%226) : (tensor<f32>) -> tensor<480xf32>
    %228 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %229 = call @aten.expand.758(%228) : (tensor<f32>) -> tensor<480xf32>
    %230 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %231 = call @aten.expand.758(%230) : (tensor<f32>) -> tensor<480xf32>
    %232 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %233 = call @aten.expand.758(%232) : (tensor<f32>) -> tensor<480xf32>
    %234 = call @aten.native_batch_norm.1124(%225, %227, %229, %231, %233) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %235 = "mhlo.get_tuple_element"(%234) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %236 = "mhlo.get_tuple_element"(%234) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %237 = call @aten.view.1144(%236) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %238 = call @aten.mul.1148(%237, %arg410) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %239 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %240 = call @aten.mul.1154(%238, %239) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %241 = call @aten.add.1160(%arg409, %240) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %242 = call @aten.view.1169(%241) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %243 = call @aten.permute.752(%arg452) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %244 = call @aten.addmm.1173(%242, %243, %arg451) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %245 = call @aten.view.1090(%244) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %246 = call @aten.view.1185(%245) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %247 = call @aten.permute.1189(%246) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %248 = call @aten.expand.1193(%247) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %249 = call @aten.view.1197(%248) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %250 = call @aten.view.1169(%241) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %251 = call @aten.permute.752(%arg450) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %252 = call @aten.addmm.1173(%250, %251, %arg449) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %253 = call @aten.view.1090(%252) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %254 = call @aten.view.1185(%253) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %255 = call @aten.permute.1189(%254) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %256 = call @aten.permute.1249(%255) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %257 = call @aten.expand.1253(%256) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %258 = call @aten.view.1257(%257) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %259 = call @aten.matmul.1269(%249, %258) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %260 = call @aten.view.1274(%259) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %261 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %262 = call @aten.div.1278(%260, %261) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %263 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %264 = call @aten.expand.1202(%263) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %265 = call @aten.mul.1223(%72, %264) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %266 = call @aten.add.1284(%262, %265) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %267 = call @aten.softmax.1300(%266) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %268 = call @aten.expand.1312(%267) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %269 = call @aten.view.1316(%268) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %270 = call @aten.view.1169(%241) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %271 = call @aten.permute.752(%arg454) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %272 = call @aten.addmm.1173(%270, %271, %arg453) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %273 = call @aten.view.1090(%272) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %274 = call @aten.view.1185(%273) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %275 = call @aten.permute.1189(%274) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %276 = call @aten.expand.1193(%275) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %277 = call @aten.view.1197(%276) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %278 = call @aten.matmul.1320(%269, %277) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %279 = call @aten.view.1325(%278) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %280 = call @aten.permute.1329(%279) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %281 = call @aten.view.1333(%280) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %282 = call @aten.view.1169(%281) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %283 = call @aten.permute.752(%arg448) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %284 = call @aten.addmm.1173(%282, %283, %arg447) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %285 = call @aten.view.1090(%284) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %286 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %287 = call @aten.expand.772(%286) : (tensor<f32>) -> tensor<2x240x768xf32>
    %288 = call @aten.mul.1094(%241, %287) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %289 = call @aten.add.1107(%285, %288) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %290 = call @aten.view.1120(%289) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %291 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %292 = call @aten.expand.758(%291) : (tensor<f32>) -> tensor<480xf32>
    %293 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %294 = call @aten.expand.758(%293) : (tensor<f32>) -> tensor<480xf32>
    %295 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %296 = call @aten.expand.758(%295) : (tensor<f32>) -> tensor<480xf32>
    %297 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %298 = call @aten.expand.758(%297) : (tensor<f32>) -> tensor<480xf32>
    %299 = call @aten.native_batch_norm.1124(%290, %292, %294, %296, %298) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %300 = "mhlo.get_tuple_element"(%299) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %301 = "mhlo.get_tuple_element"(%299) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %302 = call @aten.view.1144(%301) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %303 = call @aten.mul.1148(%302, %arg446) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %304 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %305 = call @aten.mul.1154(%303, %304) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %306 = call @aten.add.1160(%arg445, %305) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %307 = call @aten.view.1169(%306) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %308 = call @aten.permute.1356(%arg456) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %309 = call @aten.addmm.1361(%307, %308, %arg455) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %310 = call @aten.view.1372(%309) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %311 = call @aten.gelu.1376(%310) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %312 = call @aten.view.1450(%311) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %313 = call @aten.permute.1352(%arg460) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %314 = call @aten.addmm.1454(%312, %313, %arg459) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %315 = call @aten.view.1090(%314) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %316 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %317 = call @aten.expand.772(%316) : (tensor<f32>) -> tensor<2x240x768xf32>
    %318 = call @aten.mul.1094(%306, %317) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %319 = call @aten.add.1107(%315, %318) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %320 = call @aten.view.1120(%319) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %321 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %322 = call @aten.expand.758(%321) : (tensor<f32>) -> tensor<480xf32>
    %323 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %324 = call @aten.expand.758(%323) : (tensor<f32>) -> tensor<480xf32>
    %325 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %326 = call @aten.expand.758(%325) : (tensor<f32>) -> tensor<480xf32>
    %327 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %328 = call @aten.expand.758(%327) : (tensor<f32>) -> tensor<480xf32>
    %329 = call @aten.native_batch_norm.1124(%320, %322, %324, %326, %328) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %330 = "mhlo.get_tuple_element"(%329) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %331 = "mhlo.get_tuple_element"(%329) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %332 = call @aten.view.1144(%331) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %333 = call @aten.mul.1148(%332, %arg458) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %334 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %335 = call @aten.mul.1154(%333, %334) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %336 = call @aten.add.1160(%arg457, %335) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %337 = call @aten.view.1169(%336) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %338 = call @aten.permute.752(%arg468) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %339 = call @aten.addmm.1173(%337, %338, %arg467) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %340 = call @aten.view.1090(%339) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %341 = call @aten.view.1185(%340) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %342 = call @aten.permute.1189(%341) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %343 = call @aten.expand.1193(%342) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %344 = call @aten.view.1197(%343) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %345 = call @aten.view.1169(%336) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %346 = call @aten.permute.752(%arg466) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %347 = call @aten.addmm.1173(%345, %346, %arg465) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %348 = call @aten.view.1090(%347) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %349 = call @aten.view.1185(%348) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %350 = call @aten.permute.1189(%349) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %351 = call @aten.permute.1249(%350) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %352 = call @aten.expand.1253(%351) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %353 = call @aten.view.1257(%352) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %354 = call @aten.matmul.1269(%344, %353) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %355 = call @aten.view.1274(%354) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %356 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %357 = call @aten.div.1278(%355, %356) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %358 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %359 = call @aten.expand.1202(%358) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %360 = call @aten.mul.1223(%72, %359) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %361 = call @aten.add.1284(%357, %360) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %362 = call @aten.softmax.1300(%361) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %363 = call @aten.expand.1312(%362) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %364 = call @aten.view.1316(%363) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %365 = call @aten.view.1169(%336) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %366 = call @aten.permute.752(%arg470) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %367 = call @aten.addmm.1173(%365, %366, %arg469) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %368 = call @aten.view.1090(%367) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %369 = call @aten.view.1185(%368) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %370 = call @aten.permute.1189(%369) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %371 = call @aten.expand.1193(%370) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %372 = call @aten.view.1197(%371) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %373 = call @aten.matmul.1320(%364, %372) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %374 = call @aten.view.1325(%373) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %375 = call @aten.permute.1329(%374) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %376 = call @aten.view.1333(%375) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %377 = call @aten.view.1169(%376) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %378 = call @aten.permute.752(%arg464) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %379 = call @aten.addmm.1173(%377, %378, %arg463) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %380 = call @aten.view.1090(%379) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %381 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %382 = call @aten.expand.772(%381) : (tensor<f32>) -> tensor<2x240x768xf32>
    %383 = call @aten.mul.1094(%336, %382) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %384 = call @aten.add.1107(%380, %383) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %385 = call @aten.view.1120(%384) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %386 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %387 = call @aten.expand.758(%386) : (tensor<f32>) -> tensor<480xf32>
    %388 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %389 = call @aten.expand.758(%388) : (tensor<f32>) -> tensor<480xf32>
    %390 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %391 = call @aten.expand.758(%390) : (tensor<f32>) -> tensor<480xf32>
    %392 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %393 = call @aten.expand.758(%392) : (tensor<f32>) -> tensor<480xf32>
    %394 = call @aten.native_batch_norm.1124(%385, %387, %389, %391, %393) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %395 = "mhlo.get_tuple_element"(%394) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %396 = "mhlo.get_tuple_element"(%394) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %397 = call @aten.view.1144(%396) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %398 = call @aten.mul.1148(%397, %arg462) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %399 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %400 = call @aten.mul.1154(%398, %399) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %401 = call @aten.add.1160(%arg461, %400) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %402 = call @aten.view.1169(%401) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %403 = call @aten.permute.1356(%arg472) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %404 = call @aten.addmm.1361(%402, %403, %arg471) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %405 = call @aten.view.1372(%404) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %406 = call @aten.gelu.1376(%405) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %407 = call @aten.view.1450(%406) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %408 = call @aten.permute.1352(%arg476) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %409 = call @aten.addmm.1454(%407, %408, %arg475) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %410 = call @aten.view.1090(%409) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %411 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %412 = call @aten.expand.772(%411) : (tensor<f32>) -> tensor<2x240x768xf32>
    %413 = call @aten.mul.1094(%401, %412) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %414 = call @aten.add.1107(%410, %413) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %415 = call @aten.view.1120(%414) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %416 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %417 = call @aten.expand.758(%416) : (tensor<f32>) -> tensor<480xf32>
    %418 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %419 = call @aten.expand.758(%418) : (tensor<f32>) -> tensor<480xf32>
    %420 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %421 = call @aten.expand.758(%420) : (tensor<f32>) -> tensor<480xf32>
    %422 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %423 = call @aten.expand.758(%422) : (tensor<f32>) -> tensor<480xf32>
    %424 = call @aten.native_batch_norm.1124(%415, %417, %419, %421, %423) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %425 = "mhlo.get_tuple_element"(%424) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %426 = "mhlo.get_tuple_element"(%424) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %427 = call @aten.view.1144(%426) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %428 = call @aten.mul.1148(%427, %arg474) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %429 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %430 = call @aten.mul.1154(%428, %429) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %431 = call @aten.add.1160(%arg473, %430) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %432 = call @aten.view.1169(%431) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %433 = call @aten.permute.752(%arg484) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %434 = call @aten.addmm.1173(%432, %433, %arg483) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %435 = call @aten.view.1090(%434) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %436 = call @aten.view.1185(%435) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %437 = call @aten.permute.1189(%436) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %438 = call @aten.expand.1193(%437) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %439 = call @aten.view.1197(%438) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %440 = call @aten.view.1169(%431) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %441 = call @aten.permute.752(%arg482) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %442 = call @aten.addmm.1173(%440, %441, %arg481) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %443 = call @aten.view.1090(%442) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %444 = call @aten.view.1185(%443) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %445 = call @aten.permute.1189(%444) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %446 = call @aten.permute.1249(%445) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %447 = call @aten.expand.1253(%446) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %448 = call @aten.view.1257(%447) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %449 = call @aten.matmul.1269(%439, %448) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %450 = call @aten.view.1274(%449) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %451 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %452 = call @aten.div.1278(%450, %451) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %453 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %454 = call @aten.expand.1202(%453) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %455 = call @aten.mul.1223(%72, %454) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %456 = call @aten.add.1284(%452, %455) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %457 = call @aten.softmax.1300(%456) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %458 = call @aten.expand.1312(%457) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %459 = call @aten.view.1316(%458) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %460 = call @aten.view.1169(%431) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %461 = call @aten.permute.752(%arg486) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %462 = call @aten.addmm.1173(%460, %461, %arg485) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %463 = call @aten.view.1090(%462) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %464 = call @aten.view.1185(%463) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %465 = call @aten.permute.1189(%464) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %466 = call @aten.expand.1193(%465) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %467 = call @aten.view.1197(%466) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %468 = call @aten.matmul.1320(%459, %467) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %469 = call @aten.view.1325(%468) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %470 = call @aten.permute.1329(%469) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %471 = call @aten.view.1333(%470) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %472 = call @aten.view.1169(%471) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %473 = call @aten.permute.752(%arg480) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %474 = call @aten.addmm.1173(%472, %473, %arg479) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %475 = call @aten.view.1090(%474) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %476 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %477 = call @aten.expand.772(%476) : (tensor<f32>) -> tensor<2x240x768xf32>
    %478 = call @aten.mul.1094(%431, %477) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %479 = call @aten.add.1107(%475, %478) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %480 = call @aten.view.1120(%479) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %481 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %482 = call @aten.expand.758(%481) : (tensor<f32>) -> tensor<480xf32>
    %483 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %484 = call @aten.expand.758(%483) : (tensor<f32>) -> tensor<480xf32>
    %485 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %486 = call @aten.expand.758(%485) : (tensor<f32>) -> tensor<480xf32>
    %487 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %488 = call @aten.expand.758(%487) : (tensor<f32>) -> tensor<480xf32>
    %489 = call @aten.native_batch_norm.1124(%480, %482, %484, %486, %488) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %490 = "mhlo.get_tuple_element"(%489) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %491 = "mhlo.get_tuple_element"(%489) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %492 = call @aten.view.1144(%491) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %493 = call @aten.mul.1148(%492, %arg478) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %494 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %495 = call @aten.mul.1154(%493, %494) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %496 = call @aten.add.1160(%arg477, %495) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %497 = call @aten.view.1169(%496) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %498 = call @aten.permute.1356(%arg488) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %499 = call @aten.addmm.1361(%497, %498, %arg487) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %500 = call @aten.view.1372(%499) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %501 = call @aten.gelu.1376(%500) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %502 = call @aten.view.1450(%501) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %503 = call @aten.permute.1352(%arg492) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %504 = call @aten.addmm.1454(%502, %503, %arg491) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %505 = call @aten.view.1090(%504) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %506 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %507 = call @aten.expand.772(%506) : (tensor<f32>) -> tensor<2x240x768xf32>
    %508 = call @aten.mul.1094(%496, %507) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %509 = call @aten.add.1107(%505, %508) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %510 = call @aten.view.1120(%509) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %511 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %512 = call @aten.expand.758(%511) : (tensor<f32>) -> tensor<480xf32>
    %513 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %514 = call @aten.expand.758(%513) : (tensor<f32>) -> tensor<480xf32>
    %515 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %516 = call @aten.expand.758(%515) : (tensor<f32>) -> tensor<480xf32>
    %517 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %518 = call @aten.expand.758(%517) : (tensor<f32>) -> tensor<480xf32>
    %519 = call @aten.native_batch_norm.1124(%510, %512, %514, %516, %518) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %520 = "mhlo.get_tuple_element"(%519) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %521 = "mhlo.get_tuple_element"(%519) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %522 = call @aten.view.1144(%521) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %523 = call @aten.mul.1148(%522, %arg490) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %524 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %525 = call @aten.mul.1154(%523, %524) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %526 = call @aten.add.1160(%arg489, %525) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %527 = call @aten.view.1169(%526) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %528 = call @aten.permute.752(%arg500) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %529 = call @aten.addmm.1173(%527, %528, %arg499) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %530 = call @aten.view.1090(%529) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %531 = call @aten.view.1185(%530) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %532 = call @aten.permute.1189(%531) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %533 = call @aten.expand.1193(%532) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %534 = call @aten.view.1197(%533) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %535 = call @aten.view.1169(%526) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %536 = call @aten.permute.752(%arg498) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %537 = call @aten.addmm.1173(%535, %536, %arg497) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %538 = call @aten.view.1090(%537) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %539 = call @aten.view.1185(%538) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %540 = call @aten.permute.1189(%539) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %541 = call @aten.permute.1249(%540) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %542 = call @aten.expand.1253(%541) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %543 = call @aten.view.1257(%542) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %544 = call @aten.matmul.1269(%534, %543) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %545 = call @aten.view.1274(%544) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %546 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %547 = call @aten.div.1278(%545, %546) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %548 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %549 = call @aten.expand.1202(%548) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %550 = call @aten.mul.1223(%72, %549) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %551 = call @aten.add.1284(%547, %550) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %552 = call @aten.softmax.1300(%551) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %553 = call @aten.expand.1312(%552) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %554 = call @aten.view.1316(%553) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %555 = call @aten.view.1169(%526) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %556 = call @aten.permute.752(%arg502) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %557 = call @aten.addmm.1173(%555, %556, %arg501) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %558 = call @aten.view.1090(%557) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %559 = call @aten.view.1185(%558) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %560 = call @aten.permute.1189(%559) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %561 = call @aten.expand.1193(%560) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %562 = call @aten.view.1197(%561) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %563 = call @aten.matmul.1320(%554, %562) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %564 = call @aten.view.1325(%563) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %565 = call @aten.permute.1329(%564) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %566 = call @aten.view.1333(%565) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %567 = call @aten.view.1169(%566) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %568 = call @aten.permute.752(%arg496) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %569 = call @aten.addmm.1173(%567, %568, %arg495) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %570 = call @aten.view.1090(%569) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %571 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %572 = call @aten.expand.772(%571) : (tensor<f32>) -> tensor<2x240x768xf32>
    %573 = call @aten.mul.1094(%526, %572) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %574 = call @aten.add.1107(%570, %573) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %575 = call @aten.view.1120(%574) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %576 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %577 = call @aten.expand.758(%576) : (tensor<f32>) -> tensor<480xf32>
    %578 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %579 = call @aten.expand.758(%578) : (tensor<f32>) -> tensor<480xf32>
    %580 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %581 = call @aten.expand.758(%580) : (tensor<f32>) -> tensor<480xf32>
    %582 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %583 = call @aten.expand.758(%582) : (tensor<f32>) -> tensor<480xf32>
    %584 = call @aten.native_batch_norm.1124(%575, %577, %579, %581, %583) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %585 = "mhlo.get_tuple_element"(%584) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %586 = "mhlo.get_tuple_element"(%584) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %587 = call @aten.view.1144(%586) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %588 = call @aten.mul.1148(%587, %arg494) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %589 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %590 = call @aten.mul.1154(%588, %589) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %591 = call @aten.add.1160(%arg493, %590) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %592 = call @aten.view.1169(%591) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %593 = call @aten.permute.1356(%arg504) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %594 = call @aten.addmm.1361(%592, %593, %arg503) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %595 = call @aten.view.1372(%594) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %596 = call @aten.gelu.1376(%595) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %597 = call @aten.view.1450(%596) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %598 = call @aten.permute.1352(%arg508) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %599 = call @aten.addmm.1454(%597, %598, %arg507) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %600 = call @aten.view.1090(%599) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %601 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %602 = call @aten.expand.772(%601) : (tensor<f32>) -> tensor<2x240x768xf32>
    %603 = call @aten.mul.1094(%591, %602) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %604 = call @aten.add.1107(%600, %603) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %605 = call @aten.view.1120(%604) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %606 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %607 = call @aten.expand.758(%606) : (tensor<f32>) -> tensor<480xf32>
    %608 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %609 = call @aten.expand.758(%608) : (tensor<f32>) -> tensor<480xf32>
    %610 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %611 = call @aten.expand.758(%610) : (tensor<f32>) -> tensor<480xf32>
    %612 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %613 = call @aten.expand.758(%612) : (tensor<f32>) -> tensor<480xf32>
    %614 = call @aten.native_batch_norm.1124(%605, %607, %609, %611, %613) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %615 = "mhlo.get_tuple_element"(%614) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %616 = "mhlo.get_tuple_element"(%614) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %617 = call @aten.view.1144(%616) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %618 = call @aten.mul.1148(%617, %arg506) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %619 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %620 = call @aten.mul.1154(%618, %619) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %621 = call @aten.add.1160(%arg505, %620) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %622 = call @aten.view.1169(%621) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %623 = call @aten.permute.752(%arg516) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %624 = call @aten.addmm.1173(%622, %623, %arg515) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %625 = call @aten.view.1090(%624) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %626 = call @aten.view.1185(%625) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %627 = call @aten.permute.1189(%626) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %628 = call @aten.expand.1193(%627) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %629 = call @aten.view.1197(%628) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %630 = call @aten.view.1169(%621) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %631 = call @aten.permute.752(%arg514) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %632 = call @aten.addmm.1173(%630, %631, %arg513) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %633 = call @aten.view.1090(%632) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %634 = call @aten.view.1185(%633) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %635 = call @aten.permute.1189(%634) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %636 = call @aten.permute.1249(%635) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %637 = call @aten.expand.1253(%636) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %638 = call @aten.view.1257(%637) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %639 = call @aten.matmul.1269(%629, %638) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %640 = call @aten.view.1274(%639) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %641 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %642 = call @aten.div.1278(%640, %641) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %643 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %644 = call @aten.expand.1202(%643) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %645 = call @aten.mul.1223(%72, %644) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %646 = call @aten.add.1284(%642, %645) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %647 = call @aten.softmax.1300(%646) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %648 = call @aten.expand.1312(%647) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %649 = call @aten.view.1316(%648) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %650 = call @aten.view.1169(%621) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %651 = call @aten.permute.752(%arg518) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %652 = call @aten.addmm.1173(%650, %651, %arg517) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %653 = call @aten.view.1090(%652) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %654 = call @aten.view.1185(%653) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %655 = call @aten.permute.1189(%654) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %656 = call @aten.expand.1193(%655) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %657 = call @aten.view.1197(%656) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %658 = call @aten.matmul.1320(%649, %657) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %659 = call @aten.view.1325(%658) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %660 = call @aten.permute.1329(%659) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %661 = call @aten.view.1333(%660) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %662 = call @aten.view.1169(%661) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %663 = call @aten.permute.752(%arg512) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %664 = call @aten.addmm.1173(%662, %663, %arg511) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %665 = call @aten.view.1090(%664) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %666 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %667 = call @aten.expand.772(%666) : (tensor<f32>) -> tensor<2x240x768xf32>
    %668 = call @aten.mul.1094(%621, %667) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %669 = call @aten.add.1107(%665, %668) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %670 = call @aten.view.1120(%669) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %671 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %672 = call @aten.expand.758(%671) : (tensor<f32>) -> tensor<480xf32>
    %673 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %674 = call @aten.expand.758(%673) : (tensor<f32>) -> tensor<480xf32>
    %675 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %676 = call @aten.expand.758(%675) : (tensor<f32>) -> tensor<480xf32>
    %677 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %678 = call @aten.expand.758(%677) : (tensor<f32>) -> tensor<480xf32>
    %679 = call @aten.native_batch_norm.1124(%670, %672, %674, %676, %678) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %680 = "mhlo.get_tuple_element"(%679) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %681 = "mhlo.get_tuple_element"(%679) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %682 = call @aten.view.1144(%681) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %683 = call @aten.mul.1148(%682, %arg510) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %684 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %685 = call @aten.mul.1154(%683, %684) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %686 = call @aten.add.1160(%arg509, %685) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %687 = call @aten.view.1169(%686) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %688 = call @aten.permute.1356(%arg520) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %689 = call @aten.addmm.1361(%687, %688, %arg519) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %690 = call @aten.view.1372(%689) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %691 = call @aten.gelu.1376(%690) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %692 = call @aten.view.1450(%691) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %693 = call @aten.permute.1352(%arg524) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %694 = call @aten.addmm.1454(%692, %693, %arg523) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %695 = call @aten.view.1090(%694) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %696 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %697 = call @aten.expand.772(%696) : (tensor<f32>) -> tensor<2x240x768xf32>
    %698 = call @aten.mul.1094(%686, %697) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %699 = call @aten.add.1107(%695, %698) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %700 = call @aten.view.1120(%699) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %701 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %702 = call @aten.expand.758(%701) : (tensor<f32>) -> tensor<480xf32>
    %703 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %704 = call @aten.expand.758(%703) : (tensor<f32>) -> tensor<480xf32>
    %705 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %706 = call @aten.expand.758(%705) : (tensor<f32>) -> tensor<480xf32>
    %707 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %708 = call @aten.expand.758(%707) : (tensor<f32>) -> tensor<480xf32>
    %709 = call @aten.native_batch_norm.1124(%700, %702, %704, %706, %708) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %710 = "mhlo.get_tuple_element"(%709) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %711 = "mhlo.get_tuple_element"(%709) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %712 = call @aten.view.1144(%711) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %713 = call @aten.mul.1148(%712, %arg522) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %714 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %715 = call @aten.mul.1154(%713, %714) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %716 = call @aten.add.1160(%arg521, %715) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %717 = call @aten.view.1169(%716) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %718 = call @aten.permute.752(%arg532) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %719 = call @aten.addmm.1173(%717, %718, %arg531) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %720 = call @aten.view.1090(%719) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %721 = call @aten.view.1185(%720) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %722 = call @aten.permute.1189(%721) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %723 = call @aten.expand.1193(%722) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %724 = call @aten.view.1197(%723) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %725 = call @aten.view.1169(%716) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %726 = call @aten.permute.752(%arg530) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %727 = call @aten.addmm.1173(%725, %726, %arg529) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %728 = call @aten.view.1090(%727) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %729 = call @aten.view.1185(%728) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %730 = call @aten.permute.1189(%729) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %731 = call @aten.permute.1249(%730) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %732 = call @aten.expand.1253(%731) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %733 = call @aten.view.1257(%732) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %734 = call @aten.matmul.1269(%724, %733) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %735 = call @aten.view.1274(%734) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %736 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %737 = call @aten.div.1278(%735, %736) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %738 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %739 = call @aten.expand.1202(%738) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %740 = call @aten.mul.1223(%72, %739) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %741 = call @aten.add.1284(%737, %740) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %742 = call @aten.softmax.1300(%741) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %743 = call @aten.expand.1312(%742) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %744 = call @aten.view.1316(%743) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %745 = call @aten.view.1169(%716) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %746 = call @aten.permute.752(%arg534) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %747 = call @aten.addmm.1173(%745, %746, %arg533) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %748 = call @aten.view.1090(%747) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %749 = call @aten.view.1185(%748) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %750 = call @aten.permute.1189(%749) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %751 = call @aten.expand.1193(%750) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %752 = call @aten.view.1197(%751) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %753 = call @aten.matmul.1320(%744, %752) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %754 = call @aten.view.1325(%753) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %755 = call @aten.permute.1329(%754) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %756 = call @aten.view.1333(%755) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %757 = call @aten.view.1169(%756) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %758 = call @aten.permute.752(%arg528) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %759 = call @aten.addmm.1173(%757, %758, %arg527) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %760 = call @aten.view.1090(%759) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %761 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %762 = call @aten.expand.772(%761) : (tensor<f32>) -> tensor<2x240x768xf32>
    %763 = call @aten.mul.1094(%716, %762) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %764 = call @aten.add.1107(%760, %763) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %765 = call @aten.view.1120(%764) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %766 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %767 = call @aten.expand.758(%766) : (tensor<f32>) -> tensor<480xf32>
    %768 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %769 = call @aten.expand.758(%768) : (tensor<f32>) -> tensor<480xf32>
    %770 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %771 = call @aten.expand.758(%770) : (tensor<f32>) -> tensor<480xf32>
    %772 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %773 = call @aten.expand.758(%772) : (tensor<f32>) -> tensor<480xf32>
    %774 = call @aten.native_batch_norm.1124(%765, %767, %769, %771, %773) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %775 = "mhlo.get_tuple_element"(%774) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %776 = "mhlo.get_tuple_element"(%774) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %777 = call @aten.view.1144(%776) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %778 = call @aten.mul.1148(%777, %arg526) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %779 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %780 = call @aten.mul.1154(%778, %779) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %781 = call @aten.add.1160(%arg525, %780) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %782 = call @aten.view.1169(%781) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %783 = call @aten.permute.1356(%arg536) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %784 = call @aten.addmm.1361(%782, %783, %arg535) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %785 = call @aten.view.1372(%784) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %786 = call @aten.gelu.1376(%785) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %787 = call @aten.view.1450(%786) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %788 = call @aten.permute.1352(%arg540) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %789 = call @aten.addmm.1454(%787, %788, %arg539) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %790 = call @aten.view.1090(%789) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %791 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %792 = call @aten.expand.772(%791) : (tensor<f32>) -> tensor<2x240x768xf32>
    %793 = call @aten.mul.1094(%781, %792) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %794 = call @aten.add.1107(%790, %793) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %795 = call @aten.view.1120(%794) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %796 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %797 = call @aten.expand.758(%796) : (tensor<f32>) -> tensor<480xf32>
    %798 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %799 = call @aten.expand.758(%798) : (tensor<f32>) -> tensor<480xf32>
    %800 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %801 = call @aten.expand.758(%800) : (tensor<f32>) -> tensor<480xf32>
    %802 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %803 = call @aten.expand.758(%802) : (tensor<f32>) -> tensor<480xf32>
    %804 = call @aten.native_batch_norm.1124(%795, %797, %799, %801, %803) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %805 = "mhlo.get_tuple_element"(%804) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %806 = "mhlo.get_tuple_element"(%804) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %807 = call @aten.view.1144(%806) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %808 = call @aten.mul.1148(%807, %arg538) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %809 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %810 = call @aten.mul.1154(%808, %809) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %811 = call @aten.add.1160(%arg537, %810) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %812 = call @aten.view.1169(%811) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %813 = call @aten.permute.752(%arg548) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %814 = call @aten.addmm.1173(%812, %813, %arg547) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %815 = call @aten.view.1090(%814) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %816 = call @aten.view.1185(%815) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %817 = call @aten.permute.1189(%816) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %818 = call @aten.expand.1193(%817) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %819 = call @aten.view.1197(%818) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %820 = call @aten.view.1169(%811) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %821 = call @aten.permute.752(%arg546) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %822 = call @aten.addmm.1173(%820, %821, %arg545) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %823 = call @aten.view.1090(%822) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %824 = call @aten.view.1185(%823) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %825 = call @aten.permute.1189(%824) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %826 = call @aten.permute.1249(%825) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %827 = call @aten.expand.1253(%826) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %828 = call @aten.view.1257(%827) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %829 = call @aten.matmul.1269(%819, %828) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %830 = call @aten.view.1274(%829) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %831 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %832 = call @aten.div.1278(%830, %831) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %833 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %834 = call @aten.expand.1202(%833) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %835 = call @aten.mul.1223(%72, %834) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %836 = call @aten.add.1284(%832, %835) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %837 = call @aten.softmax.1300(%836) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %838 = call @aten.expand.1312(%837) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %839 = call @aten.view.1316(%838) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %840 = call @aten.view.1169(%811) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %841 = call @aten.permute.752(%arg550) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %842 = call @aten.addmm.1173(%840, %841, %arg549) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %843 = call @aten.view.1090(%842) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %844 = call @aten.view.1185(%843) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %845 = call @aten.permute.1189(%844) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %846 = call @aten.expand.1193(%845) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %847 = call @aten.view.1197(%846) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %848 = call @aten.matmul.1320(%839, %847) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %849 = call @aten.view.1325(%848) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %850 = call @aten.permute.1329(%849) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %851 = call @aten.view.1333(%850) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %852 = call @aten.view.1169(%851) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %853 = call @aten.permute.752(%arg544) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %854 = call @aten.addmm.1173(%852, %853, %arg543) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %855 = call @aten.view.1090(%854) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %856 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %857 = call @aten.expand.772(%856) : (tensor<f32>) -> tensor<2x240x768xf32>
    %858 = call @aten.mul.1094(%811, %857) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %859 = call @aten.add.1107(%855, %858) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %860 = call @aten.view.1120(%859) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %861 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %862 = call @aten.expand.758(%861) : (tensor<f32>) -> tensor<480xf32>
    %863 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %864 = call @aten.expand.758(%863) : (tensor<f32>) -> tensor<480xf32>
    %865 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %866 = call @aten.expand.758(%865) : (tensor<f32>) -> tensor<480xf32>
    %867 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %868 = call @aten.expand.758(%867) : (tensor<f32>) -> tensor<480xf32>
    %869 = call @aten.native_batch_norm.1124(%860, %862, %864, %866, %868) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %870 = "mhlo.get_tuple_element"(%869) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %871 = "mhlo.get_tuple_element"(%869) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %872 = call @aten.view.1144(%871) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %873 = call @aten.mul.1148(%872, %arg542) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %874 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %875 = call @aten.mul.1154(%873, %874) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %876 = call @aten.add.1160(%arg541, %875) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %877 = call @aten.view.1169(%876) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %878 = call @aten.permute.1356(%arg552) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %879 = call @aten.addmm.1361(%877, %878, %arg551) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %880 = call @aten.view.1372(%879) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %881 = call @aten.gelu.1376(%880) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %882 = call @aten.view.1450(%881) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %883 = call @aten.permute.1352(%arg556) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %884 = call @aten.addmm.1454(%882, %883, %arg555) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %885 = call @aten.view.1090(%884) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %886 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %887 = call @aten.expand.772(%886) : (tensor<f32>) -> tensor<2x240x768xf32>
    %888 = call @aten.mul.1094(%876, %887) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %889 = call @aten.add.1107(%885, %888) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %890 = call @aten.view.1120(%889) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %891 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %892 = call @aten.expand.758(%891) : (tensor<f32>) -> tensor<480xf32>
    %893 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %894 = call @aten.expand.758(%893) : (tensor<f32>) -> tensor<480xf32>
    %895 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %896 = call @aten.expand.758(%895) : (tensor<f32>) -> tensor<480xf32>
    %897 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %898 = call @aten.expand.758(%897) : (tensor<f32>) -> tensor<480xf32>
    %899 = call @aten.native_batch_norm.1124(%890, %892, %894, %896, %898) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %900 = "mhlo.get_tuple_element"(%899) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %901 = "mhlo.get_tuple_element"(%899) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %902 = call @aten.view.1144(%901) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %903 = call @aten.mul.1148(%902, %arg554) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %904 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %905 = call @aten.mul.1154(%903, %904) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %906 = call @aten.add.1160(%arg553, %905) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %907 = call @aten.view.1169(%906) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %908 = call @aten.permute.752(%arg564) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %909 = call @aten.addmm.1173(%907, %908, %arg563) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %910 = call @aten.view.1090(%909) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %911 = call @aten.view.1185(%910) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %912 = call @aten.permute.1189(%911) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %913 = call @aten.expand.1193(%912) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %914 = call @aten.view.1197(%913) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %915 = call @aten.view.1169(%906) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %916 = call @aten.permute.752(%arg562) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %917 = call @aten.addmm.1173(%915, %916, %arg561) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %918 = call @aten.view.1090(%917) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %919 = call @aten.view.1185(%918) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %920 = call @aten.permute.1189(%919) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %921 = call @aten.permute.1249(%920) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %922 = call @aten.expand.1253(%921) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %923 = call @aten.view.1257(%922) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %924 = call @aten.matmul.1269(%914, %923) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %925 = call @aten.view.1274(%924) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %926 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %927 = call @aten.div.1278(%925, %926) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %928 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %929 = call @aten.expand.1202(%928) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %930 = call @aten.mul.1223(%72, %929) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %931 = call @aten.add.1284(%927, %930) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %932 = call @aten.softmax.1300(%931) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %933 = call @aten.expand.1312(%932) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %934 = call @aten.view.1316(%933) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %935 = call @aten.view.1169(%906) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %936 = call @aten.permute.752(%arg566) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %937 = call @aten.addmm.1173(%935, %936, %arg565) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %938 = call @aten.view.1090(%937) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %939 = call @aten.view.1185(%938) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %940 = call @aten.permute.1189(%939) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %941 = call @aten.expand.1193(%940) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %942 = call @aten.view.1197(%941) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %943 = call @aten.matmul.1320(%934, %942) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %944 = call @aten.view.1325(%943) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %945 = call @aten.permute.1329(%944) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %946 = call @aten.view.1333(%945) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %947 = call @aten.view.1169(%946) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %948 = call @aten.permute.752(%arg560) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %949 = call @aten.addmm.1173(%947, %948, %arg559) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %950 = call @aten.view.1090(%949) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %951 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %952 = call @aten.expand.772(%951) : (tensor<f32>) -> tensor<2x240x768xf32>
    %953 = call @aten.mul.1094(%906, %952) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %954 = call @aten.add.1107(%950, %953) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %955 = call @aten.view.1120(%954) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %956 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %957 = call @aten.expand.758(%956) : (tensor<f32>) -> tensor<480xf32>
    %958 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %959 = call @aten.expand.758(%958) : (tensor<f32>) -> tensor<480xf32>
    %960 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %961 = call @aten.expand.758(%960) : (tensor<f32>) -> tensor<480xf32>
    %962 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %963 = call @aten.expand.758(%962) : (tensor<f32>) -> tensor<480xf32>
    %964 = call @aten.native_batch_norm.1124(%955, %957, %959, %961, %963) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %965 = "mhlo.get_tuple_element"(%964) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %966 = "mhlo.get_tuple_element"(%964) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %967 = call @aten.view.1144(%966) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %968 = call @aten.mul.1148(%967, %arg558) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %969 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %970 = call @aten.mul.1154(%968, %969) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %971 = call @aten.add.1160(%arg557, %970) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %972 = call @aten.view.1169(%971) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %973 = call @aten.permute.1356(%arg568) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %974 = call @aten.addmm.1361(%972, %973, %arg567) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %975 = call @aten.view.1372(%974) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %976 = call @aten.gelu.1376(%975) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %977 = call @aten.view.1450(%976) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %978 = call @aten.permute.1352(%arg572) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %979 = call @aten.addmm.1454(%977, %978, %arg571) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %980 = call @aten.view.1090(%979) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %981 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %982 = call @aten.expand.772(%981) : (tensor<f32>) -> tensor<2x240x768xf32>
    %983 = call @aten.mul.1094(%971, %982) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %984 = call @aten.add.1107(%980, %983) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %985 = call @aten.view.1120(%984) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %986 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %987 = call @aten.expand.758(%986) : (tensor<f32>) -> tensor<480xf32>
    %988 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %989 = call @aten.expand.758(%988) : (tensor<f32>) -> tensor<480xf32>
    %990 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %991 = call @aten.expand.758(%990) : (tensor<f32>) -> tensor<480xf32>
    %992 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %993 = call @aten.expand.758(%992) : (tensor<f32>) -> tensor<480xf32>
    %994 = call @aten.native_batch_norm.1124(%985, %987, %989, %991, %993) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %995 = "mhlo.get_tuple_element"(%994) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %996 = "mhlo.get_tuple_element"(%994) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %997 = call @aten.view.1144(%996) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %998 = call @aten.mul.1148(%997, %arg570) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %999 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1000 = call @aten.mul.1154(%998, %999) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %1001 = call @aten.add.1160(%arg569, %1000) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1002 = call @aten.view.1169(%1001) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1003 = call @aten.permute.752(%arg420) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1004 = call @aten.addmm.1173(%1002, %1003, %arg419) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1005 = call @aten.view.1090(%1004) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1006 = call @aten.view.1185(%1005) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1007 = call @aten.permute.1189(%1006) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1008 = call @aten.expand.1193(%1007) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1009 = call @aten.view.1197(%1008) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1010 = call @aten.view.1169(%1001) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1011 = call @aten.permute.752(%arg418) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1012 = call @aten.addmm.1173(%1010, %1011, %arg417) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1013 = call @aten.view.1090(%1012) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1014 = call @aten.view.1185(%1013) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1015 = call @aten.permute.1189(%1014) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1016 = call @aten.permute.1249(%1015) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %1017 = call @aten.expand.1253(%1016) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1018 = call @aten.view.1257(%1017) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %1019 = call @aten.matmul.1269(%1009, %1018) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1020 = call @aten.view.1274(%1019) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1021 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1022 = call @aten.div.1278(%1020, %1021) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1023 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1024 = call @aten.expand.1202(%1023) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %1025 = call @aten.mul.1223(%72, %1024) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %1026 = call @aten.add.1284(%1022, %1025) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %1027 = call @aten.softmax.1300(%1026) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1028 = call @aten.expand.1312(%1027) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1029 = call @aten.view.1316(%1028) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1030 = call @aten.view.1169(%1001) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1031 = call @aten.permute.752(%arg422) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1032 = call @aten.addmm.1173(%1030, %1031, %arg421) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1033 = call @aten.view.1090(%1032) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1034 = call @aten.view.1185(%1033) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1035 = call @aten.permute.1189(%1034) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1036 = call @aten.expand.1193(%1035) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1037 = call @aten.view.1197(%1036) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1038 = call @aten.matmul.1320(%1029, %1037) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1039 = call @aten.view.1325(%1038) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1040 = call @aten.permute.1329(%1039) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1041 = call @aten.view.1333(%1040) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1042 = call @aten.view.1169(%1041) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1043 = call @aten.permute.752(%arg416) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1044 = call @aten.addmm.1173(%1042, %1043, %arg415) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1045 = call @aten.view.1090(%1044) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1046 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1047 = call @aten.expand.772(%1046) : (tensor<f32>) -> tensor<2x240x768xf32>
    %1048 = call @aten.mul.1094(%1001, %1047) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1049 = call @aten.add.1107(%1045, %1048) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1050 = call @aten.view.1120(%1049) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %1051 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1052 = call @aten.expand.758(%1051) : (tensor<f32>) -> tensor<480xf32>
    %1053 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1054 = call @aten.expand.758(%1053) : (tensor<f32>) -> tensor<480xf32>
    %1055 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1056 = call @aten.expand.758(%1055) : (tensor<f32>) -> tensor<480xf32>
    %1057 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1058 = call @aten.expand.758(%1057) : (tensor<f32>) -> tensor<480xf32>
    %1059 = call @aten.native_batch_norm.1124(%1050, %1052, %1054, %1056, %1058) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %1060 = "mhlo.get_tuple_element"(%1059) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1061 = "mhlo.get_tuple_element"(%1059) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %1062 = call @aten.view.1144(%1061) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %1063 = call @aten.mul.1148(%1062, %arg414) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %1064 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1065 = call @aten.mul.1154(%1063, %1064) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %1066 = call @aten.add.1160(%arg413, %1065) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1067 = call @aten.view.1169(%1066) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1068 = call @aten.permute.1356(%arg424) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1069 = call @aten.addmm.1361(%1067, %1068, %arg423) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %1070 = call @aten.view.1372(%1069) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1071 = call @aten.gelu.1376(%1070) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1072 = call @aten.view.1450(%1071) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1073 = call @aten.permute.1352(%arg428) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1074 = call @aten.addmm.1454(%1072, %1073, %arg427) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1075 = call @aten.view.1090(%1074) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1076 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1077 = call @aten.expand.772(%1076) : (tensor<f32>) -> tensor<2x240x768xf32>
    %1078 = call @aten.mul.1094(%1066, %1077) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1079 = call @aten.add.1107(%1075, %1078) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1080 = call @aten.view.1120(%1079) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %1081 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1082 = call @aten.expand.758(%1081) : (tensor<f32>) -> tensor<480xf32>
    %1083 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1084 = call @aten.expand.758(%1083) : (tensor<f32>) -> tensor<480xf32>
    %1085 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1086 = call @aten.expand.758(%1085) : (tensor<f32>) -> tensor<480xf32>
    %1087 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1088 = call @aten.expand.758(%1087) : (tensor<f32>) -> tensor<480xf32>
    %1089 = call @aten.native_batch_norm.1124(%1080, %1082, %1084, %1086, %1088) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %1090 = "mhlo.get_tuple_element"(%1089) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1091 = "mhlo.get_tuple_element"(%1089) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %1092 = call @aten.view.1144(%1091) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %1093 = call @aten.mul.1148(%1092, %arg426) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %1094 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1095 = call @aten.mul.1154(%1093, %1094) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %1096 = call @aten.add.1160(%arg425, %1095) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1097 = call @aten.view.1169(%1096) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1098 = call @aten.permute.752(%arg436) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1099 = call @aten.addmm.1173(%1097, %1098, %arg435) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1100 = call @aten.view.1090(%1099) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1101 = call @aten.view.1185(%1100) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1102 = call @aten.permute.1189(%1101) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1103 = call @aten.expand.1193(%1102) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1104 = call @aten.view.1197(%1103) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1105 = call @aten.view.1169(%1096) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1106 = call @aten.permute.752(%arg434) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1107 = call @aten.addmm.1173(%1105, %1106, %arg433) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1108 = call @aten.view.1090(%1107) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1109 = call @aten.view.1185(%1108) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1110 = call @aten.permute.1189(%1109) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1111 = call @aten.permute.1249(%1110) {xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    %1112 = call @aten.expand.1253(%1111) : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1113 = call @aten.view.1257(%1112) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    %1114 = call @aten.matmul.1269(%1104, %1113) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1115 = call @aten.view.1274(%1114) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1116 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1117 = call @aten.div.1278(%1115, %1116) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1118 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1119 = call @aten.expand.1202(%1118) : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %1120 = call @aten.mul.1223(%72, %1119) : (tensor<2x1x1x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %1121 = call @aten.add.1284(%1117, %1120) : (tensor<2x12x240x240xf32>, tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32>
    %1122 = call @aten.softmax.1300(%1121) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1123 = call @aten.expand.1312(%1122) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1124 = call @aten.view.1316(%1123) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1125 = call @aten.view.1169(%1096) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1126 = call @aten.permute.752(%arg438) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1127 = call @aten.addmm.1173(%1125, %1126, %arg437) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1128 = call @aten.view.1090(%1127) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1129 = call @aten.view.1185(%1128) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1130 = call @aten.permute.1189(%1129) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1131 = call @aten.expand.1193(%1130) : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1132 = call @aten.view.1197(%1131) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1133 = call @aten.matmul.1320(%1124, %1132) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1134 = call @aten.view.1325(%1133) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1135 = call @aten.permute.1329(%1134) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1136 = call @aten.view.1333(%1135) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1137 = call @aten.view.1169(%1136) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1138 = call @aten.permute.752(%arg432) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1139 = call @aten.addmm.1173(%1137, %1138, %arg431) : (tensor<480x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1140 = call @aten.view.1090(%1139) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1141 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1142 = call @aten.expand.772(%1141) : (tensor<f32>) -> tensor<2x240x768xf32>
    %1143 = call @aten.mul.1094(%1096, %1142) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1144 = call @aten.add.1107(%1140, %1143) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1145 = call @aten.view.1120(%1144) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %1146 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1147 = call @aten.expand.758(%1146) : (tensor<f32>) -> tensor<480xf32>
    %1148 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1149 = call @aten.expand.758(%1148) : (tensor<f32>) -> tensor<480xf32>
    %1150 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1151 = call @aten.expand.758(%1150) : (tensor<f32>) -> tensor<480xf32>
    %1152 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1153 = call @aten.expand.758(%1152) : (tensor<f32>) -> tensor<480xf32>
    %1154 = call @aten.native_batch_norm.1124(%1145, %1147, %1149, %1151, %1153) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %1155 = "mhlo.get_tuple_element"(%1154) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1156 = "mhlo.get_tuple_element"(%1154) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %1157 = call @aten.view.1144(%1156) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %1158 = call @aten.mul.1148(%1157, %arg430) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %1159 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1160 = call @aten.mul.1154(%1158, %1159) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %1161 = call @aten.add.1160(%arg429, %1160) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1162 = call @aten.view.1169(%1161) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1163 = call @aten.permute.1356(%arg440) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1164 = call @aten.addmm.1361(%1162, %1163, %arg439) : (tensor<480x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>) -> tensor<480x3072xf32>
    %1165 = call @aten.view.1372(%1164) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1166 = call @aten.gelu.1376(%1165) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1167 = call @aten.view.1450(%1166) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1168 = call @aten.permute.1352(%arg444) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1169 = call @aten.addmm.1454(%1167, %1168, %arg443) : (tensor<480x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>) -> tensor<480x768xf32>
    %1170 = call @aten.view.1090(%1169) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1171 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1172 = call @aten.expand.772(%1171) : (tensor<f32>) -> tensor<2x240x768xf32>
    %1173 = call @aten.mul.1094(%1161, %1172) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1174 = call @aten.add.1107(%1170, %1173) : (tensor<2x240x768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1175 = call @aten.view.1120(%1174) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    %1176 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1177 = call @aten.expand.758(%1176) : (tensor<f32>) -> tensor<480xf32>
    %1178 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1179 = call @aten.expand.758(%1178) : (tensor<f32>) -> tensor<480xf32>
    %1180 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1181 = call @aten.expand.758(%1180) : (tensor<f32>) -> tensor<480xf32>
    %1182 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1183 = call @aten.expand.758(%1182) : (tensor<f32>) -> tensor<480xf32>
    %1184 = call @aten.native_batch_norm.1124(%1175, %1177, %1179, %1181, %1183) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    %1185 = "mhlo.get_tuple_element"(%1184) {index = 2 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1186 = call @aten.convolution_overrideable.2401(%arg738, %arg20) : (tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>) -> tensor<16x64x128x128xf32>
    %1187 = call @aten.native_batch_norm.2406(%1186, %arg19, %arg18, %arg576, %arg577) {xla_shape = "(f32[16,64,128,128]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %1188 = "mhlo.get_tuple_element"(%1187) {index = 0 : i32} : (tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x128x128xf32>
    %1189 = call @aten.relu.2426(%1188) : (tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32>
    %1190 = call @aten.max_pool2d.2501(%1189) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, u32[16,64,64,64]{3,2,1,0})"} : (tensor<16x64x128x128xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<16x64x64x64xui32>>
    %1191 = "mhlo.get_tuple_element"(%1190) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<16x64x64x64xui32>>) -> tensor<16x64x64x64xui32>
    %1192 = "mhlo.get_tuple_element"(%1190) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<16x64x64x64xui32>>) -> tensor<16x64x64x64xf32>
    %1193 = call @aten.convolution_overrideable.2571(%1192, %arg32) : (tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tensor<16x256x64x64xf32>
    %1194 = call @aten.native_batch_norm.2576(%1193, %arg31, %arg30, %arg588, %arg589) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1195 = "mhlo.get_tuple_element"(%1194) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x64x64xf32>
    %1196 = call @aten.convolution_overrideable.2529(%1192, %arg23) : (tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>) -> tensor<16x64x64x64xf32>
    %1197 = call @aten.native_batch_norm.2534(%1196, %arg22, %arg21, %arg579, %arg580) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %1198 = "mhlo.get_tuple_element"(%1197) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %1199 = call @aten.relu.2554(%1198) : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %1200 = call @aten.convolution_overrideable.2560(%1199, %arg26) : (tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>) -> tensor<16x64x64x64xf32>
    %1201 = call @aten.native_batch_norm.2534(%1200, %arg25, %arg24, %arg582, %arg583) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %1202 = "mhlo.get_tuple_element"(%1201) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %1203 = call @aten.relu.2554(%1202) : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %1204 = call @aten.convolution_overrideable.2571(%1203, %arg29) : (tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tensor<16x256x64x64xf32>
    %1205 = call @aten.native_batch_norm.2576(%1204, %arg28, %arg27, %arg585, %arg586) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1206 = "mhlo.get_tuple_element"(%1205) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x64x64xf32>
    %1207 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1208 = call @aten.expand.2390(%1207) : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %1209 = call @aten.mul.2596(%1206, %1208) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1210 = call @aten.add.2607(%1195, %1209) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1211 = call @aten.relu.2612(%1210) : (tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1212 = call @aten.convolution_overrideable.2618(%1211, %arg35) : (tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>) -> tensor<16x64x64x64xf32>
    %1213 = call @aten.native_batch_norm.2534(%1212, %arg34, %arg33, %arg591, %arg592) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %1214 = "mhlo.get_tuple_element"(%1213) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %1215 = call @aten.relu.2554(%1214) : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %1216 = call @aten.convolution_overrideable.2560(%1215, %arg38) : (tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>) -> tensor<16x64x64x64xf32>
    %1217 = call @aten.native_batch_norm.2534(%1216, %arg37, %arg36, %arg594, %arg595) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %1218 = "mhlo.get_tuple_element"(%1217) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %1219 = call @aten.relu.2554(%1218) : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %1220 = call @aten.convolution_overrideable.2571(%1219, %arg41) : (tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tensor<16x256x64x64xf32>
    %1221 = call @aten.native_batch_norm.2576(%1220, %arg40, %arg39, %arg597, %arg598) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1222 = "mhlo.get_tuple_element"(%1221) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x64x64xf32>
    %1223 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1224 = call @aten.expand.2390(%1223) : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %1225 = call @aten.mul.2596(%1222, %1224) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1226 = call @aten.add.2607(%1211, %1225) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1227 = call @aten.relu.2612(%1226) : (tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1228 = call @aten.convolution_overrideable.2618(%1227, %arg44) : (tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>) -> tensor<16x64x64x64xf32>
    %1229 = call @aten.native_batch_norm.2534(%1228, %arg43, %arg42, %arg600, %arg601) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %1230 = "mhlo.get_tuple_element"(%1229) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %1231 = call @aten.relu.2554(%1230) : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %1232 = call @aten.convolution_overrideable.2560(%1231, %arg47) : (tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>) -> tensor<16x64x64x64xf32>
    %1233 = call @aten.native_batch_norm.2534(%1232, %arg46, %arg45, %arg603, %arg604) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %1234 = "mhlo.get_tuple_element"(%1233) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %1235 = call @aten.relu.2554(%1234) : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %1236 = call @aten.convolution_overrideable.2571(%1235, %arg50) : (tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tensor<16x256x64x64xf32>
    %1237 = call @aten.native_batch_norm.2576(%1236, %arg49, %arg48, %arg606, %arg607) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1238 = "mhlo.get_tuple_element"(%1237) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x64x64xf32>
    %1239 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1240 = call @aten.expand.2390(%1239) : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %1241 = call @aten.mul.2596(%1238, %1240) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1242 = call @aten.add.2607(%1227, %1241) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1243 = call @aten.relu.2612(%1242) : (tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %1244 = call @aten.convolution_overrideable.2760(%1243, %arg62) : (tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>) -> tensor<16x512x32x32xf32>
    %1245 = call @aten.native_batch_norm.2735(%1244, %arg61, %arg60, %arg618, %arg619) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1246 = "mhlo.get_tuple_element"(%1245) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %1247 = call @aten.convolution_overrideable.2668(%1243, %arg53) : (tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>) -> tensor<16x128x64x64xf32>
    %1248 = call @aten.native_batch_norm.2673(%1247, %arg52, %arg51, %arg609, %arg610) {xla_shape = "(f32[16,128,64,64]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %1249 = "mhlo.get_tuple_element"(%1248) {index = 0 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x64x64xf32>
    %1250 = call @aten.relu.2693(%1249) : (tensor<16x128x64x64xf32>) -> tensor<16x128x64x64xf32>
    %1251 = call @aten.convolution_overrideable.2699(%1250, %arg56) : (tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>) -> tensor<16x128x32x32xf32>
    %1252 = call @aten.native_batch_norm.2704(%1251, %arg55, %arg54, %arg612, %arg613) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %1253 = "mhlo.get_tuple_element"(%1252) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %1254 = call @aten.relu.2724(%1253) : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %1255 = call @aten.convolution_overrideable.2730(%1254, %arg59) : (tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tensor<16x512x32x32xf32>
    %1256 = call @aten.native_batch_norm.2735(%1255, %arg58, %arg57, %arg615, %arg616) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1257 = "mhlo.get_tuple_element"(%1256) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %1258 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1259 = call @aten.expand.2376(%1258) : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %1260 = call @aten.mul.2755(%1257, %1259) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1261 = call @aten.add.2770(%1246, %1260) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1262 = call @aten.relu.2775(%1261) : (tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1263 = call @aten.convolution_overrideable.2781(%1262, %arg65) : (tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>) -> tensor<16x128x32x32xf32>
    %1264 = call @aten.native_batch_norm.2704(%1263, %arg64, %arg63, %arg621, %arg622) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %1265 = "mhlo.get_tuple_element"(%1264) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %1266 = call @aten.relu.2724(%1265) : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %1267 = call @aten.convolution_overrideable.2792(%1266, %arg68) : (tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>) -> tensor<16x128x32x32xf32>
    %1268 = call @aten.native_batch_norm.2704(%1267, %arg67, %arg66, %arg624, %arg625) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %1269 = "mhlo.get_tuple_element"(%1268) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %1270 = call @aten.relu.2724(%1269) : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %1271 = call @aten.convolution_overrideable.2730(%1270, %arg71) : (tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tensor<16x512x32x32xf32>
    %1272 = call @aten.native_batch_norm.2735(%1271, %arg70, %arg69, %arg627, %arg628) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1273 = "mhlo.get_tuple_element"(%1272) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %1274 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1275 = call @aten.expand.2376(%1274) : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %1276 = call @aten.mul.2755(%1273, %1275) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1277 = call @aten.add.2770(%1262, %1276) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1278 = call @aten.relu.2775(%1277) : (tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1279 = call @aten.convolution_overrideable.2781(%1278, %arg74) : (tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>) -> tensor<16x128x32x32xf32>
    %1280 = call @aten.native_batch_norm.2704(%1279, %arg73, %arg72, %arg630, %arg631) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %1281 = "mhlo.get_tuple_element"(%1280) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %1282 = call @aten.relu.2724(%1281) : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %1283 = call @aten.convolution_overrideable.2792(%1282, %arg77) : (tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>) -> tensor<16x128x32x32xf32>
    %1284 = call @aten.native_batch_norm.2704(%1283, %arg76, %arg75, %arg633, %arg634) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %1285 = "mhlo.get_tuple_element"(%1284) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %1286 = call @aten.relu.2724(%1285) : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %1287 = call @aten.convolution_overrideable.2730(%1286, %arg80) : (tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tensor<16x512x32x32xf32>
    %1288 = call @aten.native_batch_norm.2735(%1287, %arg79, %arg78, %arg636, %arg637) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1289 = "mhlo.get_tuple_element"(%1288) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %1290 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1291 = call @aten.expand.2376(%1290) : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %1292 = call @aten.mul.2755(%1289, %1291) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1293 = call @aten.add.2770(%1278, %1292) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1294 = call @aten.relu.2775(%1293) : (tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1295 = call @aten.convolution_overrideable.2781(%1294, %arg83) : (tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>) -> tensor<16x128x32x32xf32>
    %1296 = call @aten.native_batch_norm.2704(%1295, %arg82, %arg81, %arg639, %arg640) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %1297 = "mhlo.get_tuple_element"(%1296) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %1298 = call @aten.relu.2724(%1297) : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %1299 = call @aten.convolution_overrideable.2792(%1298, %arg86) : (tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>) -> tensor<16x128x32x32xf32>
    %1300 = call @aten.native_batch_norm.2704(%1299, %arg85, %arg84, %arg642, %arg643) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %1301 = "mhlo.get_tuple_element"(%1300) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %1302 = call @aten.relu.2724(%1301) : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %1303 = call @aten.convolution_overrideable.2730(%1302, %arg89) : (tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tensor<16x512x32x32xf32>
    %1304 = call @aten.native_batch_norm.2735(%1303, %arg88, %arg87, %arg645, %arg646) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1305 = "mhlo.get_tuple_element"(%1304) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %1306 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1307 = call @aten.expand.2376(%1306) : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %1308 = call @aten.mul.2755(%1305, %1307) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1309 = call @aten.add.2770(%1294, %1308) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1310 = call @aten.relu.2775(%1309) : (tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %1311 = call @aten.convolution_overrideable.2950(%1310, %arg101) : (tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>) -> tensor<16x1024x16x16xf32>
    %1312 = call @aten.native_batch_norm.2925(%1311, %arg100, %arg99, %arg657, %arg658) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %1313 = "mhlo.get_tuple_element"(%1312) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %1314 = call @aten.convolution_overrideable.2858(%1310, %arg92) : (tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>) -> tensor<16x256x32x32xf32>
    %1315 = call @aten.native_batch_norm.2863(%1314, %arg91, %arg90, %arg648, %arg649) {xla_shape = "(f32[16,256,32,32]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1316 = "mhlo.get_tuple_element"(%1315) {index = 0 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x32x32xf32>
    %1317 = call @aten.relu.2883(%1316) : (tensor<16x256x32x32xf32>) -> tensor<16x256x32x32xf32>
    %1318 = call @aten.convolution_overrideable.2889(%1317, %arg95) : (tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32>
    %1319 = call @aten.native_batch_norm.2894(%1318, %arg94, %arg93, %arg651, %arg652) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1320 = "mhlo.get_tuple_element"(%1319) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1321 = call @aten.relu.2914(%1320) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1322 = call @aten.convolution_overrideable.2920(%1321, %arg98) : (tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tensor<16x1024x16x16xf32>
    %1323 = call @aten.native_batch_norm.2925(%1322, %arg97, %arg96, %arg654, %arg655) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %1324 = "mhlo.get_tuple_element"(%1323) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %1325 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1326 = call @aten.expand.2358(%1325) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %1327 = call @aten.mul.2945(%1324, %1326) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1328 = call @aten.add.2960(%1313, %1327) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1329 = call @aten.relu.2965(%1328) : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1330 = call @aten.convolution_overrideable.2971(%1329, %arg104) : (tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tensor<16x256x16x16xf32>
    %1331 = call @aten.native_batch_norm.2894(%1330, %arg103, %arg102, %arg660, %arg661) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1332 = "mhlo.get_tuple_element"(%1331) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1333 = call @aten.relu.2914(%1332) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1334 = call @aten.convolution_overrideable.2982(%1333, %arg107) : (tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32>
    %1335 = call @aten.native_batch_norm.2894(%1334, %arg106, %arg105, %arg663, %arg664) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1336 = "mhlo.get_tuple_element"(%1335) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1337 = call @aten.relu.2914(%1336) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1338 = call @aten.convolution_overrideable.2920(%1337, %arg110) : (tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tensor<16x1024x16x16xf32>
    %1339 = call @aten.native_batch_norm.2925(%1338, %arg109, %arg108, %arg666, %arg667) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %1340 = "mhlo.get_tuple_element"(%1339) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %1341 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1342 = call @aten.expand.2358(%1341) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %1343 = call @aten.mul.2945(%1340, %1342) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1344 = call @aten.add.2960(%1329, %1343) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1345 = call @aten.relu.2965(%1344) : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1346 = call @aten.convolution_overrideable.2971(%1345, %arg113) : (tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tensor<16x256x16x16xf32>
    %1347 = call @aten.native_batch_norm.2894(%1346, %arg112, %arg111, %arg669, %arg670) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1348 = "mhlo.get_tuple_element"(%1347) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1349 = call @aten.relu.2914(%1348) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1350 = call @aten.convolution_overrideable.2982(%1349, %arg116) : (tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32>
    %1351 = call @aten.native_batch_norm.2894(%1350, %arg115, %arg114, %arg672, %arg673) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1352 = "mhlo.get_tuple_element"(%1351) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1353 = call @aten.relu.2914(%1352) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1354 = call @aten.convolution_overrideable.2920(%1353, %arg119) : (tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tensor<16x1024x16x16xf32>
    %1355 = call @aten.native_batch_norm.2925(%1354, %arg118, %arg117, %arg675, %arg676) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %1356 = "mhlo.get_tuple_element"(%1355) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %1357 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1358 = call @aten.expand.2358(%1357) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %1359 = call @aten.mul.2945(%1356, %1358) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1360 = call @aten.add.2960(%1345, %1359) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1361 = call @aten.relu.2965(%1360) : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1362 = call @aten.convolution_overrideable.2971(%1361, %arg122) : (tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tensor<16x256x16x16xf32>
    %1363 = call @aten.native_batch_norm.2894(%1362, %arg121, %arg120, %arg678, %arg679) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1364 = "mhlo.get_tuple_element"(%1363) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1365 = call @aten.relu.2914(%1364) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1366 = call @aten.convolution_overrideable.2982(%1365, %arg125) : (tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32>
    %1367 = call @aten.native_batch_norm.2894(%1366, %arg124, %arg123, %arg681, %arg682) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1368 = "mhlo.get_tuple_element"(%1367) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1369 = call @aten.relu.2914(%1368) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1370 = call @aten.convolution_overrideable.2920(%1369, %arg128) : (tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tensor<16x1024x16x16xf32>
    %1371 = call @aten.native_batch_norm.2925(%1370, %arg127, %arg126, %arg684, %arg685) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %1372 = "mhlo.get_tuple_element"(%1371) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %1373 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1374 = call @aten.expand.2358(%1373) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %1375 = call @aten.mul.2945(%1372, %1374) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1376 = call @aten.add.2960(%1361, %1375) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1377 = call @aten.relu.2965(%1376) : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1378 = call @aten.convolution_overrideable.2971(%1377, %arg131) : (tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tensor<16x256x16x16xf32>
    %1379 = call @aten.native_batch_norm.2894(%1378, %arg130, %arg129, %arg687, %arg688) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1380 = "mhlo.get_tuple_element"(%1379) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1381 = call @aten.relu.2914(%1380) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1382 = call @aten.convolution_overrideable.2982(%1381, %arg134) : (tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32>
    %1383 = call @aten.native_batch_norm.2894(%1382, %arg133, %arg132, %arg690, %arg691) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1384 = "mhlo.get_tuple_element"(%1383) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1385 = call @aten.relu.2914(%1384) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1386 = call @aten.convolution_overrideable.2920(%1385, %arg137) : (tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tensor<16x1024x16x16xf32>
    %1387 = call @aten.native_batch_norm.2925(%1386, %arg136, %arg135, %arg693, %arg694) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %1388 = "mhlo.get_tuple_element"(%1387) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %1389 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1390 = call @aten.expand.2358(%1389) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %1391 = call @aten.mul.2945(%1388, %1390) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1392 = call @aten.add.2960(%1377, %1391) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1393 = call @aten.relu.2965(%1392) : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1394 = call @aten.convolution_overrideable.2971(%1393, %arg140) : (tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tensor<16x256x16x16xf32>
    %1395 = call @aten.native_batch_norm.2894(%1394, %arg139, %arg138, %arg696, %arg697) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1396 = "mhlo.get_tuple_element"(%1395) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1397 = call @aten.relu.2914(%1396) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1398 = call @aten.convolution_overrideable.2982(%1397, %arg143) : (tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32>
    %1399 = call @aten.native_batch_norm.2894(%1398, %arg142, %arg141, %arg699, %arg700) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %1400 = "mhlo.get_tuple_element"(%1399) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %1401 = call @aten.relu.2914(%1400) : (tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %1402 = call @aten.convolution_overrideable.2920(%1401, %arg146) : (tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tensor<16x1024x16x16xf32>
    %1403 = call @aten.native_batch_norm.2925(%1402, %arg145, %arg144, %arg702, %arg703) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %1404 = "mhlo.get_tuple_element"(%1403) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %1405 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1406 = call @aten.expand.2358(%1405) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %1407 = call @aten.mul.2945(%1404, %1406) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1408 = call @aten.add.2960(%1393, %1407) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1409 = call @aten.relu.2965(%1408) : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %1410 = call @aten.convolution_overrideable.3186(%1409, %arg158) : (tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>) -> tensor<16x2048x8x8xf32>
    %1411 = call @aten.native_batch_norm.3161(%1410, %arg157, %arg156, %arg714, %arg715) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>
    %1412 = "mhlo.get_tuple_element"(%1411) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<16x2048x8x8xf32>
    %1413 = call @aten.convolution_overrideable.3094(%1409, %arg149) : (tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>) -> tensor<16x512x16x16xf32>
    %1414 = call @aten.native_batch_norm.3099(%1413, %arg148, %arg147, %arg705, %arg706) {xla_shape = "(f32[16,512,16,16]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1415 = "mhlo.get_tuple_element"(%1414) {index = 0 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x16x16xf32>
    %1416 = call @aten.relu.3119(%1415) : (tensor<16x512x16x16xf32>) -> tensor<16x512x16x16xf32>
    %1417 = call @aten.convolution_overrideable.3125(%1416, %arg152) : (tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>) -> tensor<16x512x8x8xf32>
    %1418 = call @aten.native_batch_norm.3130(%1417, %arg151, %arg150, %arg708, %arg709) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1419 = "mhlo.get_tuple_element"(%1418) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %1420 = call @aten.relu.3150(%1419) : (tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %1421 = call @aten.convolution_overrideable.3156(%1420, %arg155) : (tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>) -> tensor<16x2048x8x8xf32>
    %1422 = call @aten.native_batch_norm.3161(%1421, %arg154, %arg153, %arg711, %arg712) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>
    %1423 = "mhlo.get_tuple_element"(%1422) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<16x2048x8x8xf32>
    %1424 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1425 = call @aten.expand.2346(%1424) : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    %1426 = call @aten.mul.3181(%1423, %1425) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1427 = call @aten.add.3196(%1412, %1426) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1428 = call @aten.relu.3201(%1427) : (tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1429 = call @aten.convolution_overrideable.3207(%1428, %arg161) : (tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>) -> tensor<16x512x8x8xf32>
    %1430 = call @aten.native_batch_norm.3130(%1429, %arg160, %arg159, %arg717, %arg718) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1431 = "mhlo.get_tuple_element"(%1430) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %1432 = call @aten.relu.3150(%1431) : (tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %1433 = call @aten.convolution_overrideable.3218(%1432, %arg164) : (tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>) -> tensor<16x512x8x8xf32>
    %1434 = call @aten.native_batch_norm.3130(%1433, %arg163, %arg162, %arg720, %arg721) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1435 = "mhlo.get_tuple_element"(%1434) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %1436 = call @aten.relu.3150(%1435) : (tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %1437 = call @aten.convolution_overrideable.3156(%1436, %arg167) : (tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>) -> tensor<16x2048x8x8xf32>
    %1438 = call @aten.native_batch_norm.3161(%1437, %arg166, %arg165, %arg723, %arg724) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>
    %1439 = "mhlo.get_tuple_element"(%1438) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<16x2048x8x8xf32>
    %1440 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1441 = call @aten.expand.2346(%1440) : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    %1442 = call @aten.mul.3181(%1439, %1441) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1443 = call @aten.add.3196(%1428, %1442) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1444 = call @aten.relu.3201(%1443) : (tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1445 = call @aten.convolution_overrideable.3207(%1444, %arg170) : (tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>) -> tensor<16x512x8x8xf32>
    %1446 = call @aten.native_batch_norm.3130(%1445, %arg169, %arg168, %arg726, %arg727) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1447 = "mhlo.get_tuple_element"(%1446) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %1448 = call @aten.relu.3150(%1447) : (tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %1449 = call @aten.convolution_overrideable.3218(%1448, %arg173) : (tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>) -> tensor<16x512x8x8xf32>
    %1450 = call @aten.native_batch_norm.3130(%1449, %arg172, %arg171, %arg729, %arg730) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %1451 = "mhlo.get_tuple_element"(%1450) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %1452 = call @aten.relu.3150(%1451) : (tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %1453 = call @aten.convolution_overrideable.3156(%1452, %arg176) : (tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>) -> tensor<16x2048x8x8xf32>
    %1454 = call @aten.native_batch_norm.3161(%1453, %arg175, %arg174, %arg732, %arg733) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>
    %1455 = "mhlo.get_tuple_element"(%1454) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<16x2048x8x8xf32>
    %1456 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1457 = call @aten.expand.2346(%1456) : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    %1458 = call @aten.mul.3181(%1455, %1457) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1459 = call @aten.add.3196(%1444, %1458) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1460 = call @aten.relu.3201(%1459) : (tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %1461 = call @aten.view.3261(%1460) : (tensor<16x2048x8x8xf32>) -> tensor<2x8x2048x8x8xf32>
    %1462 = call @aten.permute.3265(%1461) {xla_shape = "f32[2,2048,8,8,8]{4,3,1,2,0}"} : (tensor<2x8x2048x8x8xf32>) -> tensor<2x2048x8x8x8xf32>
    %1463 = call @aten.mean.3273(%1462) : (tensor<2x2048x8x8x8xf32>) -> tensor<2x2048xf32>
    %1464 = "mhlo.get_tuple_element"(%1184) {index = 0 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<1x480x768xf32>
    %1465 = call @aten.view.1144(%1464) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    %1466 = call @aten.mul.1148(%1465, %arg442) : (tensor<2x240x768xf32>, tensor<768xf32>) -> tensor<2x240x768xf32>
    %1467 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1468 = call @aten.mul.1154(%1466, %1467) : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<2x240x768xf32>
    %1469 = call @aten.add.1160(%arg441, %1468) : (tensor<768xf32>, tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1470 = "mhlo.slice"(%1469) {limit_indices = dense<[2, 240, 768]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x240x768xf32>) -> tensor<2x240x768xf32>
    %1471 = "mhlo.slice"(%1470) {limit_indices = dense<[2, 1, 768]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x240x768xf32>) -> tensor<2x1x768xf32>
    %1472 = call @aten.view.2326(%1471) : (tensor<2x1x768xf32>) -> tensor<2x768xf32>
    %1473 = call @aten.permute.752(%arg574) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1474 = call @aten.addmm.2330(%1472, %1473, %arg573) : (tensor<2x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<2x768xf32>
    %1475 = call @aten.tanh.2341(%1474) : (tensor<2x768xf32>) -> tensor<2x768xf32>
    %1476 = call @aten.cat.3289(%1463, %1475) : (tensor<2x2048xf32>, tensor<2x768xf32>) -> tensor<2x2816xf32>
    %1477 = call @aten.permute.748(%arg9) {xla_shape = "f32[2816,256]{0,1}"} : (tensor<256x2816xf32>) -> tensor<2816x256xf32>
    %1478 = call @aten.addmm.3294(%1476, %1477, %arg8) : (tensor<2x2816xf32>, tensor<2816x256xf32>, tensor<256xf32>) -> tensor<2x256xf32>
    %1479 = call @aten.relu.3305(%1478) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1480 = call @aten.permute.744(%arg1) {xla_shape = "f32[256,19]{0,1}"} : (tensor<19x256xf32>) -> tensor<256x19xf32>
    %1481 = call @aten.addmm.3311(%1479, %1480, %arg0) : (tensor<2x256xf32>, tensor<256x19xf32>, tensor<19xf32>) -> tensor<2x19xf32>
    %1482 = call @aten.permute.3326(%arg11) {xla_shape = "f32[2816,128]{0,1}"} : (tensor<128x2816xf32>) -> tensor<2816x128xf32>
    %1483 = call @aten.addmm.3330(%1476, %1482, %arg10) : (tensor<2x2816xf32>, tensor<2816x128xf32>, tensor<128xf32>) -> tensor<2x128xf32>
    %1484 = call @aten.relu.3341(%1483) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %1485 = call @aten.permute.3322(%arg3) {xla_shape = "f32[128,19]{0,1}"} : (tensor<19x128xf32>) -> tensor<128x19xf32>
    %1486 = call @aten.addmm.3347(%1484, %1485, %arg2) : (tensor<2x128xf32>, tensor<128x19xf32>, tensor<19xf32>) -> tensor<2x19xf32>
    %1487 = call @aten.permute.3362(%arg13) {xla_shape = "f32[2048,256]{0,1}"} : (tensor<256x2048xf32>) -> tensor<2048x256xf32>
    %1488 = call @aten.addmm.3366(%1463, %1487, %arg12) : (tensor<2x2048xf32>, tensor<2048x256xf32>, tensor<256xf32>) -> tensor<2x256xf32>
    %1489 = call @aten.relu.3305(%1488) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1490 = call @aten.permute.3358(%arg5) {xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    %1491 = call @aten.addmm.3378(%1489, %1490, %arg4) : (tensor<2x256xf32>, tensor<256x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    %1492 = call @aten.permute.3390(%arg15) {xla_shape = "f32[768,256]{0,1}"} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    %1493 = call @aten.addmm.3394(%1475, %1492, %arg14) : (tensor<2x768xf32>, tensor<768x256xf32>, tensor<256xf32>) -> tensor<2x256xf32>
    %1494 = call @aten.relu.3305(%1493) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1495 = call @aten.permute.3358(%arg7) {xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    %1496 = call @aten.addmm.3378(%1494, %1495, %arg6) : (tensor<2x256xf32>, tensor<256x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    %1497 = "mhlo.slice"(%arg742) {limit_indices = dense<[1, 19]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x19xi64>) -> tensor<1x19xi64>
    %1498 = call @aten.view.3408(%1497) : (tensor<1x19xi64>) -> tensor<19xi64>
    %1499 = "mhlo.slice"(%arg742) {limit_indices = dense<[1, 19]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x19xi64>) -> tensor<1x19xi64>
    %1500 = call @aten.view.3408(%1499) : (tensor<1x19xi64>) -> tensor<19xi64>
    %1501 = mhlo.constant dense<1> : tensor<i64>
    %1502 = call @aten.ge.3415(%1500, %1501) : (tensor<19xi64>, tensor<i64>) -> tensor<19xi1>
    %1503 = "mhlo.convert"(%1502) : (tensor<19xi1>) -> tensor<19xi64>
    %1504 = "mhlo.get_tuple_element"(%1454) {index = 1 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1505 = "mhlo.get_tuple_element"(%1454) {index = 3 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1506 = "mhlo.get_tuple_element"(%1154) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1507 = call @aten.view.3422(%1506) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1508 = "mhlo.get_tuple_element"(%1154) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1509 = call @aten.view.3422(%1508) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1510 = "mhlo.get_tuple_element"(%1387) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1511 = "mhlo.get_tuple_element"(%1387) {index = 3 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1512 = "mhlo.get_tuple_element"(%1395) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1513 = "mhlo.get_tuple_element"(%1395) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1514 = "mhlo.get_tuple_element"(%1059) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1515 = call @aten.view.3422(%1514) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1516 = "mhlo.get_tuple_element"(%1059) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1517 = call @aten.view.3422(%1516) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1518 = "mhlo.get_tuple_element"(%1399) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1519 = "mhlo.get_tuple_element"(%1399) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1520 = "mhlo.get_tuple_element"(%964) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1521 = call @aten.view.3422(%1520) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1522 = "mhlo.get_tuple_element"(%964) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1523 = call @aten.view.3422(%1522) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1524 = "mhlo.get_tuple_element"(%1422) {index = 1 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1525 = "mhlo.get_tuple_element"(%1422) {index = 3 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1526 = "mhlo.get_tuple_element"(%1430) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1527 = "mhlo.get_tuple_element"(%1430) {index = 3 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1528 = "mhlo.get_tuple_element"(%994) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1529 = call @aten.view.3422(%1528) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1530 = "mhlo.get_tuple_element"(%994) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1531 = call @aten.view.3422(%1530) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1532 = "mhlo.get_tuple_element"(%1434) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1533 = "mhlo.get_tuple_element"(%1434) {index = 3 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1534 = "mhlo.get_tuple_element"(%299) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1535 = call @aten.view.3422(%1534) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1536 = "mhlo.get_tuple_element"(%299) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1537 = call @aten.view.3422(%1536) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1538 = "mhlo.get_tuple_element"(%329) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1539 = call @aten.view.3422(%1538) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1540 = "mhlo.get_tuple_element"(%329) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1541 = call @aten.view.3422(%1540) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1542 = "mhlo.get_tuple_element"(%804) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1543 = call @aten.view.3422(%1542) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1544 = "mhlo.get_tuple_element"(%804) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1545 = call @aten.view.3422(%1544) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1546 = "mhlo.get_tuple_element"(%899) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1547 = call @aten.view.3422(%1546) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1548 = "mhlo.get_tuple_element"(%899) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1549 = call @aten.view.3422(%1548) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1550 = "mhlo.get_tuple_element"(%1323) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1551 = "mhlo.get_tuple_element"(%1323) {index = 3 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1552 = "mhlo.get_tuple_element"(%1296) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1553 = "mhlo.get_tuple_element"(%1296) {index = 3 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1554 = "mhlo.get_tuple_element"(%1331) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1555 = "mhlo.get_tuple_element"(%1331) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1556 = "mhlo.get_tuple_element"(%1300) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1557 = "mhlo.get_tuple_element"(%1300) {index = 3 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1558 = "mhlo.get_tuple_element"(%1335) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1559 = "mhlo.get_tuple_element"(%1335) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1560 = "mhlo.get_tuple_element"(%1304) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1561 = "mhlo.get_tuple_element"(%1304) {index = 3 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1562 = "mhlo.get_tuple_element"(%1355) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1563 = "mhlo.get_tuple_element"(%1355) {index = 3 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1564 = "mhlo.get_tuple_element"(%1264) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1565 = "mhlo.get_tuple_element"(%1264) {index = 3 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1566 = "mhlo.get_tuple_element"(%1363) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1567 = "mhlo.get_tuple_element"(%1363) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1568 = "mhlo.get_tuple_element"(%1268) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1569 = "mhlo.get_tuple_element"(%1268) {index = 3 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1570 = "mhlo.get_tuple_element"(%1367) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1571 = "mhlo.get_tuple_element"(%1367) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1572 = "mhlo.get_tuple_element"(%584) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1573 = call @aten.view.3422(%1572) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1574 = "mhlo.get_tuple_element"(%1272) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1575 = "mhlo.get_tuple_element"(%584) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1576 = call @aten.view.3422(%1575) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1577 = "mhlo.get_tuple_element"(%1272) {index = 3 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1578 = "mhlo.get_tuple_element"(%234) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1579 = call @aten.view.3422(%1578) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1580 = "mhlo.get_tuple_element"(%234) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1581 = call @aten.view.3422(%1580) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1582 = "mhlo.get_tuple_element"(%1237) {index = 1 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1583 = "mhlo.get_tuple_element"(%1233) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %1584 = "mhlo.get_tuple_element"(%1233) {index = 3 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %1585 = "mhlo.get_tuple_element"(%1237) {index = 3 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1586 = "mhlo.get_tuple_element"(%1245) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1587 = "mhlo.get_tuple_element"(%1245) {index = 3 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1588 = "mhlo.get_tuple_element"(%489) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1589 = call @aten.view.3422(%1588) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1590 = "mhlo.get_tuple_element"(%489) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1591 = call @aten.view.3422(%1590) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1592 = "mhlo.get_tuple_element"(%519) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1593 = call @aten.view.3422(%1592) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1594 = "mhlo.get_tuple_element"(%519) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %1595 = call @aten.view.3422(%1594) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %1596 = "mhlo.get_tuple_element"(%1383) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1597 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1598 = "mhlo.broadcast_in_dim"(%1597) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1599 = mhlo.multiply %1596, %1598 : tensor<256xf32>
    %1600 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1601 = mhlo.subtract %1600, %1597 : tensor<f32>
    %1602 = "mhlo.broadcast_in_dim"(%1601) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1603 = mhlo.multiply %arg691, %1602 : tensor<256xf32>
    %1604 = mhlo.add %1599, %1603 : tensor<256xf32>
    %1605 = "mhlo.get_tuple_element"(%1446) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1606 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1607 = "mhlo.broadcast_in_dim"(%1606) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1608 = mhlo.multiply %1605, %1607 : tensor<512xf32>
    %1609 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1610 = mhlo.subtract %1609, %1606 : tensor<f32>
    %1611 = "mhlo.broadcast_in_dim"(%1610) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1612 = mhlo.multiply %arg726, %1611 : tensor<512xf32>
    %1613 = mhlo.add %1608, %1612 : tensor<512xf32>
    %1614 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1615 = "mhlo.broadcast_in_dim"(%1614) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1616 = mhlo.multiply %1510, %1615 : tensor<1024xf32>
    %1617 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1618 = mhlo.subtract %1617, %1614 : tensor<f32>
    %1619 = "mhlo.broadcast_in_dim"(%1618) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1620 = mhlo.multiply %arg693, %1619 : tensor<1024xf32>
    %1621 = mhlo.add %1616, %1620 : tensor<1024xf32>
    %1622 = "mhlo.get_tuple_element"(%1387) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1623 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1624 = "mhlo.broadcast_in_dim"(%1623) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1625 = mhlo.multiply %1622, %1624 : tensor<1024xf32>
    %1626 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1627 = mhlo.subtract %1626, %1623 : tensor<f32>
    %1628 = "mhlo.broadcast_in_dim"(%1627) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1629 = mhlo.multiply %arg694, %1628 : tensor<1024xf32>
    %1630 = mhlo.add %1625, %1629 : tensor<1024xf32>
    %1631 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1632 = "mhlo.broadcast_in_dim"(%1631) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1633 = mhlo.multiply %1512, %1632 : tensor<256xf32>
    %1634 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1635 = mhlo.subtract %1634, %1631 : tensor<f32>
    %1636 = "mhlo.broadcast_in_dim"(%1635) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1637 = mhlo.multiply %arg696, %1636 : tensor<256xf32>
    %1638 = mhlo.add %1633, %1637 : tensor<256xf32>
    %1639 = "mhlo.get_tuple_element"(%1395) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1640 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1641 = "mhlo.broadcast_in_dim"(%1640) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1642 = mhlo.multiply %1639, %1641 : tensor<256xf32>
    %1643 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1644 = mhlo.subtract %1643, %1640 : tensor<f32>
    %1645 = "mhlo.broadcast_in_dim"(%1644) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1646 = mhlo.multiply %arg697, %1645 : tensor<256xf32>
    %1647 = mhlo.add %1642, %1646 : tensor<256xf32>
    %1648 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1649 = "mhlo.broadcast_in_dim"(%1648) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1650 = mhlo.multiply %1518, %1649 : tensor<256xf32>
    %1651 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1652 = mhlo.subtract %1651, %1648 : tensor<f32>
    %1653 = "mhlo.broadcast_in_dim"(%1652) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1654 = mhlo.multiply %arg699, %1653 : tensor<256xf32>
    %1655 = mhlo.add %1650, %1654 : tensor<256xf32>
    %1656 = "mhlo.get_tuple_element"(%1399) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1657 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1658 = "mhlo.broadcast_in_dim"(%1657) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1659 = mhlo.multiply %1656, %1658 : tensor<256xf32>
    %1660 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1661 = mhlo.subtract %1660, %1657 : tensor<f32>
    %1662 = "mhlo.broadcast_in_dim"(%1661) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1663 = mhlo.multiply %arg700, %1662 : tensor<256xf32>
    %1664 = mhlo.add %1659, %1663 : tensor<256xf32>
    %1665 = "mhlo.get_tuple_element"(%1403) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1666 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1667 = "mhlo.broadcast_in_dim"(%1666) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1668 = mhlo.multiply %1665, %1667 : tensor<1024xf32>
    %1669 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1670 = mhlo.subtract %1669, %1666 : tensor<f32>
    %1671 = "mhlo.broadcast_in_dim"(%1670) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1672 = mhlo.multiply %arg702, %1671 : tensor<1024xf32>
    %1673 = mhlo.add %1668, %1672 : tensor<1024xf32>
    %1674 = "mhlo.get_tuple_element"(%1403) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %1675 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1676 = "mhlo.broadcast_in_dim"(%1675) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1677 = mhlo.multiply %1674, %1676 : tensor<1024xf32>
    %1678 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1679 = mhlo.subtract %1678, %1675 : tensor<f32>
    %1680 = "mhlo.broadcast_in_dim"(%1679) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1681 = mhlo.multiply %arg703, %1680 : tensor<1024xf32>
    %1682 = mhlo.add %1677, %1681 : tensor<1024xf32>
    %1683 = "mhlo.get_tuple_element"(%1414) {index = 1 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1684 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1685 = "mhlo.broadcast_in_dim"(%1684) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1686 = mhlo.multiply %1683, %1685 : tensor<512xf32>
    %1687 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1688 = mhlo.subtract %1687, %1684 : tensor<f32>
    %1689 = "mhlo.broadcast_in_dim"(%1688) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1690 = mhlo.multiply %arg705, %1689 : tensor<512xf32>
    %1691 = mhlo.add %1686, %1690 : tensor<512xf32>
    %1692 = "mhlo.get_tuple_element"(%1414) {index = 2 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1693 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1694 = "mhlo.broadcast_in_dim"(%1693) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1695 = mhlo.multiply %1692, %1694 : tensor<512xf32>
    %1696 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1697 = mhlo.subtract %1696, %1693 : tensor<f32>
    %1698 = "mhlo.broadcast_in_dim"(%1697) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1699 = mhlo.multiply %arg706, %1698 : tensor<512xf32>
    %1700 = mhlo.add %1695, %1699 : tensor<512xf32>
    %1701 = "mhlo.get_tuple_element"(%1418) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1702 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1703 = "mhlo.broadcast_in_dim"(%1702) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1704 = mhlo.multiply %1701, %1703 : tensor<512xf32>
    %1705 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1706 = mhlo.subtract %1705, %1702 : tensor<f32>
    %1707 = "mhlo.broadcast_in_dim"(%1706) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1708 = mhlo.multiply %arg708, %1707 : tensor<512xf32>
    %1709 = mhlo.add %1704, %1708 : tensor<512xf32>
    %1710 = "mhlo.get_tuple_element"(%1418) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1711 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1712 = "mhlo.broadcast_in_dim"(%1711) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1713 = mhlo.multiply %1710, %1712 : tensor<512xf32>
    %1714 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1715 = mhlo.subtract %1714, %1711 : tensor<f32>
    %1716 = "mhlo.broadcast_in_dim"(%1715) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1717 = mhlo.multiply %arg709, %1716 : tensor<512xf32>
    %1718 = mhlo.add %1713, %1717 : tensor<512xf32>
    %1719 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1720 = "mhlo.broadcast_in_dim"(%1719) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1721 = mhlo.multiply %1524, %1720 : tensor<2048xf32>
    %1722 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1723 = mhlo.subtract %1722, %1719 : tensor<f32>
    %1724 = "mhlo.broadcast_in_dim"(%1723) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1725 = mhlo.multiply %arg711, %1724 : tensor<2048xf32>
    %1726 = mhlo.add %1721, %1725 : tensor<2048xf32>
    %1727 = "mhlo.get_tuple_element"(%1422) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1728 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1729 = "mhlo.broadcast_in_dim"(%1728) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1730 = mhlo.multiply %1727, %1729 : tensor<2048xf32>
    %1731 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1732 = mhlo.subtract %1731, %1728 : tensor<f32>
    %1733 = "mhlo.broadcast_in_dim"(%1732) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1734 = mhlo.multiply %arg712, %1733 : tensor<2048xf32>
    %1735 = mhlo.add %1730, %1734 : tensor<2048xf32>
    %1736 = "mhlo.get_tuple_element"(%1411) {index = 1 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1737 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1738 = "mhlo.broadcast_in_dim"(%1737) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1739 = mhlo.multiply %1736, %1738 : tensor<2048xf32>
    %1740 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1741 = mhlo.subtract %1740, %1737 : tensor<f32>
    %1742 = "mhlo.broadcast_in_dim"(%1741) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1743 = mhlo.multiply %arg714, %1742 : tensor<2048xf32>
    %1744 = mhlo.add %1739, %1743 : tensor<2048xf32>
    %1745 = "mhlo.get_tuple_element"(%1411) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1746 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1747 = "mhlo.broadcast_in_dim"(%1746) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1748 = mhlo.multiply %1745, %1747 : tensor<2048xf32>
    %1749 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1750 = mhlo.subtract %1749, %1746 : tensor<f32>
    %1751 = "mhlo.broadcast_in_dim"(%1750) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1752 = mhlo.multiply %arg715, %1751 : tensor<2048xf32>
    %1753 = mhlo.add %1748, %1752 : tensor<2048xf32>
    %1754 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1755 = "mhlo.broadcast_in_dim"(%1754) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1756 = mhlo.multiply %1526, %1755 : tensor<512xf32>
    %1757 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1758 = mhlo.subtract %1757, %1754 : tensor<f32>
    %1759 = "mhlo.broadcast_in_dim"(%1758) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1760 = mhlo.multiply %arg717, %1759 : tensor<512xf32>
    %1761 = mhlo.add %1756, %1760 : tensor<512xf32>
    %1762 = "mhlo.get_tuple_element"(%1430) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1763 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1764 = "mhlo.broadcast_in_dim"(%1763) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1765 = mhlo.multiply %1762, %1764 : tensor<512xf32>
    %1766 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1767 = mhlo.subtract %1766, %1763 : tensor<f32>
    %1768 = "mhlo.broadcast_in_dim"(%1767) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1769 = mhlo.multiply %arg718, %1768 : tensor<512xf32>
    %1770 = mhlo.add %1765, %1769 : tensor<512xf32>
    %1771 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1772 = "mhlo.broadcast_in_dim"(%1771) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1773 = mhlo.multiply %1532, %1772 : tensor<512xf32>
    %1774 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1775 = mhlo.subtract %1774, %1771 : tensor<f32>
    %1776 = "mhlo.broadcast_in_dim"(%1775) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1777 = mhlo.multiply %arg720, %1776 : tensor<512xf32>
    %1778 = mhlo.add %1773, %1777 : tensor<512xf32>
    %1779 = "mhlo.get_tuple_element"(%1434) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1780 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1781 = "mhlo.broadcast_in_dim"(%1780) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1782 = mhlo.multiply %1779, %1781 : tensor<512xf32>
    %1783 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1784 = mhlo.subtract %1783, %1780 : tensor<f32>
    %1785 = "mhlo.broadcast_in_dim"(%1784) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1786 = mhlo.multiply %arg721, %1785 : tensor<512xf32>
    %1787 = mhlo.add %1782, %1786 : tensor<512xf32>
    %1788 = "mhlo.get_tuple_element"(%1438) {index = 1 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1789 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1790 = "mhlo.broadcast_in_dim"(%1789) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1791 = mhlo.multiply %1788, %1790 : tensor<2048xf32>
    %1792 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1793 = mhlo.subtract %1792, %1789 : tensor<f32>
    %1794 = "mhlo.broadcast_in_dim"(%1793) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1795 = mhlo.multiply %arg723, %1794 : tensor<2048xf32>
    %1796 = mhlo.add %1791, %1795 : tensor<2048xf32>
    %1797 = "mhlo.get_tuple_element"(%1438) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %1798 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1799 = "mhlo.broadcast_in_dim"(%1798) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1800 = mhlo.multiply %1797, %1799 : tensor<2048xf32>
    %1801 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1802 = mhlo.subtract %1801, %1798 : tensor<f32>
    %1803 = "mhlo.broadcast_in_dim"(%1802) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %1804 = mhlo.multiply %arg724, %1803 : tensor<2048xf32>
    %1805 = mhlo.add %1800, %1804 : tensor<2048xf32>
    %1806 = "mhlo.get_tuple_element"(%1245) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1807 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1808 = "mhlo.broadcast_in_dim"(%1807) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1809 = mhlo.multiply %1806, %1808 : tensor<512xf32>
    %1810 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1811 = mhlo.subtract %1810, %1807 : tensor<f32>
    %1812 = "mhlo.broadcast_in_dim"(%1811) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1813 = mhlo.multiply %arg619, %1812 : tensor<512xf32>
    %1814 = mhlo.add %1809, %1813 : tensor<512xf32>
    %1815 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1816 = "mhlo.broadcast_in_dim"(%1815) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1817 = mhlo.multiply %1550, %1816 : tensor<1024xf32>
    %1818 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1819 = mhlo.subtract %1818, %1815 : tensor<f32>
    %1820 = "mhlo.broadcast_in_dim"(%1819) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %1821 = mhlo.multiply %arg654, %1820 : tensor<1024xf32>
    %1822 = mhlo.add %1817, %1821 : tensor<1024xf32>
    %1823 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1824 = "mhlo.broadcast_in_dim"(%1823) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1825 = mhlo.multiply %1564, %1824 : tensor<128xf32>
    %1826 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1827 = mhlo.subtract %1826, %1823 : tensor<f32>
    %1828 = "mhlo.broadcast_in_dim"(%1827) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1829 = mhlo.multiply %arg621, %1828 : tensor<128xf32>
    %1830 = mhlo.add %1825, %1829 : tensor<128xf32>
    %1831 = "mhlo.get_tuple_element"(%1264) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1832 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1833 = "mhlo.broadcast_in_dim"(%1832) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1834 = mhlo.multiply %1831, %1833 : tensor<128xf32>
    %1835 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1836 = mhlo.subtract %1835, %1832 : tensor<f32>
    %1837 = "mhlo.broadcast_in_dim"(%1836) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1838 = mhlo.multiply %arg622, %1837 : tensor<128xf32>
    %1839 = mhlo.add %1834, %1838 : tensor<128xf32>
    %1840 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1841 = "mhlo.broadcast_in_dim"(%1840) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1842 = mhlo.multiply %1568, %1841 : tensor<128xf32>
    %1843 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1844 = mhlo.subtract %1843, %1840 : tensor<f32>
    %1845 = "mhlo.broadcast_in_dim"(%1844) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1846 = mhlo.multiply %arg624, %1845 : tensor<128xf32>
    %1847 = mhlo.add %1842, %1846 : tensor<128xf32>
    %1848 = "mhlo.get_tuple_element"(%1268) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1849 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1850 = "mhlo.broadcast_in_dim"(%1849) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1851 = mhlo.multiply %1848, %1850 : tensor<128xf32>
    %1852 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1853 = mhlo.subtract %1852, %1849 : tensor<f32>
    %1854 = "mhlo.broadcast_in_dim"(%1853) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1855 = mhlo.multiply %arg625, %1854 : tensor<128xf32>
    %1856 = mhlo.add %1851, %1855 : tensor<128xf32>
    %1857 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1858 = "mhlo.broadcast_in_dim"(%1857) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1859 = mhlo.multiply %1574, %1858 : tensor<512xf32>
    %1860 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1861 = mhlo.subtract %1860, %1857 : tensor<f32>
    %1862 = "mhlo.broadcast_in_dim"(%1861) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1863 = mhlo.multiply %arg627, %1862 : tensor<512xf32>
    %1864 = mhlo.add %1859, %1863 : tensor<512xf32>
    %1865 = "mhlo.get_tuple_element"(%1272) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1866 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1867 = "mhlo.broadcast_in_dim"(%1866) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1868 = mhlo.multiply %1865, %1867 : tensor<512xf32>
    %1869 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1870 = mhlo.subtract %1869, %1866 : tensor<f32>
    %1871 = "mhlo.broadcast_in_dim"(%1870) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1872 = mhlo.multiply %arg628, %1871 : tensor<512xf32>
    %1873 = mhlo.add %1868, %1872 : tensor<512xf32>
    %1874 = "mhlo.get_tuple_element"(%1280) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1875 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1876 = "mhlo.broadcast_in_dim"(%1875) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1877 = mhlo.multiply %1874, %1876 : tensor<128xf32>
    %1878 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1879 = mhlo.subtract %1878, %1875 : tensor<f32>
    %1880 = "mhlo.broadcast_in_dim"(%1879) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1881 = mhlo.multiply %arg630, %1880 : tensor<128xf32>
    %1882 = mhlo.add %1877, %1881 : tensor<128xf32>
    %1883 = "mhlo.get_tuple_element"(%1280) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1884 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1885 = "mhlo.broadcast_in_dim"(%1884) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1886 = mhlo.multiply %1883, %1885 : tensor<128xf32>
    %1887 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1888 = mhlo.subtract %1887, %1884 : tensor<f32>
    %1889 = "mhlo.broadcast_in_dim"(%1888) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1890 = mhlo.multiply %arg631, %1889 : tensor<128xf32>
    %1891 = mhlo.add %1886, %1890 : tensor<128xf32>
    %1892 = "mhlo.get_tuple_element"(%1284) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1893 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1894 = "mhlo.broadcast_in_dim"(%1893) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1895 = mhlo.multiply %1892, %1894 : tensor<128xf32>
    %1896 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1897 = mhlo.subtract %1896, %1893 : tensor<f32>
    %1898 = "mhlo.broadcast_in_dim"(%1897) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1899 = mhlo.multiply %arg633, %1898 : tensor<128xf32>
    %1900 = mhlo.add %1895, %1899 : tensor<128xf32>
    %1901 = "mhlo.get_tuple_element"(%1284) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1902 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1903 = "mhlo.broadcast_in_dim"(%1902) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1904 = mhlo.multiply %1901, %1903 : tensor<128xf32>
    %1905 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1906 = mhlo.subtract %1905, %1902 : tensor<f32>
    %1907 = "mhlo.broadcast_in_dim"(%1906) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1908 = mhlo.multiply %arg634, %1907 : tensor<128xf32>
    %1909 = mhlo.add %1904, %1908 : tensor<128xf32>
    %1910 = "mhlo.get_tuple_element"(%1288) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1911 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1912 = "mhlo.broadcast_in_dim"(%1911) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1913 = mhlo.multiply %1910, %1912 : tensor<512xf32>
    %1914 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1915 = mhlo.subtract %1914, %1911 : tensor<f32>
    %1916 = "mhlo.broadcast_in_dim"(%1915) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1917 = mhlo.multiply %arg636, %1916 : tensor<512xf32>
    %1918 = mhlo.add %1913, %1917 : tensor<512xf32>
    %1919 = "mhlo.get_tuple_element"(%1288) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1920 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1921 = "mhlo.broadcast_in_dim"(%1920) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1922 = mhlo.multiply %1919, %1921 : tensor<512xf32>
    %1923 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1924 = mhlo.subtract %1923, %1920 : tensor<f32>
    %1925 = "mhlo.broadcast_in_dim"(%1924) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1926 = mhlo.multiply %arg637, %1925 : tensor<512xf32>
    %1927 = mhlo.add %1922, %1926 : tensor<512xf32>
    %1928 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1929 = "mhlo.broadcast_in_dim"(%1928) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1930 = mhlo.multiply %1552, %1929 : tensor<128xf32>
    %1931 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1932 = mhlo.subtract %1931, %1928 : tensor<f32>
    %1933 = "mhlo.broadcast_in_dim"(%1932) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1934 = mhlo.multiply %arg639, %1933 : tensor<128xf32>
    %1935 = mhlo.add %1930, %1934 : tensor<128xf32>
    %1936 = "mhlo.get_tuple_element"(%1296) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1937 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1938 = "mhlo.broadcast_in_dim"(%1937) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1939 = mhlo.multiply %1936, %1938 : tensor<128xf32>
    %1940 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1941 = mhlo.subtract %1940, %1937 : tensor<f32>
    %1942 = "mhlo.broadcast_in_dim"(%1941) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1943 = mhlo.multiply %arg640, %1942 : tensor<128xf32>
    %1944 = mhlo.add %1939, %1943 : tensor<128xf32>
    %1945 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1946 = "mhlo.broadcast_in_dim"(%1945) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1947 = mhlo.multiply %1556, %1946 : tensor<128xf32>
    %1948 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1949 = mhlo.subtract %1948, %1945 : tensor<f32>
    %1950 = "mhlo.broadcast_in_dim"(%1949) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1951 = mhlo.multiply %arg642, %1950 : tensor<128xf32>
    %1952 = mhlo.add %1947, %1951 : tensor<128xf32>
    %1953 = "mhlo.get_tuple_element"(%1300) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %1954 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1955 = "mhlo.broadcast_in_dim"(%1954) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1956 = mhlo.multiply %1953, %1955 : tensor<128xf32>
    %1957 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1958 = mhlo.subtract %1957, %1954 : tensor<f32>
    %1959 = "mhlo.broadcast_in_dim"(%1958) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %1960 = mhlo.multiply %arg643, %1959 : tensor<128xf32>
    %1961 = mhlo.add %1956, %1960 : tensor<128xf32>
    %1962 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1963 = "mhlo.broadcast_in_dim"(%1962) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1964 = mhlo.multiply %1560, %1963 : tensor<512xf32>
    %1965 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1966 = mhlo.subtract %1965, %1962 : tensor<f32>
    %1967 = "mhlo.broadcast_in_dim"(%1966) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1968 = mhlo.multiply %arg645, %1967 : tensor<512xf32>
    %1969 = mhlo.add %1964, %1968 : tensor<512xf32>
    %1970 = "mhlo.get_tuple_element"(%1304) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %1971 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1972 = "mhlo.broadcast_in_dim"(%1971) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1973 = mhlo.multiply %1970, %1972 : tensor<512xf32>
    %1974 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1975 = mhlo.subtract %1974, %1971 : tensor<f32>
    %1976 = "mhlo.broadcast_in_dim"(%1975) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %1977 = mhlo.multiply %arg646, %1976 : tensor<512xf32>
    %1978 = mhlo.add %1973, %1977 : tensor<512xf32>
    %1979 = "mhlo.get_tuple_element"(%1315) {index = 1 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1980 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1981 = "mhlo.broadcast_in_dim"(%1980) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1982 = mhlo.multiply %1979, %1981 : tensor<256xf32>
    %1983 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1984 = mhlo.subtract %1983, %1980 : tensor<f32>
    %1985 = "mhlo.broadcast_in_dim"(%1984) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1986 = mhlo.multiply %arg648, %1985 : tensor<256xf32>
    %1987 = mhlo.add %1982, %1986 : tensor<256xf32>
    %1988 = "mhlo.get_tuple_element"(%1315) {index = 2 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1989 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1990 = "mhlo.broadcast_in_dim"(%1989) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1991 = mhlo.multiply %1988, %1990 : tensor<256xf32>
    %1992 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1993 = mhlo.subtract %1992, %1989 : tensor<f32>
    %1994 = "mhlo.broadcast_in_dim"(%1993) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %1995 = mhlo.multiply %arg649, %1994 : tensor<256xf32>
    %1996 = mhlo.add %1991, %1995 : tensor<256xf32>
    %1997 = "mhlo.get_tuple_element"(%1319) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %1998 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %1999 = "mhlo.broadcast_in_dim"(%1998) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2000 = mhlo.multiply %1997, %1999 : tensor<256xf32>
    %2001 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2002 = mhlo.subtract %2001, %1998 : tensor<f32>
    %2003 = "mhlo.broadcast_in_dim"(%2002) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2004 = mhlo.multiply %arg651, %2003 : tensor<256xf32>
    %2005 = mhlo.add %2000, %2004 : tensor<256xf32>
    %2006 = "mhlo.get_tuple_element"(%1319) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2007 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2008 = "mhlo.broadcast_in_dim"(%2007) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2009 = mhlo.multiply %2006, %2008 : tensor<256xf32>
    %2010 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2011 = mhlo.subtract %2010, %2007 : tensor<f32>
    %2012 = "mhlo.broadcast_in_dim"(%2011) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2013 = mhlo.multiply %arg652, %2012 : tensor<256xf32>
    %2014 = mhlo.add %2009, %2013 : tensor<256xf32>
    %2015 = "mhlo.get_tuple_element"(%1201) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2016 = "mhlo.get_tuple_element"(%1201) {index = 3 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2017 = "mhlo.get_tuple_element"(%424) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2018 = call @aten.view.3422(%2017) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2019 = "mhlo.get_tuple_element"(%424) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2020 = call @aten.view.3422(%2019) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2021 = "mhlo.get_tuple_element"(%1205) {index = 1 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2022 = "mhlo.get_tuple_element"(%1205) {index = 3 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2023 = "mhlo.get_tuple_element"(%1213) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2024 = "mhlo.get_tuple_element"(%1213) {index = 3 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2025 = "mhlo.get_tuple_element"(%1438) {index = 3 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %2026 = "mhlo.get_tuple_element"(%1446) {index = 3 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2027 = "mhlo.get_tuple_element"(%1450) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2028 = "mhlo.get_tuple_element"(%1450) {index = 3 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2029 = "mhlo.get_tuple_element"(%1371) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2030 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2031 = "mhlo.broadcast_in_dim"(%2030) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2032 = mhlo.multiply %2015, %2031 : tensor<64xf32>
    %2033 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2034 = mhlo.subtract %2033, %2030 : tensor<f32>
    %2035 = "mhlo.broadcast_in_dim"(%2034) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2036 = mhlo.multiply %arg582, %2035 : tensor<64xf32>
    %2037 = mhlo.add %2032, %2036 : tensor<64xf32>
    %2038 = "mhlo.get_tuple_element"(%1371) {index = 3 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2039 = "mhlo.get_tuple_element"(%1403) {index = 3 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2040 = "mhlo.get_tuple_element"(%1379) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2041 = "mhlo.get_tuple_element"(%1379) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2042 = "mhlo.get_tuple_element"(%1411) {index = 3 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %2043 = "mhlo.get_tuple_element"(%774) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2044 = call @aten.view.3422(%2043) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2045 = "mhlo.get_tuple_element"(%774) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2046 = call @aten.view.3422(%2045) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2047 = "mhlo.get_tuple_element"(%1414) {index = 3 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2048 = "mhlo.get_tuple_element"(%1383) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2049 = "mhlo.get_tuple_element"(%1383) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2050 = "mhlo.get_tuple_element"(%869) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2051 = call @aten.view.3422(%2050) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2052 = "mhlo.get_tuple_element"(%1187) {index = 1 : i32} : (tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2053 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2054 = "mhlo.broadcast_in_dim"(%2053) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2055 = mhlo.multiply %2052, %2054 : tensor<64xf32>
    %2056 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2057 = mhlo.subtract %2056, %2053 : tensor<f32>
    %2058 = "mhlo.broadcast_in_dim"(%2057) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2059 = mhlo.multiply %arg576, %2058 : tensor<64xf32>
    %2060 = mhlo.add %2055, %2059 : tensor<64xf32>
    %2061 = "mhlo.get_tuple_element"(%869) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2062 = call @aten.view.3422(%2061) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2063 = "mhlo.get_tuple_element"(%1187) {index = 2 : i32} : (tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2064 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2065 = "mhlo.broadcast_in_dim"(%2064) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2066 = mhlo.multiply %2063, %2065 : tensor<64xf32>
    %2067 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2068 = mhlo.subtract %2067, %2064 : tensor<f32>
    %2069 = "mhlo.broadcast_in_dim"(%2068) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2070 = mhlo.multiply %arg577, %2069 : tensor<64xf32>
    %2071 = mhlo.add %2066, %2070 : tensor<64xf32>
    %2072 = "mhlo.get_tuple_element"(%1197) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2073 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2074 = "mhlo.broadcast_in_dim"(%2073) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2075 = mhlo.multiply %2072, %2074 : tensor<64xf32>
    %2076 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2077 = mhlo.subtract %2076, %2073 : tensor<f32>
    %2078 = "mhlo.broadcast_in_dim"(%2077) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2079 = mhlo.multiply %arg579, %2078 : tensor<64xf32>
    %2080 = mhlo.add %2075, %2079 : tensor<64xf32>
    %2081 = "mhlo.get_tuple_element"(%1418) {index = 3 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2082 = "mhlo.get_tuple_element"(%1197) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2083 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2084 = "mhlo.broadcast_in_dim"(%2083) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2085 = mhlo.multiply %2082, %2084 : tensor<64xf32>
    %2086 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2087 = mhlo.subtract %2086, %2083 : tensor<f32>
    %2088 = "mhlo.broadcast_in_dim"(%2087) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2089 = mhlo.multiply %arg580, %2088 : tensor<64xf32>
    %2090 = mhlo.add %2085, %2089 : tensor<64xf32>
    %2091 = "mhlo.get_tuple_element"(%1184) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2092 = call @aten.view.3422(%2091) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2093 = "mhlo.get_tuple_element"(%1184) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2094 = call @aten.view.3422(%2093) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2095 = "mhlo.get_tuple_element"(%109) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2096 = call @aten.view.3422(%2095) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2097 = "mhlo.get_tuple_element"(%109) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2098 = call @aten.view.3422(%2097) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2099 = "mhlo.get_tuple_element"(%1089) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2100 = call @aten.view.3422(%2099) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2101 = "mhlo.get_tuple_element"(%1089) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2102 = call @aten.view.3422(%2101) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2103 = "mhlo.get_tuple_element"(%1339) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2104 = "mhlo.get_tuple_element"(%1339) {index = 3 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2105 = "mhlo.get_tuple_element"(%679) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2106 = call @aten.view.3422(%2105) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2107 = "mhlo.get_tuple_element"(%679) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2108 = call @aten.view.3422(%2107) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2109 = "mhlo.get_tuple_element"(%1312) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2110 = "mhlo.get_tuple_element"(%1312) {index = 3 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2111 = "mhlo.get_tuple_element"(%1347) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2112 = "mhlo.get_tuple_element"(%1315) {index = 3 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2113 = "mhlo.get_tuple_element"(%1347) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2114 = "mhlo.get_tuple_element"(%709) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2115 = call @aten.view.3422(%2114) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2116 = "mhlo.get_tuple_element"(%1351) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2117 = "mhlo.get_tuple_element"(%709) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2118 = call @aten.view.3422(%2117) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2119 = "mhlo.get_tuple_element"(%1319) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2120 = "mhlo.get_tuple_element"(%1351) {index = 3 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2121 = "mhlo.get_tuple_element"(%614) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2122 = call @aten.view.3422(%2121) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2123 = "mhlo.get_tuple_element"(%614) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2124 = call @aten.view.3422(%2123) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2125 = "mhlo.get_tuple_element"(%1248) {index = 1 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %2126 = "mhlo.get_tuple_element"(%1248) {index = 3 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %2127 = "mhlo.get_tuple_element"(%1252) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %2128 = "mhlo.get_tuple_element"(%1252) {index = 3 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %2129 = "mhlo.get_tuple_element"(%1256) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2130 = "mhlo.get_tuple_element"(%1256) {index = 3 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2131 = "mhlo.get_tuple_element"(%1446) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2132 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2133 = "mhlo.broadcast_in_dim"(%2132) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2134 = mhlo.multiply %2131, %2133 : tensor<512xf32>
    %2135 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2136 = mhlo.subtract %2135, %2132 : tensor<f32>
    %2137 = "mhlo.broadcast_in_dim"(%2136) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2138 = mhlo.multiply %arg727, %2137 : tensor<512xf32>
    %2139 = mhlo.add %2134, %2138 : tensor<512xf32>
    %2140 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2141 = "mhlo.broadcast_in_dim"(%2140) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2142 = mhlo.multiply %2027, %2141 : tensor<512xf32>
    %2143 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2144 = mhlo.subtract %2143, %2140 : tensor<f32>
    %2145 = "mhlo.broadcast_in_dim"(%2144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2146 = mhlo.multiply %arg729, %2145 : tensor<512xf32>
    %2147 = mhlo.add %2142, %2146 : tensor<512xf32>
    %2148 = "mhlo.get_tuple_element"(%1450) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2149 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2150 = "mhlo.broadcast_in_dim"(%2149) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2151 = mhlo.multiply %2148, %2150 : tensor<512xf32>
    %2152 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2153 = mhlo.subtract %2152, %2149 : tensor<f32>
    %2154 = "mhlo.broadcast_in_dim"(%2153) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2155 = mhlo.multiply %arg730, %2154 : tensor<512xf32>
    %2156 = mhlo.add %2151, %2155 : tensor<512xf32>
    %2157 = "mhlo.get_tuple_element"(%1280) {index = 3 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %2158 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2159 = "mhlo.broadcast_in_dim"(%2158) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %2160 = mhlo.multiply %1504, %2159 : tensor<2048xf32>
    %2161 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2162 = mhlo.subtract %2161, %2158 : tensor<f32>
    %2163 = "mhlo.broadcast_in_dim"(%2162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %2164 = mhlo.multiply %arg732, %2163 : tensor<2048xf32>
    %2165 = mhlo.add %2160, %2164 : tensor<2048xf32>
    %2166 = "mhlo.get_tuple_element"(%1454) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %2167 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2168 = "mhlo.broadcast_in_dim"(%2167) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %2169 = mhlo.multiply %2166, %2168 : tensor<2048xf32>
    %2170 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2171 = mhlo.subtract %2170, %2167 : tensor<f32>
    %2172 = "mhlo.broadcast_in_dim"(%2171) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %2173 = mhlo.multiply %arg733, %2172 : tensor<2048xf32>
    %2174 = mhlo.add %2169, %2173 : tensor<2048xf32>
    %2175 = "mhlo.get_tuple_element"(%1284) {index = 3 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %2176 = "mhlo.get_tuple_element"(%1288) {index = 3 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2177 = "mhlo.get_tuple_element"(%1217) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2178 = "mhlo.get_tuple_element"(%1217) {index = 3 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2179 = "mhlo.get_tuple_element"(%1221) {index = 1 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2180 = "mhlo.get_tuple_element"(%1221) {index = 3 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2181 = "mhlo.get_tuple_element"(%1229) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2182 = "mhlo.get_tuple_element"(%1229) {index = 3 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2183 = "mhlo.get_tuple_element"(%1323) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2184 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2185 = "mhlo.broadcast_in_dim"(%2184) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2186 = mhlo.multiply %2183, %2185 : tensor<1024xf32>
    %2187 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2188 = mhlo.subtract %2187, %2184 : tensor<f32>
    %2189 = "mhlo.broadcast_in_dim"(%2188) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2190 = mhlo.multiply %arg655, %2189 : tensor<1024xf32>
    %2191 = mhlo.add %2186, %2190 : tensor<1024xf32>
    %2192 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2193 = "mhlo.broadcast_in_dim"(%2192) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2194 = mhlo.multiply %2048, %2193 : tensor<256xf32>
    %2195 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2196 = mhlo.subtract %2195, %2192 : tensor<f32>
    %2197 = "mhlo.broadcast_in_dim"(%2196) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2198 = mhlo.multiply %arg690, %2197 : tensor<256xf32>
    %2199 = mhlo.add %2194, %2198 : tensor<256xf32>
    %2200 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2201 = "mhlo.broadcast_in_dim"(%2200) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2202 = mhlo.multiply %2109, %2201 : tensor<1024xf32>
    %2203 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2204 = mhlo.subtract %2203, %2200 : tensor<f32>
    %2205 = "mhlo.broadcast_in_dim"(%2204) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2206 = mhlo.multiply %arg657, %2205 : tensor<1024xf32>
    %2207 = mhlo.add %2202, %2206 : tensor<1024xf32>
    %2208 = "mhlo.get_tuple_element"(%1312) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2209 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2210 = "mhlo.broadcast_in_dim"(%2209) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2211 = mhlo.multiply %2208, %2210 : tensor<1024xf32>
    %2212 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2213 = mhlo.subtract %2212, %2209 : tensor<f32>
    %2214 = "mhlo.broadcast_in_dim"(%2213) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2215 = mhlo.multiply %arg658, %2214 : tensor<1024xf32>
    %2216 = mhlo.add %2211, %2215 : tensor<1024xf32>
    %2217 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2218 = "mhlo.broadcast_in_dim"(%2217) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2219 = mhlo.multiply %1554, %2218 : tensor<256xf32>
    %2220 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2221 = mhlo.subtract %2220, %2217 : tensor<f32>
    %2222 = "mhlo.broadcast_in_dim"(%2221) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2223 = mhlo.multiply %arg660, %2222 : tensor<256xf32>
    %2224 = mhlo.add %2219, %2223 : tensor<256xf32>
    %2225 = "mhlo.get_tuple_element"(%1331) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2226 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2227 = "mhlo.broadcast_in_dim"(%2226) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2228 = mhlo.multiply %2225, %2227 : tensor<256xf32>
    %2229 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2230 = mhlo.subtract %2229, %2226 : tensor<f32>
    %2231 = "mhlo.broadcast_in_dim"(%2230) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2232 = mhlo.multiply %arg661, %2231 : tensor<256xf32>
    %2233 = mhlo.add %2228, %2232 : tensor<256xf32>
    %2234 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2235 = "mhlo.broadcast_in_dim"(%2234) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2236 = mhlo.multiply %1558, %2235 : tensor<256xf32>
    %2237 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2238 = mhlo.subtract %2237, %2234 : tensor<f32>
    %2239 = "mhlo.broadcast_in_dim"(%2238) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2240 = mhlo.multiply %arg663, %2239 : tensor<256xf32>
    %2241 = mhlo.add %2236, %2240 : tensor<256xf32>
    %2242 = "mhlo.get_tuple_element"(%1335) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2243 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2244 = "mhlo.broadcast_in_dim"(%2243) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2245 = mhlo.multiply %2242, %2244 : tensor<256xf32>
    %2246 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2247 = mhlo.subtract %2246, %2243 : tensor<f32>
    %2248 = "mhlo.broadcast_in_dim"(%2247) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2249 = mhlo.multiply %arg664, %2248 : tensor<256xf32>
    %2250 = mhlo.add %2245, %2249 : tensor<256xf32>
    %2251 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2252 = "mhlo.broadcast_in_dim"(%2251) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2253 = mhlo.multiply %2103, %2252 : tensor<1024xf32>
    %2254 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2255 = mhlo.subtract %2254, %2251 : tensor<f32>
    %2256 = "mhlo.broadcast_in_dim"(%2255) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2257 = mhlo.multiply %arg666, %2256 : tensor<1024xf32>
    %2258 = mhlo.add %2253, %2257 : tensor<1024xf32>
    %2259 = "mhlo.get_tuple_element"(%1339) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2260 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2261 = "mhlo.broadcast_in_dim"(%2260) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2262 = mhlo.multiply %2259, %2261 : tensor<1024xf32>
    %2263 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2264 = mhlo.subtract %2263, %2260 : tensor<f32>
    %2265 = "mhlo.broadcast_in_dim"(%2264) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2266 = mhlo.multiply %arg667, %2265 : tensor<1024xf32>
    %2267 = mhlo.add %2262, %2266 : tensor<1024xf32>
    %2268 = "mhlo.get_tuple_element"(%204) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2269 = call @aten.view.3422(%2268) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2270 = "mhlo.get_tuple_element"(%204) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2271 = call @aten.view.3422(%2270) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2272 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2273 = "mhlo.broadcast_in_dim"(%2272) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2274 = mhlo.multiply %2111, %2273 : tensor<256xf32>
    %2275 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2276 = mhlo.subtract %2275, %2272 : tensor<f32>
    %2277 = "mhlo.broadcast_in_dim"(%2276) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2278 = mhlo.multiply %arg669, %2277 : tensor<256xf32>
    %2279 = mhlo.add %2274, %2278 : tensor<256xf32>
    %2280 = "mhlo.get_tuple_element"(%1347) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2281 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2282 = "mhlo.broadcast_in_dim"(%2281) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2283 = mhlo.multiply %2280, %2282 : tensor<256xf32>
    %2284 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2285 = mhlo.subtract %2284, %2281 : tensor<f32>
    %2286 = "mhlo.broadcast_in_dim"(%2285) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2287 = mhlo.multiply %arg670, %2286 : tensor<256xf32>
    %2288 = mhlo.add %2283, %2287 : tensor<256xf32>
    %2289 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2290 = "mhlo.broadcast_in_dim"(%2289) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2291 = mhlo.multiply %2116, %2290 : tensor<256xf32>
    %2292 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2293 = mhlo.subtract %2292, %2289 : tensor<f32>
    %2294 = "mhlo.broadcast_in_dim"(%2293) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2295 = mhlo.multiply %arg672, %2294 : tensor<256xf32>
    %2296 = mhlo.add %2291, %2295 : tensor<256xf32>
    %2297 = "mhlo.get_tuple_element"(%1351) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2298 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2299 = "mhlo.broadcast_in_dim"(%2298) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2300 = mhlo.multiply %2297, %2299 : tensor<256xf32>
    %2301 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2302 = mhlo.subtract %2301, %2298 : tensor<f32>
    %2303 = "mhlo.broadcast_in_dim"(%2302) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2304 = mhlo.multiply %arg673, %2303 : tensor<256xf32>
    %2305 = mhlo.add %2300, %2304 : tensor<256xf32>
    %2306 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2307 = "mhlo.broadcast_in_dim"(%2306) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2308 = mhlo.multiply %1562, %2307 : tensor<1024xf32>
    %2309 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2310 = mhlo.subtract %2309, %2306 : tensor<f32>
    %2311 = "mhlo.broadcast_in_dim"(%2310) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2312 = mhlo.multiply %arg675, %2311 : tensor<1024xf32>
    %2313 = mhlo.add %2308, %2312 : tensor<1024xf32>
    %2314 = "mhlo.get_tuple_element"(%1355) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2315 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2316 = "mhlo.broadcast_in_dim"(%2315) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2317 = mhlo.multiply %2314, %2316 : tensor<1024xf32>
    %2318 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2319 = mhlo.subtract %2318, %2315 : tensor<f32>
    %2320 = "mhlo.broadcast_in_dim"(%2319) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2321 = mhlo.multiply %arg676, %2320 : tensor<1024xf32>
    %2322 = mhlo.add %2317, %2321 : tensor<1024xf32>
    %2323 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2324 = "mhlo.broadcast_in_dim"(%2323) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2325 = mhlo.multiply %1566, %2324 : tensor<256xf32>
    %2326 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2327 = mhlo.subtract %2326, %2323 : tensor<f32>
    %2328 = "mhlo.broadcast_in_dim"(%2327) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2329 = mhlo.multiply %arg678, %2328 : tensor<256xf32>
    %2330 = mhlo.add %2325, %2329 : tensor<256xf32>
    %2331 = "mhlo.get_tuple_element"(%1363) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2332 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2333 = "mhlo.broadcast_in_dim"(%2332) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2334 = mhlo.multiply %2331, %2333 : tensor<256xf32>
    %2335 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2336 = mhlo.subtract %2335, %2332 : tensor<f32>
    %2337 = "mhlo.broadcast_in_dim"(%2336) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2338 = mhlo.multiply %arg679, %2337 : tensor<256xf32>
    %2339 = mhlo.add %2334, %2338 : tensor<256xf32>
    %2340 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2341 = "mhlo.broadcast_in_dim"(%2340) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2342 = mhlo.multiply %1570, %2341 : tensor<256xf32>
    %2343 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2344 = mhlo.subtract %2343, %2340 : tensor<f32>
    %2345 = "mhlo.broadcast_in_dim"(%2344) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2346 = mhlo.multiply %arg681, %2345 : tensor<256xf32>
    %2347 = mhlo.add %2342, %2346 : tensor<256xf32>
    %2348 = "mhlo.get_tuple_element"(%1367) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2349 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2350 = "mhlo.broadcast_in_dim"(%2349) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2351 = mhlo.multiply %2348, %2350 : tensor<256xf32>
    %2352 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2353 = mhlo.subtract %2352, %2349 : tensor<f32>
    %2354 = "mhlo.broadcast_in_dim"(%2353) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2355 = mhlo.multiply %arg682, %2354 : tensor<256xf32>
    %2356 = mhlo.add %2351, %2355 : tensor<256xf32>
    %2357 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2358 = "mhlo.broadcast_in_dim"(%2357) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2359 = mhlo.multiply %2029, %2358 : tensor<1024xf32>
    %2360 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2361 = mhlo.subtract %2360, %2357 : tensor<f32>
    %2362 = "mhlo.broadcast_in_dim"(%2361) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2363 = mhlo.multiply %arg684, %2362 : tensor<1024xf32>
    %2364 = mhlo.add %2359, %2363 : tensor<1024xf32>
    %2365 = "mhlo.get_tuple_element"(%1371) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %2366 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2367 = "mhlo.broadcast_in_dim"(%2366) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2368 = mhlo.multiply %2365, %2367 : tensor<1024xf32>
    %2369 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2370 = mhlo.subtract %2369, %2366 : tensor<f32>
    %2371 = "mhlo.broadcast_in_dim"(%2370) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2372 = mhlo.multiply %arg685, %2371 : tensor<1024xf32>
    %2373 = mhlo.add %2368, %2372 : tensor<1024xf32>
    %2374 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2375 = "mhlo.broadcast_in_dim"(%2374) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2376 = mhlo.multiply %2040, %2375 : tensor<256xf32>
    %2377 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2378 = mhlo.subtract %2377, %2374 : tensor<f32>
    %2379 = "mhlo.broadcast_in_dim"(%2378) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2380 = mhlo.multiply %arg687, %2379 : tensor<256xf32>
    %2381 = mhlo.add %2376, %2380 : tensor<256xf32>
    %2382 = "mhlo.get_tuple_element"(%1379) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2383 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2384 = "mhlo.broadcast_in_dim"(%2383) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2385 = mhlo.multiply %2382, %2384 : tensor<256xf32>
    %2386 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2387 = mhlo.subtract %2386, %2383 : tensor<f32>
    %2388 = "mhlo.broadcast_in_dim"(%2387) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2389 = mhlo.multiply %arg688, %2388 : tensor<256xf32>
    %2390 = mhlo.add %2385, %2389 : tensor<256xf32>
    %2391 = "mhlo.get_tuple_element"(%1187) {index = 3 : i32} : (tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2392 = "mhlo.get_tuple_element"(%1194) {index = 1 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2393 = "mhlo.get_tuple_element"(%1194) {index = 3 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2394 = "mhlo.get_tuple_element"(%1197) {index = 3 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2395 = "mhlo.slice"(%arg736) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %2396 = "mhlo.slice"(%2395) {limit_indices = dense<[1, 240]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x240xi64>
    %2397 = "mhlo.get_tuple_element"(%31) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2398 = call @aten.view.3422(%2397) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2399 = "mhlo.get_tuple_element"(%31) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2400 = call @aten.view.3422(%2399) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2401 = "mhlo.get_tuple_element"(%1201) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2402 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2403 = "mhlo.broadcast_in_dim"(%2402) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2404 = mhlo.multiply %2401, %2403 : tensor<64xf32>
    %2405 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2406 = mhlo.subtract %2405, %2402 : tensor<f32>
    %2407 = "mhlo.broadcast_in_dim"(%2406) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2408 = mhlo.multiply %arg583, %2407 : tensor<64xf32>
    %2409 = mhlo.add %2404, %2408 : tensor<64xf32>
    %2410 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2411 = "mhlo.broadcast_in_dim"(%2410) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2412 = mhlo.multiply %1586, %2411 : tensor<512xf32>
    %2413 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2414 = mhlo.subtract %2413, %2410 : tensor<f32>
    %2415 = "mhlo.broadcast_in_dim"(%2414) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2416 = mhlo.multiply %arg618, %2415 : tensor<512xf32>
    %2417 = mhlo.add %2412, %2416 : tensor<512xf32>
    %2418 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2419 = "mhlo.broadcast_in_dim"(%2418) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2420 = mhlo.multiply %2021, %2419 : tensor<256xf32>
    %2421 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2422 = mhlo.subtract %2421, %2418 : tensor<f32>
    %2423 = "mhlo.broadcast_in_dim"(%2422) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2424 = mhlo.multiply %arg585, %2423 : tensor<256xf32>
    %2425 = mhlo.add %2420, %2424 : tensor<256xf32>
    %2426 = "mhlo.get_tuple_element"(%1205) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2427 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2428 = "mhlo.broadcast_in_dim"(%2427) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2429 = mhlo.multiply %2426, %2428 : tensor<256xf32>
    %2430 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2431 = mhlo.subtract %2430, %2427 : tensor<f32>
    %2432 = "mhlo.broadcast_in_dim"(%2431) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2433 = mhlo.multiply %arg586, %2432 : tensor<256xf32>
    %2434 = mhlo.add %2429, %2433 : tensor<256xf32>
    %2435 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2436 = "mhlo.broadcast_in_dim"(%2435) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2437 = mhlo.multiply %2392, %2436 : tensor<256xf32>
    %2438 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2439 = mhlo.subtract %2438, %2435 : tensor<f32>
    %2440 = "mhlo.broadcast_in_dim"(%2439) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2441 = mhlo.multiply %arg588, %2440 : tensor<256xf32>
    %2442 = mhlo.add %2437, %2441 : tensor<256xf32>
    %2443 = "mhlo.get_tuple_element"(%1194) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2444 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2445 = "mhlo.broadcast_in_dim"(%2444) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2446 = mhlo.multiply %2443, %2445 : tensor<256xf32>
    %2447 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2448 = mhlo.subtract %2447, %2444 : tensor<f32>
    %2449 = "mhlo.broadcast_in_dim"(%2448) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2450 = mhlo.multiply %arg589, %2449 : tensor<256xf32>
    %2451 = mhlo.add %2446, %2450 : tensor<256xf32>
    %2452 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2453 = "mhlo.broadcast_in_dim"(%2452) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2454 = mhlo.multiply %2023, %2453 : tensor<64xf32>
    %2455 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2456 = mhlo.subtract %2455, %2452 : tensor<f32>
    %2457 = "mhlo.broadcast_in_dim"(%2456) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2458 = mhlo.multiply %arg591, %2457 : tensor<64xf32>
    %2459 = mhlo.add %2454, %2458 : tensor<64xf32>
    %2460 = "mhlo.get_tuple_element"(%1213) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2461 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2462 = "mhlo.broadcast_in_dim"(%2461) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2463 = mhlo.multiply %2460, %2462 : tensor<64xf32>
    %2464 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2465 = mhlo.subtract %2464, %2461 : tensor<f32>
    %2466 = "mhlo.broadcast_in_dim"(%2465) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2467 = mhlo.multiply %arg592, %2466 : tensor<64xf32>
    %2468 = mhlo.add %2463, %2467 : tensor<64xf32>
    %2469 = "mhlo.get_tuple_element"(%139) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2470 = call @aten.view.3422(%2469) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2471 = "mhlo.get_tuple_element"(%139) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2472 = call @aten.view.3422(%2471) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2473 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2474 = "mhlo.broadcast_in_dim"(%2473) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2475 = mhlo.multiply %2177, %2474 : tensor<64xf32>
    %2476 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2477 = mhlo.subtract %2476, %2473 : tensor<f32>
    %2478 = "mhlo.broadcast_in_dim"(%2477) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2479 = mhlo.multiply %arg594, %2478 : tensor<64xf32>
    %2480 = mhlo.add %2475, %2479 : tensor<64xf32>
    %2481 = "mhlo.get_tuple_element"(%1217) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2482 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2483 = "mhlo.broadcast_in_dim"(%2482) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2484 = mhlo.multiply %2481, %2483 : tensor<64xf32>
    %2485 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2486 = mhlo.subtract %2485, %2482 : tensor<f32>
    %2487 = "mhlo.broadcast_in_dim"(%2486) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2488 = mhlo.multiply %arg595, %2487 : tensor<64xf32>
    %2489 = mhlo.add %2484, %2488 : tensor<64xf32>
    %2490 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2491 = "mhlo.broadcast_in_dim"(%2490) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2492 = mhlo.multiply %2179, %2491 : tensor<256xf32>
    %2493 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2494 = mhlo.subtract %2493, %2490 : tensor<f32>
    %2495 = "mhlo.broadcast_in_dim"(%2494) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2496 = mhlo.multiply %arg597, %2495 : tensor<256xf32>
    %2497 = mhlo.add %2492, %2496 : tensor<256xf32>
    %2498 = "mhlo.get_tuple_element"(%1221) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2499 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2500 = "mhlo.broadcast_in_dim"(%2499) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2501 = mhlo.multiply %2498, %2500 : tensor<256xf32>
    %2502 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2503 = mhlo.subtract %2502, %2499 : tensor<f32>
    %2504 = "mhlo.broadcast_in_dim"(%2503) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2505 = mhlo.multiply %arg598, %2504 : tensor<256xf32>
    %2506 = mhlo.add %2501, %2505 : tensor<256xf32>
    %2507 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2508 = "mhlo.broadcast_in_dim"(%2507) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2509 = mhlo.multiply %2181, %2508 : tensor<64xf32>
    %2510 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2511 = mhlo.subtract %2510, %2507 : tensor<f32>
    %2512 = "mhlo.broadcast_in_dim"(%2511) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2513 = mhlo.multiply %arg600, %2512 : tensor<64xf32>
    %2514 = mhlo.add %2509, %2513 : tensor<64xf32>
    %2515 = "mhlo.get_tuple_element"(%1229) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2516 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2517 = "mhlo.broadcast_in_dim"(%2516) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2518 = mhlo.multiply %2515, %2517 : tensor<64xf32>
    %2519 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2520 = mhlo.subtract %2519, %2516 : tensor<f32>
    %2521 = "mhlo.broadcast_in_dim"(%2520) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2522 = mhlo.multiply %arg601, %2521 : tensor<64xf32>
    %2523 = mhlo.add %2518, %2522 : tensor<64xf32>
    %2524 = "mhlo.get_tuple_element"(%394) {index = 1 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2525 = call @aten.view.3422(%2524) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2526 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2527 = "mhlo.broadcast_in_dim"(%2526) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2528 = mhlo.multiply %1583, %2527 : tensor<64xf32>
    %2529 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2530 = mhlo.subtract %2529, %2526 : tensor<f32>
    %2531 = "mhlo.broadcast_in_dim"(%2530) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2532 = mhlo.multiply %arg603, %2531 : tensor<64xf32>
    %2533 = mhlo.add %2528, %2532 : tensor<64xf32>
    %2534 = "mhlo.get_tuple_element"(%394) {index = 3 : i32} : (tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>) -> tensor<480xf32>
    %2535 = call @aten.view.3422(%2534) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    %2536 = "mhlo.get_tuple_element"(%1233) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %2537 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2538 = "mhlo.broadcast_in_dim"(%2537) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2539 = mhlo.multiply %2536, %2538 : tensor<64xf32>
    %2540 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2541 = mhlo.subtract %2540, %2537 : tensor<f32>
    %2542 = "mhlo.broadcast_in_dim"(%2541) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2543 = mhlo.multiply %arg604, %2542 : tensor<64xf32>
    %2544 = mhlo.add %2539, %2543 : tensor<64xf32>
    %2545 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2546 = "mhlo.broadcast_in_dim"(%2545) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2547 = mhlo.multiply %1582, %2546 : tensor<256xf32>
    %2548 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2549 = mhlo.subtract %2548, %2545 : tensor<f32>
    %2550 = "mhlo.broadcast_in_dim"(%2549) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2551 = mhlo.multiply %arg606, %2550 : tensor<256xf32>
    %2552 = mhlo.add %2547, %2551 : tensor<256xf32>
    %2553 = "mhlo.get_tuple_element"(%1237) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %2554 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2555 = "mhlo.broadcast_in_dim"(%2554) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2556 = mhlo.multiply %2553, %2555 : tensor<256xf32>
    %2557 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2558 = mhlo.subtract %2557, %2554 : tensor<f32>
    %2559 = "mhlo.broadcast_in_dim"(%2558) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2560 = mhlo.multiply %arg607, %2559 : tensor<256xf32>
    %2561 = mhlo.add %2556, %2560 : tensor<256xf32>
    %2562 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2563 = "mhlo.broadcast_in_dim"(%2562) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2564 = mhlo.multiply %2125, %2563 : tensor<128xf32>
    %2565 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2566 = mhlo.subtract %2565, %2562 : tensor<f32>
    %2567 = "mhlo.broadcast_in_dim"(%2566) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2568 = mhlo.multiply %arg609, %2567 : tensor<128xf32>
    %2569 = mhlo.add %2564, %2568 : tensor<128xf32>
    %2570 = "mhlo.get_tuple_element"(%1248) {index = 2 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %2571 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2572 = "mhlo.broadcast_in_dim"(%2571) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2573 = mhlo.multiply %2570, %2572 : tensor<128xf32>
    %2574 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2575 = mhlo.subtract %2574, %2571 : tensor<f32>
    %2576 = "mhlo.broadcast_in_dim"(%2575) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2577 = mhlo.multiply %arg610, %2576 : tensor<128xf32>
    %2578 = mhlo.add %2573, %2577 : tensor<128xf32>
    %2579 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2580 = "mhlo.broadcast_in_dim"(%2579) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2581 = mhlo.multiply %2127, %2580 : tensor<128xf32>
    %2582 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2583 = mhlo.subtract %2582, %2579 : tensor<f32>
    %2584 = "mhlo.broadcast_in_dim"(%2583) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2585 = mhlo.multiply %arg612, %2584 : tensor<128xf32>
    %2586 = mhlo.add %2581, %2585 : tensor<128xf32>
    %2587 = "mhlo.get_tuple_element"(%1252) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %2588 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2589 = "mhlo.broadcast_in_dim"(%2588) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2590 = mhlo.multiply %2587, %2589 : tensor<128xf32>
    %2591 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2592 = mhlo.subtract %2591, %2588 : tensor<f32>
    %2593 = "mhlo.broadcast_in_dim"(%2592) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2594 = mhlo.multiply %arg613, %2593 : tensor<128xf32>
    %2595 = mhlo.add %2590, %2594 : tensor<128xf32>
    %2596 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2597 = "mhlo.broadcast_in_dim"(%2596) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2598 = mhlo.multiply %2129, %2597 : tensor<512xf32>
    %2599 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2600 = mhlo.subtract %2599, %2596 : tensor<f32>
    %2601 = "mhlo.broadcast_in_dim"(%2600) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2602 = mhlo.multiply %arg615, %2601 : tensor<512xf32>
    %2603 = mhlo.add %2598, %2602 : tensor<512xf32>
    %2604 = "mhlo.get_tuple_element"(%1256) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %2605 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %2606 = "mhlo.broadcast_in_dim"(%2605) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2607 = mhlo.multiply %2604, %2606 : tensor<512xf32>
    %2608 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2609 = mhlo.subtract %2608, %2605 : tensor<f32>
    %2610 = "mhlo.broadcast_in_dim"(%2609) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2611 = mhlo.multiply %arg616, %2610 : tensor<512xf32>
    %2612 = mhlo.add %2607, %2611 : tensor<512xf32>
    %2613 = "mhlo.tuple"(%1481, %1486, %1491, %1496, %1498, %1498, %1503, %1503, %arg741, %1452, %1124, %1504, %1505, %1137, %1460, %1463, %1476, %1482, %1163, %1507, %1509, %1484, %1162, %1410, %1510, %1511, %1029, %1393, %1420, %1042, %1402, %1512, %1513, %1068, %1397, %1515, %1517, %1070, %1067, %1518, %1519, %1401, %1073, %1072, %1079, %arg473, %arg474, %arg441, %973, %arg442, %1521, %1523, %arg445, %1524, %arg446, %1525, %975, %972, %1428, %978, %1445, %1433, %arg457, %977, %arg458, %1526, %1527, %984, %1432, %arg461, %arg462, %1529, %1531, %1003, %1436, %1002, %1532, %1011, %1533, %1535, %1537, %307, %439, %310, %arg521, %arg522, %313, %312, %arg525, %arg526, %319, %1539, %1541, %338, %arg537, %337, %arg538, %378, %346, %arg541, %arg542, %345, %880, %5, %1543, %1545, %813, %1267, %932, %812, %883, %638, %820, %859, %882, %853, %821, %889, %1547, %1549, %841, %908, %907, %915, %663, %289, %840, %916, %662, %657, %647, %649, %935, %752, %936, %1009, %954, %828, %819, %847, %1122, %1550, %1551, %758, %746, %1329, %1552, %1553, %1298, %745, %1350, %1554, %1555, %arg20, %1333, %1556, %1557, %1302, %733, %1346, %1311, %1338, %1558, %1559, %1337, %1314, %1560, %1561, %764, %1310, %1562, %arg19, %1563, %arg22, %1370, %1564, %1361, %1565, %1266, %554, %1299, %1378, %1566, %1567, %1568, %1365, %1569, %1270, %567, %1279, %1398, %1570, %593, %1571, %1287, %1573, %1574, %1369, %1576, %1577, %1278, %592, %595, %arg82, %1579, %arg79, %arg83, %1581, %arg80, %arg85, %283, %arg86, %242, %arg88, %arg89, %arg91, %251, %arg92, %arg94, %250, %arg95, %arg97, %arg98, %271, %arg100, %arg101, %arg113, %270, %arg103, %arg104, %arg106, %arg107, %arg109, %457, %arg110, %258, %arg112, %77, %1236, %46, %1582, %1583, %267, %1584, %1235, %55, %1585, %87, %1243, %172, %99, %1255, %1586, %1587, %498, %1589, %1591, %500, %497, %629, %503, %502, %742, %509, %1593, %1595, %528, %527, %574, %243, %163, %194, %174, %182, %93, %1604, %1613, %48, %1485, %1621, %1630, %1638, %1647, %1477, %1655, %1664, %1479, %1673, %1682, %39, %1691, %47, %1700, %1480, %1709, %1718, %1726, %1487, %1735, %1744, %1753, %81, %1489, %1761, %1770, %80, %1778, %1787, %1490, %1796, %1805, %1814, %1822, %1830, %1839, %1847, %1856, %1864, %1873, %1882, %1891, %1900, %1909, %1918, %1927, %1935, %1944, %1952, %1961, %1969, %1978, %1987, %1996, %2005, %2014, %407, %2015, %2016, %2018, %1203, %2020, %433, %432, %1228, %441, %2021, %2022, %440, %1211, %461, %479, %1220, %460, %2023, %2024, %1215, %562, %1449, %1788, %2025, %1444, %1605, %2026, %1448, %1453, %2027, %2028, %1386, %2029, %2037, %744, %1665, %2038, %2039, %366, %1413, %1377, %1409, %arg553, %365, %arg554, %1421, %757, %852, %1394, %arg557, %1429, %arg558, %2040, %839, %2041, %1736, %1437, %783, %1381, %2042, %2044, %2046, %785, %782, %1683, %353, %2047, %1416, %arg569, %arg570, %2048, %2049, %1385, %878, %788, %2051, %2060, %787, %362, %2062, %2071, %384, %1701, %2080, %877, %914, %794, %2081, %2090, %1417, %arg149, %1165, %arg115, %arg116, %79, %arg118, %arg119, %1168, %arg121, %arg409, %1167, %arg122, %arg410, %arg124, %arg125, %arg413, %1174, %arg414, %arg127, %2092, %arg128, %92, %2094, %arg130, %arg131, %1472, %arg133, %118, %arg134, %2096, %arg136, %1473, %2098, %arg137, %arg425, %arg426, %1186, %arg139, %117, %arg140, %arg429, %1475, %arg142, %arg430, %arg143, %120, %arg145, %arg146, %arg148, %arg509, %1010, %1492, %arg510, %2100, %2102, %arg477, %1031, %1098, %arg478, %1097, %1494, %1043, %1105, %1144, %923, %1030, %1138, %1106, %1495, %arg489, %947, %1126, %934, %arg490, %arg493, %948, %1018, %arg494, %1125, %942, %arg505, %1049, %1037, %arg506, %1113, %1104, %1132, %2103, %2104, %688, %1345, %2106, %2108, %2109, %2110, %690, %687, %692, %1330, %1362, %1979, %2111, %2112, %2113, %1317, %1349, %693, %1334, %699, %2115, %1997, %1354, %2116, %2118, %2119, %2120, %718, %1321, %1353, %717, %1322, %1366, %726, %1382, %725, %arg49, %arg46, %598, %arg50, %arg47, %597, %arg52, %arg53, %604, %arg55, %arg56, %2122, %arg58, %2124, %arg59, %623, %arg61, %622, %arg62, %630, %669, %arg64, %arg65, %631, %arg67, %arg68, %arg70, %arg71, %651, %arg73, %arg74, %arg76, %650, %534, %arg77, %1027, %837, %arg26, %arg44, %277, %arg43, %269, %372, %arg41, %arg40, %arg38, %arg37, %arg35, %arg34, %282, %arg32, %arg28, %arg31, %arg29, %308, %568, %536, %2125, %2126, %535, %1250, %556, %1283, %555, %2127, %2128, %1254, %arg25, %arg23, %1263, %1271, %2129, %543, %552, %2130, %1262, %2139, %2147, %1874, %2156, %2157, %1282, %2165, %2174, %arg738, %arg740, %1892, %2175, %1286, %1295, %1910, %2176, %1303, %1294, %1247, %2177, %2178, %1219, %1232, %2179, %2180, %1227, %1231, %1251, %1244, %2181, %249, %2182, %2191, %2199, %187, %2207, %2216, %2224, %2233, %2241, %2250, %213, %2258, %2267, %2269, %2271, %2279, %2288, %2296, %212, %2305, %215, %2313, %2322, %2330, %2339, %218, %2347, %2356, %217, %2364, %2373, %344, %2381, %224, %2390, %448, %472, %473, %459, %467, %2052, %2391, %1204, %1189, %1193, %1192, %1196, %2392, %2393, %1216, %1212, %1200, %2072, %2394, %1199, %2396, %40, %154, %21, %724, %2398, %2400, %2409, %2417, %123, %364, %2425, %2434, %122, %2442, %2451, %129, %2459, %arg376, %2468, %2470, %arg377, %2472, %2480, %148, %2489, %377, %arg381, %2497, %147, %arg382, %2506, %403, %155, %2514, %2523, %156, %2525, %2533, %2535, %2544, %2552, %402, %2561, %arg393, %2569, %176, %405, %arg394, %2578, %414, %2586, %arg397, %2595, %188, %arg398, %1318, %2603, %175, %408, %2612, %arg151, %arg152, %arg154, %arg155, %arg157, %arg158, %arg160, %arg161, %arg163, %arg164, %arg166, %arg167, %arg169, %arg170, %arg172, %arg173, %arg175, %arg176) {xla_shape = "(f32[2,19]{1,0}, f32[2,19]{1,0}, f32[2,2]{1,0}, f32[2,2]{1,0}, s64[19]{0}, /*index=5*/s64[19]{0}, s64[19]{0}, s64[19]{0}, f32[0]{0}, f32[16,512,8,8]{3,2,1,0}, /*index=10*/f32[24,240,240]{2,1,0}, f32[2048]{0}, f32[2048]{0}, f32[480,768]{1,0}, f32[16,2048,8,8]{3,2,1,0}, /*index=15*/f32[2,2048]{1,0}, f32[2,2816]{1,0}, f32[2816,128]{0,1}, f32[768,3072]{0,1}, f32[2,240,1]{2,1,0}, /*index=20*/f32[2,240,1]{2,1,0}, f32[2,128]{1,0}, f32[480,768]{1,0}, f32[16,2048,8,8]{3,2,1,0}, f32[1024]{0}, /*index=25*/f32[1024]{0}, f32[24,240,240]{2,1,0}, f32[16,1024,16,16]{3,2,1,0}, f32[16,512,8,8]{3,2,1,0}, f32[480,768]{1,0}, /*index=30*/f32[16,1024,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[768,3072]{0,1}, f32[16,256,16,16]{3,2,1,0}, /*index=35*/f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, f32[2,240,3072]{2,1,0}, f32[480,768]{1,0}, f32[256]{0}, /*index=40*/f32[256]{0}, f32[16,256,16,16]{3,2,1,0}, f32[3072,768]{0,1}, f32[480,3072]{1,0}, f32[2,240,768]{2,1,0}, /*index=45*/f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,3072]{0,1}, f32[768]{0}, /*index=50*/f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, f32[768]{0}, f32[2048]{0}, f32[768]{0}, /*index=55*/f32[2048]{0}, f32[2,240,3072]{2,1,0}, f32[480,768]{1,0}, f32[16,2048,8,8]{3,2,1,0}, f32[3072,768]{0,1}, /*index=60*/f32[16,512,8,8]{3,2,1,0}, f32[16,512,8,8]{3,2,1,0}, f32[768]{0}, f32[480,3072]{1,0}, f32[768]{0}, /*index=65*/f32[512]{0}, f32[512]{0}, f32[2,240,768]{2,1,0}, f32[16,512,8,8]{3,2,1,0}, f32[768]{0}, /*index=70*/f32[768]{0}, f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, f32[768,768]{0,1}, f32[16,512,8,8]{3,2,1,0}, /*index=75*/f32[480,768]{1,0}, f32[512]{0}, f32[768,768]{0,1}, f32[512]{0}, f32[2,240,1]{2,1,0}, /*index=80*/f32[2,240,1]{2,1,0}, f32[480,768]{1,0}, f32[24,240,64]{2,1,0}, f32[2,240,3072]{2,1,0}, f32[768]{0}, /*index=85*/f32[768]{0}, f32[3072,768]{0,1}, f32[480,3072]{1,0}, f32[768]{0}, f32[768]{0}, /*index=90*/f32[2,240,768]{2,1,0}, f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, f32[768,768]{0,1}, f32[768]{0}, /*index=95*/f32[480,768]{1,0}, f32[768]{0}, f32[768,768]{0,1}, f32[768,768]{0,1}, f32[768]{0}, /*index=100*/f32[768]{0}, f32[480,768]{1,0}, f32[2,240,3072]{2,1,0}, s64[2,240]{1,0}, f32[2,240,1]{2,1,0}, /*index=105*/f32[2,240,1]{2,1,0}, f32[768,768]{0,1}, f32[16,128,32,32]{3,2,1,0}, f32[2,12,240,240]{3,2,1,0}, f32[480,768]{1,0}, /*index=110*/f32[3072,768]{0,1}, f32[24,64,240]{2,1,0}, f32[480,768]{1,0}, f32[2,240,768]{2,1,0}, f32[480,3072]{1,0}, /*index=115*/f32[768,768]{0,1}, f32[768,768]{0,1}, f32[2,240,768]{2,1,0}, f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, /*index=120*/f32[768,768]{0,1}, f32[768,768]{0,1}, f32[480,768]{1,0}, f32[480,768]{1,0}, f32[768,768]{0,1}, /*index=125*/f32[2,240,768]{2,1,0}, f32[480,768]{1,0}, f32[768,768]{0,1}, f32[480,768]{1,0}, f32[24,240,64]{2,1,0}, /*index=130*/f32[2,12,240,240]{3,2,1,0}, f32[24,240,240]{2,1,0}, f32[480,768]{1,0}, f32[24,240,64]{2,1,0}, f32[768,768]{0,1}, /*index=135*/f32[24,240,64]{2,1,0}, f32[2,240,768]{2,1,0}, f32[24,64,240]{2,1,0}, f32[24,240,64]{2,1,0}, f32[24,240,64]{2,1,0}, /*index=140*/f32[2,12,240,240]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[768,768]{0,1}, f32[768,768]{0,1}, /*index=145*/f32[16,1024,16,16]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[16,128,32,32]{3,2,1,0}, f32[480,768]{1,0}, /*index=150*/f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[64,3,7,7]{3,2,1,0}, f32[16,256,16,16]{3,2,1,0}, /*index=155*/f32[128]{0}, f32[128]{0}, f32[16,128,32,32]{3,2,1,0}, f32[24,64,240]{2,1,0}, f32[16,256,16,16]{3,2,1,0}, /*index=160*/f32[16,1024,16,16]{3,2,1,0}, f32[16,1024,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[16,256,16,16]{3,2,1,0}, /*index=165*/f32[16,256,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[2,240,768]{2,1,0}, f32[16,512,32,32]{3,2,1,0}, /*index=170*/f32[1024]{0}, f32[64]{0}, f32[1024]{0}, f32[64]{0}, f32[16,1024,16,16]{3,2,1,0}, /*index=175*/f32[128]{0}, f32[16,1024,16,16]{3,2,1,0}, f32[128]{0}, f32[16,128,32,32]{3,2,1,0}, f32[24,240,240]{2,1,0}, /*index=180*/f32[16,128,32,32]{3,2,1,0}, f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[128]{0}, /*index=185*/f32[16,256,16,16]{3,2,1,0}, f32[128]{0}, f32[16,128,32,32]{3,2,1,0}, f32[480,768]{1,0}, f32[16,128,32,32]{3,2,1,0}, /*index=190*/f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[768,3072]{0,1}, f32[256]{0}, f32[16,512,32,32]{3,2,1,0}, /*index=195*/f32[2,240,1]{2,1,0}, f32[512]{0}, f32[16,256,16,16]{3,2,1,0}, f32[2,240,1]{2,1,0}, f32[512]{0}, /*index=200*/f32[16,512,32,32]{3,2,1,0}, f32[480,768]{1,0}, f32[2,240,3072]{2,1,0}, f32[128]{0}, f32[2,240,1]{2,1,0}, /*index=205*/f32[512]{0}, f32[128,512,1,1]{3,2,1,0}, f32[2,240,1]{2,1,0}, f32[512,128,1,1]{3,2,1,0}, f32[128]{0}, /*index=210*/f32[768,768]{0,1}, f32[128,128,3,3]{3,2,1,0}, f32[480,768]{1,0}, f32[512]{0}, f32[512,128,1,1]{3,2,1,0}, /*index=215*/f32[256]{0}, f32[768,768]{0,1}, f32[256,512,1,1]{3,2,1,0}, f32[256]{0}, f32[480,768]{1,0}, /*index=220*/f32[256,256,3,3]{3,2,1,0}, f32[1024]{0}, f32[1024,256,1,1]{3,2,1,0}, f32[768,768]{0,1}, f32[1024]{0}, /*index=225*/f32[1024,512,1,1]{3,2,1,0}, f32[256,1024,1,1]{3,2,1,0}, f32[480,768]{1,0}, f32[256]{0}, f32[256,1024,1,1]{3,2,1,0}, /*index=230*/f32[256]{0}, f32[256,256,3,3]{3,2,1,0}, f32[1024]{0}, f32[2,12,240,240]{3,2,1,0}, f32[1024,256,1,1]{3,2,1,0}, /*index=235*/f32[24,64,240]{2,1,0}, f32[256]{0}, f32[2,12,240,240]{3,2,1,0}, f32[16,256,64,64]{3,2,1,0}, f32[24,240,64]{2,1,0}, /*index=240*/f32[256]{0}, f32[64]{0}, f32[2,12,240,240]{3,2,1,0}, f32[64]{0}, f32[16,64,64,64]{3,2,1,0}, /*index=245*/f32[24,64,240]{2,1,0}, f32[256]{0}, f32[24,240,64]{2,1,0}, f32[16,256,64,64]{3,2,1,0}, f32[2,12,240,240]{3,2,1,0}, /*index=250*/f32[2,240,768]{2,1,0}, f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[768,3072]{0,1}, /*index=255*/f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, f32[2,240,3072]{2,1,0}, f32[480,768]{1,0}, f32[24,240,64]{2,1,0}, /*index=260*/f32[3072,768]{0,1}, f32[480,3072]{1,0}, f32[2,12,240,240]{3,2,1,0}, f32[2,240,768]{2,1,0}, f32[2,240,1]{2,1,0}, /*index=265*/f32[2,240,1]{2,1,0}, f32[768,768]{0,1}, f32[480,768]{1,0}, f32[2,240,768]{2,1,0}, f32[768,768]{0,1}, /*index=270*/f32[24,64,240]{2,1,0}, f32[2,240,768]{2,1,0}, f32[24,240,240]{2,1,0}, f32[24,240,64]{2,1,0}, f32[768,768]{0,1}, /*index=275*/f32[256]{0}, f32[512]{0}, f32[768,768]{0,1}, f32[128,19]{0,1}, f32[1024]{0}, /*index=280*/f32[1024]{0}, f32[256]{0}, f32[256]{0}, f32[2816,256]{0,1}, f32[256]{0}, /*index=285*/f32[256]{0}, f32[2,256]{1,0}, f32[1024]{0}, f32[1024]{0}, f32[480,768]{1,0}, /*index=290*/f32[512]{0}, f32[480,768]{1,0}, f32[512]{0}, f32[256,19]{0,1}, f32[512]{0}, /*index=295*/f32[512]{0}, f32[2048]{0}, f32[2048,256]{0,1}, f32[2048]{0}, f32[2048]{0}, /*index=300*/f32[2048]{0}, f32[768,768]{0,1}, f32[2,256]{1,0}, f32[512]{0}, f32[512]{0}, /*index=305*/f32[480,768]{1,0}, f32[512]{0}, f32[512]{0}, f32[256,2]{0,1}, f32[2048]{0}, /*index=310*/f32[2048]{0}, f32[512]{0}, f32[1024]{0}, f32[128]{0}, f32[128]{0}, /*index=315*/f32[128]{0}, f32[128]{0}, f32[512]{0}, f32[512]{0}, f32[128]{0}, /*index=320*/f32[128]{0}, f32[128]{0}, f32[128]{0}, f32[512]{0}, f32[512]{0}, /*index=325*/f32[128]{0}, f32[128]{0}, f32[128]{0}, f32[128]{0}, f32[512]{0}, /*index=330*/f32[512]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, /*index=335*/f32[480,3072]{1,0}, f32[64]{0}, f32[64]{0}, f32[2,240,1]{2,1,0}, f32[16,64,64,64]{3,2,1,0}, /*index=340*/f32[2,240,1]{2,1,0}, f32[768,768]{0,1}, f32[480,768]{1,0}, f32[16,64,64,64]{3,2,1,0}, f32[768,768]{0,1}, /*index=345*/f32[256]{0}, f32[256]{0}, f32[480,768]{1,0}, f32[16,256,64,64]{3,2,1,0}, f32[768,768]{0,1}, /*index=350*/f32[2,240,768]{2,1,0}, f32[16,256,64,64]{3,2,1,0}, f32[480,768]{1,0}, f32[64]{0}, f32[64]{0}, /*index=355*/f32[16,64,64,64]{3,2,1,0}, f32[24,240,64]{2,1,0}, f32[16,512,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0}, /*index=360*/f32[16,2048,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[16,512,8,8]{3,2,1,0}, f32[16,2048,8,8]{3,2,1,0}, /*index=365*/f32[512]{0}, f32[512]{0}, f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[64]{0}, /*index=370*/f32[24,240,240]{2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0}, f32[768,768]{0,1}, /*index=375*/f32[16,512,16,16]{3,2,1,0}, f32[16,1024,16,16]{3,2,1,0}, f32[16,1024,16,16]{3,2,1,0}, f32[768]{0}, f32[480,768]{1,0}, /*index=380*/f32[768]{0}, f32[16,2048,8,8]{3,2,1,0}, f32[480,768]{1,0}, f32[480,768]{1,0}, f32[16,256,16,16]{3,2,1,0}, /*index=385*/f32[768]{0}, f32[16,512,8,8]{3,2,1,0}, f32[768]{0}, f32[256]{0}, f32[24,240,240]{2,1,0}, /*index=390*/f32[256]{0}, f32[2048]{0}, f32[16,2048,8,8]{3,2,1,0}, f32[768,3072]{0,1}, f32[16,256,16,16]{3,2,1,0}, /*index=395*/f32[2048]{0}, f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, f32[2,240,3072]{2,1,0}, f32[480,768]{1,0}, /*index=400*/f32[512]{0}, f32[24,64,240]{2,1,0}, f32[512]{0}, f32[16,512,16,16]{3,2,1,0}, f32[768]{0}, /*index=405*/f32[768]{0}, f32[256]{0}, f32[256]{0}, f32[16,256,16,16]{3,2,1,0}, f32[768,3072]{0,1}, /*index=410*/f32[3072,768]{0,1}, f32[2,240,1]{2,1,0}, f32[64]{0}, f32[480,3072]{1,0}, f32[2,12,240,240]{3,2,1,0}, /*index=415*/f32[2,240,1]{2,1,0}, f32[64]{0}, f32[2,240,768]{2,1,0}, f32[512]{0}, f32[64]{0}, /*index=420*/f32[480,768]{1,0}, f32[24,240,64]{2,1,0}, f32[2,240,768]{2,1,0}, f32[512]{0}, f32[64]{0}, /*index=425*/f32[16,512,8,8]{3,2,1,0}, f32[512,1024,1,1]{3,2,1,0}, f32[2,240,3072]{2,1,0}, f32[256]{0}, f32[256,256,3,3]{3,2,1,0}, /*index=430*/f32[24,240,240]{2,1,0}, f32[1024]{0}, f32[1024,256,1,1]{3,2,1,0}, f32[3072,768]{0,1}, f32[256]{0}, /*index=435*/f32[768]{0}, f32[480,3072]{1,0}, f32[256,1024,1,1]{3,2,1,0}, f32[768]{0}, f32[256]{0}, /*index=440*/f32[256,256,3,3]{3,2,1,0}, f32[768]{0}, f32[2,240,768]{2,1,0}, f32[768]{0}, f32[1024]{0}, /*index=445*/f32[2,240,1]{2,1,0}, f32[1024,256,1,1]{3,2,1,0}, f32[480,768]{1,0}, f32[2,240,1]{2,1,0}, f32[256]{0}, /*index=450*/f32[256,1024,1,1]{3,2,1,0}, f32[2,768]{1,0}, f32[256]{0}, f32[768,3072]{0,1}, f32[256,256,3,3]{3,2,1,0}, /*index=455*/f32[2,240,1]{2,1,0}, f32[1024]{0}, f32[768,768]{0,1}, f32[2,240,1]{2,1,0}, f32[1024,256,1,1]{3,2,1,0}, /*index=460*/f32[768]{0}, f32[768]{0}, f32[16,64,128,128]{3,2,1,0}, f32[256]{0}, f32[480,768]{1,0}, /*index=465*/f32[256,1024,1,1]{3,2,1,0}, f32[768]{0}, f32[2,768]{1,0}, f32[256]{0}, f32[768]{0}, /*index=470*/f32[256,256,3,3]{3,2,1,0}, f32[2,240,3072]{2,1,0}, f32[1024]{0}, f32[1024,256,1,1]{3,2,1,0}, f32[512]{0}, /*index=475*/f32[768]{0}, f32[480,768]{1,0}, f32[768,256]{0,1}, f32[768]{0}, f32[2,240,1]{2,1,0}, /*index=480*/f32[2,240,1]{2,1,0}, f32[768]{0}, f32[768,768]{0,1}, f32[768,768]{0,1}, f32[768]{0}, /*index=485*/f32[480,768]{1,0}, f32[2,256]{1,0}, f32[768,768]{0,1}, f32[480,768]{1,0}, f32[2,240,768]{2,1,0}, /*index=490*/f32[24,64,240]{2,1,0}, f32[480,768]{1,0}, f32[768,768]{0,1}, f32[768,768]{0,1}, f32[256,2]{0,1}, /*index=495*/f32[768]{0}, f32[480,768]{1,0}, f32[768,768]{0,1}, f32[24,240,240]{2,1,0}, f32[768]{0}, /*index=500*/f32[768]{0}, f32[768,768]{0,1}, f32[24,64,240]{2,1,0}, f32[768]{0}, f32[480,768]{1,0}, /*index=505*/f32[24,240,64]{2,1,0}, f32[768]{0}, f32[2,240,768]{2,1,0}, f32[24,240,64]{2,1,0}, f32[768]{0}, /*index=510*/f32[24,64,240]{2,1,0}, f32[24,240,64]{2,1,0}, f32[24,240,64]{2,1,0}, f32[1024]{0}, f32[1024]{0}, /*index=515*/f32[768,3072]{0,1}, f32[16,1024,16,16]{3,2,1,0}, f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, f32[1024]{0}, /*index=520*/f32[1024]{0}, f32[2,240,3072]{2,1,0}, f32[480,768]{1,0}, f32[480,3072]{1,0}, f32[16,256,16,16]{3,2,1,0}, /*index=525*/f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, /*index=530*/f32[16,256,32,32]{3,2,1,0}, f32[16,256,16,16]{3,2,1,0}, f32[3072,768]{0,1}, f32[16,256,16,16]{3,2,1,0}, f32[2,240,768]{2,1,0}, /*index=535*/f32[2,240,1]{2,1,0}, f32[256]{0}, f32[16,1024,16,16]{3,2,1,0}, f32[256]{0}, f32[2,240,1]{2,1,0}, /*index=540*/f32[256]{0}, f32[256]{0}, f32[768,768]{0,1}, f32[16,256,16,16]{3,2,1,0}, f32[16,256,16,16]{3,2,1,0}, /*index=545*/f32[480,768]{1,0}, f32[16,1024,16,16]{3,2,1,0}, f32[16,256,16,16]{3,2,1,0}, f32[768,768]{0,1}, f32[16,256,16,16]{3,2,1,0}, /*index=550*/f32[480,768]{1,0}, f32[256]{0}, f32[64]{0}, f32[3072,768]{0,1}, f32[256,64,1,1]{3,2,1,0}, /*index=555*/f32[64,64,3,3]{3,2,1,0}, f32[480,3072]{1,0}, f32[128]{0}, f32[128,256,1,1]{3,2,1,0}, f32[2,240,768]{2,1,0}, /*index=560*/f32[128]{0}, f32[128,128,3,3]{3,2,1,0}, f32[2,240,1]{2,1,0}, f32[512]{0}, f32[2,240,1]{2,1,0}, /*index=565*/f32[512,128,1,1]{3,2,1,0}, f32[768,768]{0,1}, f32[512]{0}, f32[480,768]{1,0}, f32[512,256,1,1]{3,2,1,0}, /*index=570*/f32[480,768]{1,0}, f32[2,240,768]{2,1,0}, f32[128]{0}, f32[128,512,1,1]{3,2,1,0}, f32[768,768]{0,1}, /*index=575*/f32[128]{0}, f32[128,128,3,3]{3,2,1,0}, f32[512]{0}, f32[512,128,1,1]{3,2,1,0}, f32[768,768]{0,1}, /*index=580*/f32[128]{0}, f32[128,512,1,1]{3,2,1,0}, f32[128]{0}, f32[480,768]{1,0}, f32[24,240,64]{2,1,0}, /*index=585*/f32[128,128,3,3]{3,2,1,0}, f32[2,12,240,240]{3,2,1,0}, f32[2,12,240,240]{3,2,1,0}, f32[64,64,3,3]{3,2,1,0}, f32[64,256,1,1]{3,2,1,0}, /*index=590*/f32[24,240,64]{2,1,0}, f32[64]{0}, f32[24,240,240]{2,1,0}, f32[24,240,64]{2,1,0}, f32[256,64,1,1]{3,2,1,0}, /*index=595*/f32[256]{0}, f32[64,64,3,3]{3,2,1,0}, f32[64]{0}, f32[64,256,1,1]{3,2,1,0}, f32[64]{0}, /*index=600*/f32[480,768]{1,0}, f32[256,64,1,1]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256,64,1,1]{3,2,1,0}, /*index=605*/f32[768,3072]{0,1}, f32[768,768]{0,1}, f32[768,768]{0,1}, f32[128]{0}, f32[128]{0}, /*index=610*/f32[480,768]{1,0}, f32[16,128,64,64]{3,2,1,0}, f32[768,768]{0,1}, f32[16,128,32,32]{3,2,1,0}, f32[480,768]{1,0}, /*index=615*/f32[128]{0}, f32[128]{0}, f32[16,128,32,32]{3,2,1,0}, f32[64]{0}, f32[64,64,1,1]{3,2,1,0}, /*index=620*/f32[16,128,32,32]{3,2,1,0}, f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[24,64,240]{2,1,0}, f32[2,12,240,240]{3,2,1,0}, /*index=625*/f32[512]{0}, f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[128]{0}, /*index=630*/f32[512]{0}, f32[128]{0}, f32[16,128,32,32]{3,2,1,0}, f32[2048]{0}, f32[2048]{0}, /*index=635*/f32[16,3,256,256]{3,2,1,0}, s64[2,240]{1,0}, f32[128]{0}, f32[128]{0}, f32[16,128,32,32]{3,2,1,0}, /*index=640*/f32[16,128,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[16,512,32,32]{3,2,1,0}, f32[16,512,32,32]{3,2,1,0}, /*index=645*/f32[16,128,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[16,64,64,64]{3,2,1,0}, f32[16,64,64,64]{3,2,1,0}, /*index=650*/f32[256]{0}, f32[256]{0}, f32[16,256,64,64]{3,2,1,0}, f32[16,64,64,64]{3,2,1,0}, f32[16,128,32,32]{3,2,1,0}, /*index=655*/f32[16,512,32,32]{3,2,1,0}, f32[64]{0}, f32[24,240,64]{2,1,0}, f32[64]{0}, f32[1024]{0}, /*index=660*/f32[256]{0}, f32[480,768]{1,0}, f32[1024]{0}, f32[1024]{0}, f32[256]{0}, /*index=665*/f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[768,3072]{0,1}, f32[1024]{0}, /*index=670*/f32[1024]{0}, f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, f32[256]{0}, f32[256]{0}, /*index=675*/f32[256]{0}, f32[480,768]{1,0}, f32[256]{0}, f32[2,240,3072]{2,1,0}, f32[1024]{0}, /*index=680*/f32[1024]{0}, f32[256]{0}, f32[256]{0}, f32[3072,768]{0,1}, f32[256]{0}, /*index=685*/f32[256]{0}, f32[480,3072]{1,0}, f32[1024]{0}, f32[1024]{0}, f32[24,240,64]{2,1,0}, /*index=690*/f32[256]{0}, f32[2,240,768]{2,1,0}, f32[256]{0}, f32[24,64,240]{2,1,0}, f32[480,768]{1,0}, /*index=695*/f32[768,768]{0,1}, f32[24,240,240]{2,1,0}, f32[24,240,64]{2,1,0}, f32[64]{0}, f32[64]{0}, /*index=700*/f32[16,256,64,64]{3,2,1,0}, f32[16,64,128,128]{3,2,1,0}, f32[16,256,64,64]{3,2,1,0}, f32[16,64,64,64]{3,2,1,0}, f32[16,64,64,64]{3,2,1,0}, /*index=705*/f32[256]{0}, f32[256]{0}, f32[16,64,64,64]{3,2,1,0}, f32[16,64,64,64]{3,2,1,0}, f32[16,64,64,64]{3,2,1,0}, /*index=710*/f32[64]{0}, f32[64]{0}, f32[16,64,64,64]{3,2,1,0}, s64[1,240]{1,0}, f32[768,768]{0,1}, /*index=715*/f32[24,240,64]{2,1,0}, f32[2,240,768]{2,1,0}, f32[24,240,64]{2,1,0}, f32[2,240,1]{2,1,0}, f32[2,240,1]{2,1,0}, /*index=720*/f32[64]{0}, f32[512]{0}, f32[3072,768]{0,1}, f32[24,240,240]{2,1,0}, f32[256]{0}, /*index=725*/f32[256]{0}, f32[480,3072]{1,0}, f32[256]{0}, f32[256]{0}, f32[2,240,768]{2,1,0}, /*index=730*/f32[64]{0}, f32[768]{0}, f32[64]{0}, f32[2,240,1]{2,1,0}, f32[768]{0}, /*index=735*/f32[2,240,1]{2,1,0}, f32[64]{0}, f32[768,768]{0,1}, f32[64]{0}, f32[480,768]{1,0}, /*index=740*/f32[768]{0}, f32[256]{0}, f32[480,768]{1,0}, f32[768]{0}, f32[256]{0}, /*index=745*/f32[768,3072]{0,1}, f32[480,768]{1,0}, f32[64]{0}, f32[64]{0}, f32[768,768]{0,1}, /*index=750*/f32[2,240,1]{2,1,0}, f32[64]{0}, f32[2,240,1]{2,1,0}, f32[64]{0}, f32[256]{0}, /*index=755*/f32[480,768]{1,0}, f32[256]{0}, f32[768]{0}, f32[128]{0}, f32[768,768]{0,1}, /*index=760*/f32[2,240,3072]{2,1,0}, f32[768]{0}, f32[128]{0}, f32[2,240,768]{2,1,0}, f32[128]{0}, /*index=765*/f32[768]{0}, f32[128]{0}, f32[768,768]{0,1}, f32[768]{0}, f32[16,256,16,16]{3,2,1,0}, /*index=770*/f32[512]{0}, f32[480,768]{1,0}, f32[3072,768]{0,1}, f32[512]{0}, f32[512]{0}, /*index=775*/f32[512,512,3,3]{3,2,1,0}, f32[2048]{0}, f32[2048,512,1,1]{3,2,1,0}, f32[2048]{0}, f32[2048,1024,1,1]{3,2,1,0}, /*index=780*/f32[512]{0}, f32[512,2048,1,1]{3,2,1,0}, f32[512]{0}, f32[512,512,3,3]{3,2,1,0}, f32[2048]{0}, /*index=785*/f32[2048,512,1,1]{3,2,1,0}, f32[512]{0}, f32[512,2048,1,1]{3,2,1,0}, f32[512]{0}, f32[512,512,3,3]{3,2,1,0}, /*index=790*/f32[2048]{0}, f32[2048,512,1,1]{3,2,1,0})"} : (tensor<2x19xf32>, tensor<2x19xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<19xi64>, tensor<19xi64>, tensor<19xi64>, tensor<19xi64>, tensor<0xf32>, tensor<16x512x8x8xf32>, tensor<24x240x240xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<480x768xf32>, tensor<16x2048x8x8xf32>, tensor<2x2048xf32>, tensor<2x2816xf32>, tensor<2816x128xf32>, tensor<768x3072xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<2x128xf32>, tensor<480x768xf32>, tensor<16x2048x8x8xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<24x240x240xf32>, tensor<16x1024x16x16xf32>, tensor<16x512x8x8xf32>, tensor<480x768xf32>, tensor<16x1024x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<16x256x16x16xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x16x16xf32>, tensor<3072x768xf32>, tensor<480x3072xf32>, tensor<2x240x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768xf32>, tensor<2048xf32>, tensor<768xf32>, tensor<2048xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<16x2048x8x8xf32>, tensor<3072x768xf32>, tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<768xf32>, tensor<480x3072xf32>, tensor<768xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2x240x768xf32>, tensor<16x512x8x8xf32>, tensor<768xf32>, tensor<768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<16x512x8x8xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<768x768xf32>, tensor<512xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<2x240x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>, tensor<480x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<2x240x768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<2x240x3072xf32>, tensor<2x240xi64>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<16x128x32x32xf32>, tensor<2x12x240x240xf32>, tensor<480x768xf32>, tensor<3072x768xf32>, tensor<24x64x240xf32>, tensor<480x768xf32>, tensor<2x240x768xf32>, tensor<480x3072xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<2x240x768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<2x240x768xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<2x12x240x240xf32>, tensor<24x240x240xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<768x768xf32>, tensor<24x240x64xf32>, tensor<2x240x768xf32>, tensor<24x64x240xf32>, tensor<24x240x64xf32>, tensor<24x240x64xf32>, tensor<2x12x240x240xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<16x1024x16x16xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<480x768xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x3x7x7xf32>, tensor<16x256x16x16xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<24x64x240xf32>, tensor<16x256x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x16x16xf32>, tensor<16x256x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2x240x768xf32>, tensor<16x512x32x32xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<16x1024x16x16xf32>, tensor<128xf32>, tensor<16x1024x16x16xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<24x240x240xf32>, tensor<16x128x32x32xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<16x256x16x16xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<480x768xf32>, tensor<16x128x32x32xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<256xf32>, tensor<16x512x32x32xf32>, tensor<2x240x1xf32>, tensor<512xf32>, tensor<16x256x16x16xf32>, tensor<2x240x1xf32>, tensor<512xf32>, tensor<16x512x32x32xf32>, tensor<480x768xf32>, tensor<2x240x3072xf32>, tensor<128xf32>, tensor<2x240x1xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<2x240x1xf32>, tensor<512x128x1x1xf32>, tensor<128xf32>, tensor<768x768xf32>, tensor<128x128x3x3xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<256xf32>, tensor<768x768xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<768x768xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<256x1024x1x1xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<2x12x240x240xf32>, tensor<1024x256x1x1xf32>, tensor<24x64x240xf32>, tensor<256xf32>, tensor<2x12x240x240xf32>, tensor<16x256x64x64xf32>, tensor<24x240x64xf32>, tensor<256xf32>, tensor<64xf32>, tensor<2x12x240x240xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>, tensor<24x64x240xf32>, tensor<256xf32>, tensor<24x240x64xf32>, tensor<16x256x64x64xf32>, tensor<2x12x240x240xf32>, tensor<2x240x768xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<768x3072xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<3072x768xf32>, tensor<480x3072xf32>, tensor<2x12x240x240xf32>, tensor<2x240x768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<2x240x768xf32>, tensor<768x768xf32>, tensor<24x64x240xf32>, tensor<2x240x768xf32>, tensor<24x240x240xf32>, tensor<24x240x64xf32>, tensor<768x768xf32>, tensor<256xf32>, tensor<512xf32>, tensor<768x768xf32>, tensor<128x19xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2816x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<256x19xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048x256xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<768x768xf32>, tensor<2x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x2xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x3072xf32>, tensor<64xf32>, tensor<64xf32>, tensor<2x240x1xf32>, tensor<16x64x64x64xf32>, tensor<2x240x1xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<16x64x64x64xf32>, tensor<768x768xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<16x256x64x64xf32>, tensor<768x768xf32>, tensor<2x240x768xf32>, tensor<16x256x64x64xf32>, tensor<480x768xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>, tensor<24x240x64xf32>, tensor<16x512x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<16x2048x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512x8x8xf32>, tensor<16x2048x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<24x240x240xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<768x768xf32>, tensor<16x512x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<768xf32>, tensor<16x2048x8x8xf32>, tensor<480x768xf32>, tensor<480x768xf32>, tensor<16x256x16x16xf32>, tensor<768xf32>, tensor<16x512x8x8xf32>, tensor<768xf32>, tensor<256xf32>, tensor<24x240x240xf32>, tensor<256xf32>, tensor<2048xf32>, tensor<16x2048x8x8xf32>, tensor<768x3072xf32>, tensor<16x256x16x16xf32>, tensor<2048xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<512xf32>, tensor<24x64x240xf32>, tensor<512xf32>, tensor<16x512x16x16xf32>, tensor<768xf32>, tensor<768xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x16x16xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<480x3072xf32>, tensor<2x12x240x240xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<2x240x768xf32>, tensor<512xf32>, tensor<64xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<2x240x768xf32>, tensor<512xf32>, tensor<64xf32>, tensor<16x512x8x8xf32>, tensor<512x1024x1x1xf32>, tensor<2x240x3072xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<24x240x240xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<3072x768xf32>, tensor<256xf32>, tensor<768xf32>, tensor<480x3072xf32>, tensor<256x1024x1x1xf32>, tensor<768xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<768xf32>, tensor<2x240x768xf32>, tensor<768xf32>, tensor<1024xf32>, tensor<2x240x1xf32>, tensor<1024x256x1x1xf32>, tensor<480x768xf32>, tensor<2x240x1xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<2x768xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<256x256x3x3xf32>, tensor<2x240x1xf32>, tensor<1024xf32>, tensor<768x768xf32>, tensor<2x240x1xf32>, tensor<1024x256x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x64x128x128xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<256x1024x1x1xf32>, tensor<768xf32>, tensor<2x768xf32>, tensor<256xf32>, tensor<768xf32>, tensor<256x256x3x3xf32>, tensor<2x240x3072xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<512xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<768x256xf32>, tensor<768xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<2x256xf32>, tensor<768x768xf32>, tensor<480x768xf32>, tensor<2x240x768xf32>, tensor<24x64x240xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<256x2xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<24x240x240xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<24x64x240xf32>, tensor<768xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<768xf32>, tensor<2x240x768xf32>, tensor<24x240x64xf32>, tensor<768xf32>, tensor<24x64x240xf32>, tensor<24x240x64xf32>, tensor<24x240x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<768x3072xf32>, tensor<16x1024x16x16xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<2x240x3072xf32>, tensor<480x768xf32>, tensor<480x3072xf32>, tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x32x32xf32>, tensor<16x256x16x16xf32>, tensor<3072x768xf32>, tensor<16x256x16x16xf32>, tensor<2x240x768xf32>, tensor<2x240x1xf32>, tensor<256xf32>, tensor<16x1024x16x16xf32>, tensor<256xf32>, tensor<2x240x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<768x768xf32>, tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<480x768xf32>, tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>, tensor<768x768xf32>, tensor<16x256x16x16xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<64xf32>, tensor<3072x768xf32>, tensor<256x64x1x1xf32>, tensor<64x64x3x3xf32>, tensor<480x3072xf32>, tensor<128xf32>, tensor<128x256x1x1xf32>, tensor<2x240x768xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<2x240x1xf32>, tensor<512xf32>, tensor<2x240x1xf32>, tensor<512x128x1x1xf32>, tensor<768x768xf32>, tensor<512xf32>, tensor<480x768xf32>, tensor<512x256x1x1xf32>, tensor<480x768xf32>, tensor<2x240x768xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<768x768xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<768x768xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<480x768xf32>, tensor<24x240x64xf32>, tensor<128x128x3x3xf32>, tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>, tensor<64x64x3x3xf32>, tensor<64x256x1x1xf32>, tensor<24x240x64xf32>, tensor<64xf32>, tensor<24x240x240xf32>, tensor<24x240x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<480x768xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<768x3072xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<128xf32>, tensor<128xf32>, tensor<480x768xf32>, tensor<16x128x64x64xf32>, tensor<768x768xf32>, tensor<16x128x32x32xf32>, tensor<480x768xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<16x128x32x32xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<24x64x240xf32>, tensor<2x12x240x240xf32>, tensor<512xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<512xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<16x3x256x256xf32>, tensor<2x240xi64>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>, tensor<16x128x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>, tensor<16x128x32x32xf32>, tensor<16x512x32x32xf32>, tensor<64xf32>, tensor<24x240x64xf32>, tensor<64xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<2x240x3072xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3072x768xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x3072xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<24x240x64xf32>, tensor<256xf32>, tensor<2x240x768xf32>, tensor<256xf32>, tensor<24x64x240xf32>, tensor<480x768xf32>, tensor<768x768xf32>, tensor<24x240x240xf32>, tensor<24x240x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x256x64x64xf32>, tensor<16x64x128x128xf32>, tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>, tensor<1x240xi64>, tensor<768x768xf32>, tensor<24x240x64xf32>, tensor<2x240x768xf32>, tensor<24x240x64xf32>, tensor<2x240x1xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<512xf32>, tensor<3072x768xf32>, tensor<24x240x240xf32>, tensor<256xf32>, tensor<256xf32>, tensor<480x3072xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x240x768xf32>, tensor<64xf32>, tensor<768xf32>, tensor<64xf32>, tensor<2x240x1xf32>, tensor<768xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<768x768xf32>, tensor<64xf32>, tensor<480x768xf32>, tensor<768xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<768xf32>, tensor<256xf32>, tensor<768x3072xf32>, tensor<480x768xf32>, tensor<64xf32>, tensor<64xf32>, tensor<768x768xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<2x240x1xf32>, tensor<64xf32>, tensor<256xf32>, tensor<480x768xf32>, tensor<256xf32>, tensor<768xf32>, tensor<128xf32>, tensor<768x768xf32>, tensor<2x240x3072xf32>, tensor<768xf32>, tensor<128xf32>, tensor<2x240x768xf32>, tensor<128xf32>, tensor<768xf32>, tensor<128xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<16x256x16x16xf32>, tensor<512xf32>, tensor<480x768xf32>, tensor<3072x768xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<512xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<512xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>) -> !tuple
    return %2613 : !tuple
  }
  func.func private @aten.view.1080(%arg0: tensor<2x240xi64>) -> tensor<480xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240xi64>) -> tensor<480xi64>
    return %0 : tensor<480xi64>
  }
  func.func private @aten.index_select.1100(%arg0: tensor<21128x768xf32>, %arg1: tensor<480xi64>) -> tensor<480x768xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<480xi64>) -> tensor<480xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 768]> : tensor<2xi64>} : (tensor<21128x768xf32>, tensor<480xui32>) -> tensor<480x768xf32>
    return %1 : tensor<480x768xf32>
  }
  func.func private @aten.view.1090(%arg0: tensor<480x768xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    return %0 : tensor<2x240x768xf32>
  }
  func.func private @aten.expand.1074(%arg0: tensor<1x240xi64>) -> tensor<2x240xi64> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x240xi64>) -> tensor<1x240xi64>
    %1 = "mhlo.reshape"(%0) : (tensor<1x240xi64>) -> tensor<240xi64>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<240xi64>) -> tensor<2x240xi64>
    return %2 : tensor<2x240xi64>
  }
  func.func private @aten.index_select.1084(%arg0: tensor<2x768xf32>, %arg1: tensor<480xi64>) -> tensor<480x768xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<480xi64>) -> tensor<480xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 768]> : tensor<2xi64>} : (tensor<2x768xf32>, tensor<480xui32>) -> tensor<480x768xf32>
    return %1 : tensor<480x768xf32>
  }
  func.func private @aten.expand.772(%arg0: tensor<f32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    return %3 : tensor<2x240x768xf32>
  }
  func.func private @aten.mul.1094(%arg0: tensor<2x240x768xf32>, %arg1: tensor<2x240x768xf32>) -> tensor<2x240x768xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x240x768xf32>
    return %0 : tensor<2x240x768xf32>
  }
  func.func private @aten.add.1107(%arg0: tensor<2x240x768xf32>, %arg1: tensor<2x240x768xf32>) -> tensor<2x240x768xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x240x768xf32>
    return %0 : tensor<2x240x768xf32>
  }
  func.func private @aten.view.1051(%arg0: tensor<1x240xi64>) -> tensor<240xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x240xi64>) -> tensor<240xi64>
    return %0 : tensor<240xi64>
  }
  func.func private @aten.index_select.1055(%arg0: tensor<512x768xf32>, %arg1: tensor<240xi64>) -> tensor<240x768xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<240xi64>) -> tensor<240xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 768]> : tensor<2xi64>} : (tensor<512x768xf32>, tensor<240xui32>) -> tensor<240x768xf32>
    return %1 : tensor<240x768xf32>
  }
  func.func private @aten.view.1061(%arg0: tensor<240x768xf32>) -> tensor<1x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<240x768xf32>) -> tensor<1x240x768xf32>
    return %0 : tensor<1x240x768xf32>
  }
  func.func private @aten.expand.1042(%arg0: tensor<f32>) -> tensor<1x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x240x768xf32>
    return %3 : tensor<1x240x768xf32>
  }
  func.func private @aten.mul.1065(%arg0: tensor<1x240x768xf32>, %arg1: tensor<1x240x768xf32>) -> tensor<1x240x768xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x240x768xf32>
    return %0 : tensor<1x240x768xf32>
  }
  func.func private @aten.add.1112(%arg0: tensor<2x240x768xf32>, %arg1: tensor<1x240x768xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x240x768xf32>) -> tensor<1x240x768xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<1x240x768xf32>) -> tensor<240x768xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<240x768xf32>) -> tensor<2x240x768xf32>
    %3 = mhlo.add %arg0, %2 : tensor<2x240x768xf32>
    return %3 : tensor<2x240x768xf32>
  }
  func.func private @aten.view.1120(%arg0: tensor<2x240x768xf32>) -> tensor<1x480x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x768xf32>) -> tensor<1x480x768xf32>
    return %0 : tensor<1x480x768xf32>
  }
  func.func private @aten.expand.758(%arg0: tensor<f32>) -> tensor<480xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<480xf32>
    return %3 : tensor<480xf32>
  }
  func.func private @aten.native_batch_norm.1124(%arg0: tensor<1x480x768xf32>, %arg1: tensor<480xf32>, %arg2: tensor<480xf32>, %arg3: tensor<480xf32>, %arg4: tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>) -> (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>)
    %0 = mhlo.constant dense<9.99999996E-13> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<480xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<480xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<480xf32>) -> tensor<480xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[1,480,768]{2,1,0}, f32[480]{0}, f32[480]{0}, f32[480]{0})"} : (tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>) -> tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
    return %4 : tuple<tensor<1x480x768xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>>
  }
  func.func private @aten.view.1144(%arg0: tensor<1x480x768xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x480x768xf32>) -> tensor<2x240x768xf32>
    return %0 : tensor<2x240x768xf32>
  }
  func.func private @aten.mul.1148(%arg0: tensor<2x240x768xf32>, %arg1: tensor<768xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<768xf32>) -> tensor<2x240x768xf32>
    %1 = mhlo.multiply %arg0, %0 : tensor<2x240x768xf32>
    return %1 : tensor<2x240x768xf32>
  }
  func.func private @aten.mul.1154(%arg0: tensor<2x240x768xf32>, %arg1: tensor<f32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1 = mhlo.multiply %arg0, %0 : tensor<2x240x768xf32>
    return %1 : tensor<2x240x768xf32>
  }
  func.func private @aten.add.1160(%arg0: tensor<768xf32>, %arg1: tensor<2x240x768xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<768xf32>) -> tensor<2x240x768xf32>
    %1 = mhlo.add %0, %arg1 : tensor<2x240x768xf32>
    return %1 : tensor<2x240x768xf32>
  }
  func.func private @aten.view.1169(%arg0: tensor<2x240x768xf32>) -> tensor<480x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    return %0 : tensor<480x768xf32>
  }
  func.func private @aten.permute.752(%arg0: tensor<768x768xf32>) -> tensor<768x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    return %0 : tensor<768x768xf32>
  }
  func.func private @aten.addmm.1173(%arg0: tensor<480x768xf32>, %arg1: tensor<768x768xf32>, %arg2: tensor<768xf32>) -> tensor<480x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<768xf32>) -> tensor<1x768xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x768xf32>) -> tensor<768xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<768xf32>) -> tensor<480x768xf32>
    %5 = mhlo.add %0, %4 : tensor<480x768xf32>
    return %5 : tensor<480x768xf32>
  }
  func.func private @aten.view.1185(%arg0: tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    return %0 : tensor<2x240x12x64xf32>
  }
  func.func private @aten.permute.1189(%arg0: tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>, xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    return %0 : tensor<2x12x240x64xf32>
  }
  func.func private @aten.expand.1193(%arg0: tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x12x240x64xf32>) -> tensor<2x12x240x64xf32>
    return %0 : tensor<2x12x240x64xf32>
  }
  func.func private @aten.view.1197(%arg0: tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    return %0 : tensor<24x240x64xf32>
  }
  func.func private @aten.permute.1249(%arg0: tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>, xla_shape = "f32[2,12,64,240]{2,1,3,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x12x64x240xf32>
    return %0 : tensor<2x12x64x240xf32>
  }
  func.func private @aten.expand.1253(%arg0: tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x12x64x240xf32>) -> tensor<2x12x64x240xf32>
    return %0 : tensor<2x12x64x240xf32>
  }
  func.func private @aten.view.1257(%arg0: tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x12x64x240xf32>) -> tensor<24x64x240xf32>
    return %0 : tensor<24x64x240xf32>
  }
  func.func private @aten.matmul.1269(%arg0: tensor<24x240x64xf32>, %arg1: tensor<24x64x240xf32>) -> tensor<24x240x240xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    return %0 : tensor<24x240x240xf32>
  }
  func.func private @aten.view.1274(%arg0: tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    return %0 : tensor<2x12x240x240xf32>
  }
  func.func private @aten.div.1278(%arg0: tensor<2x12x240x240xf32>, %arg1: tensor<f32>) -> tensor<2x12x240x240xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x12x240x240xf32>
    %1 = mhlo.divide %arg0, %0 : tensor<2x12x240x240xf32>
    return %1 : tensor<2x12x240x240xf32>
  }
  func.func private @aten.expand.1202(%arg0: tensor<f32>) -> tensor<2x1x1x240xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<1x1xf32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<2x1x1x240xf32>
    return %3 : tensor<2x1x1x240xf32>
  }
  func.func private @aten.view.1211(%arg0: tensor<2x240xi1>) -> tensor<2x1x240xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240xi1>) -> tensor<2x1x240xi1>
    return %0 : tensor<2x1x240xi1>
  }
  func.func private @aten.view.1215(%arg0: tensor<2x1x240xi1>) -> tensor<2x1x1x240xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x1x240xi1>) -> tensor<2x1x1x240xi1>
    return %0 : tensor<2x1x1x240xi1>
  }
  func.func private @aten.mul.1223(%arg0: tensor<2x1x1x240xf32>, %arg1: tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x1x1x240xf32>
    return %0 : tensor<2x1x1x240xf32>
  }
  func.func private @aten.sub.1230(%arg0: tensor<2x1x1x240xf32>, %arg1: tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32> {
    %0 = mhlo.subtract %arg0, %arg1 : tensor<2x1x1x240xf32>
    return %0 : tensor<2x1x1x240xf32>
  }
  func.func private @aten.mul.1235(%arg0: tensor<2x1x1x240xf32>, %arg1: tensor<f32>) -> tensor<2x1x1x240xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x1x1x240xf32>
    %1 = mhlo.multiply %arg0, %0 : tensor<2x1x1x240xf32>
    return %1 : tensor<2x1x1x240xf32>
  }
  func.func private @aten.add.1284(%arg0: tensor<2x12x240x240xf32>, %arg1: tensor<2x1x1x240xf32>) -> tensor<2x12x240x240xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x1x1x240xf32>) -> tensor<2x1x1x240xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<2x1x1x240xf32>) -> tensor<2x240xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 3]> : tensor<2xi64>} : (tensor<2x240xf32>) -> tensor<2x12x240x240xf32>
    %3 = mhlo.add %arg0, %2 : tensor<2x12x240x240xf32>
    return %3 : tensor<2x12x240x240xf32>
  }
  func.func private @aten.softmax.1300(%arg0: tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32> {
    %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [3] : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %9 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%9) : (tensor<f32>) -> ()
    }
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x12x240xf32>) -> tensor<2x12x240x240xf32>
    %3 = mhlo.subtract %arg0, %2 : tensor<2x12x240x240xf32>
    %4 = "mhlo.exponential"(%3) : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%4 init: %5) across dimensions = [3] : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %9 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%9) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x12x240xf32>) -> tensor<2x12x240x240xf32>
    %8 = mhlo.divide %4, %7 : tensor<2x12x240x240xf32>
    return %8 : tensor<2x12x240x240xf32>
  }
  func.func private @aten.expand.1312(%arg0: tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    return %0 : tensor<2x12x240x240xf32>
  }
  func.func private @aten.view.1316(%arg0: tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    return %0 : tensor<24x240x240xf32>
  }
  func.func private @aten.matmul.1320(%arg0: tensor<24x240x240xf32>, %arg1: tensor<24x240x64xf32>) -> tensor<24x240x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    return %0 : tensor<24x240x64xf32>
  }
  func.func private @aten.view.1325(%arg0: tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    return %0 : tensor<2x12x240x64xf32>
  }
  func.func private @aten.permute.1329(%arg0: tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>, xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    return %0 : tensor<2x240x12x64xf32>
  }
  func.func private @aten.view.1333(%arg0: tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    return %0 : tensor<2x240x768xf32>
  }
  func.func private @aten.permute.1356(%arg0: tensor<3072x768xf32>) -> tensor<768x3072xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    return %0 : tensor<768x3072xf32>
  }
  func.func private @aten.addmm.1361(%arg0: tensor<480x768xf32>, %arg1: tensor<768x3072xf32>, %arg2: tensor<3072xf32>) -> tensor<480x3072xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3072xf32>) -> tensor<480x3072xf32>
    %5 = mhlo.add %0, %4 : tensor<480x3072xf32>
    return %5 : tensor<480x3072xf32>
  }
  func.func private @aten.view.1372(%arg0: tensor<480x3072xf32>) -> tensor<2x240x3072xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    return %0 : tensor<2x240x3072xf32>
  }
  func.func private @aten.gelu.1376(%arg0: tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32> {
    %0 = mhlo.constant dense<1.41421354> : tensor<f32>
    %1 = mhlo.constant dense<5.000000e-01> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %3 = mhlo.multiply %arg0, %2 : tensor<2x240x3072xf32>
    %4 = mhlo.constant dense<-4.000000e+00> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %6 = mhlo.constant dense<0.707106769> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %8 = mhlo.multiply %arg0, %7 : tensor<2x240x3072xf32>
    %9 = mhlo.constant dense<4.000000e+00> : tensor<f32>
    %10 = "mhlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %11 = "mhlo.clamp"(%5, %8, %10) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %12 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = "mhlo.broadcast_in_dim"(%12) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %14 = mhlo.multiply %11, %11 : tensor<2x240x3072xf32>
    %15 = mhlo.multiply %13, %14 : tensor<2x240x3072xf32>
    %16 = mhlo.constant dense<-2.72614237E-10> : tensor<f32>
    %17 = "mhlo.broadcast_in_dim"(%16) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %18 = mhlo.add %15, %17 : tensor<2x240x3072xf32>
    %19 = mhlo.multiply %18, %14 : tensor<2x240x3072xf32>
    %20 = mhlo.constant dense<2.77068146E-8> : tensor<f32>
    %21 = "mhlo.broadcast_in_dim"(%20) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %22 = mhlo.add %19, %21 : tensor<2x240x3072xf32>
    %23 = mhlo.multiply %22, %14 : tensor<2x240x3072xf32>
    %24 = mhlo.constant dense<-2.10102394E-6> : tensor<f32>
    %25 = "mhlo.broadcast_in_dim"(%24) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %26 = mhlo.add %23, %25 : tensor<2x240x3072xf32>
    %27 = mhlo.multiply %26, %14 : tensor<2x240x3072xf32>
    %28 = mhlo.constant dense<-5.69250624E-5> : tensor<f32>
    %29 = "mhlo.broadcast_in_dim"(%28) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %30 = mhlo.add %27, %29 : tensor<2x240x3072xf32>
    %31 = mhlo.multiply %30, %14 : tensor<2x240x3072xf32>
    %32 = mhlo.constant dense<-7.34990637E-4> : tensor<f32>
    %33 = "mhlo.broadcast_in_dim"(%32) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %34 = mhlo.add %31, %33 : tensor<2x240x3072xf32>
    %35 = mhlo.multiply %34, %14 : tensor<2x240x3072xf32>
    %36 = mhlo.constant dense<-2.954600e-03> : tensor<f32>
    %37 = "mhlo.broadcast_in_dim"(%36) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %38 = mhlo.add %35, %37 : tensor<2x240x3072xf32>
    %39 = mhlo.multiply %38, %14 : tensor<2x240x3072xf32>
    %40 = mhlo.constant dense<-0.0160960332> : tensor<f32>
    %41 = "mhlo.broadcast_in_dim"(%40) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %42 = mhlo.add %39, %41 : tensor<2x240x3072xf32>
    %43 = mhlo.multiply %11, %42 : tensor<2x240x3072xf32>
    %44 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = "mhlo.broadcast_in_dim"(%44) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %46 = mhlo.multiply %45, %14 : tensor<2x240x3072xf32>
    %47 = mhlo.constant dense<-1.45660715E-5> : tensor<f32>
    %48 = "mhlo.broadcast_in_dim"(%47) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %49 = mhlo.add %46, %48 : tensor<2x240x3072xf32>
    %50 = mhlo.multiply %49, %14 : tensor<2x240x3072xf32>
    %51 = mhlo.constant dense<-2.13374049E-4> : tensor<f32>
    %52 = "mhlo.broadcast_in_dim"(%51) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %53 = mhlo.add %50, %52 : tensor<2x240x3072xf32>
    %54 = mhlo.multiply %53, %14 : tensor<2x240x3072xf32>
    %55 = mhlo.constant dense<-0.00168282702> : tensor<f32>
    %56 = "mhlo.broadcast_in_dim"(%55) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %57 = mhlo.add %54, %56 : tensor<2x240x3072xf32>
    %58 = mhlo.multiply %57, %14 : tensor<2x240x3072xf32>
    %59 = mhlo.constant dense<-0.00737332925> : tensor<f32>
    %60 = "mhlo.broadcast_in_dim"(%59) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %61 = mhlo.add %58, %60 : tensor<2x240x3072xf32>
    %62 = mhlo.multiply %61, %14 : tensor<2x240x3072xf32>
    %63 = mhlo.constant dense<-0.0142647391> : tensor<f32>
    %64 = "mhlo.broadcast_in_dim"(%63) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %65 = mhlo.add %62, %64 : tensor<2x240x3072xf32>
    %66 = mhlo.divide %43, %65 : tensor<2x240x3072xf32>
    %67 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %68 = "mhlo.broadcast_in_dim"(%67) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %69 = mhlo.add %66, %68 : tensor<2x240x3072xf32>
    %70 = mhlo.multiply %3, %69 : tensor<2x240x3072xf32>
    return %70 : tensor<2x240x3072xf32>
  }
  func.func private @aten.view.1450(%arg0: tensor<2x240x3072xf32>) -> tensor<480x3072xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    return %0 : tensor<480x3072xf32>
  }
  func.func private @aten.permute.1352(%arg0: tensor<768x3072xf32>) -> tensor<3072x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    return %0 : tensor<3072x768xf32>
  }
  func.func private @aten.addmm.1454(%arg0: tensor<480x3072xf32>, %arg1: tensor<3072x768xf32>, %arg2: tensor<768xf32>) -> tensor<480x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<480x3072xf32>, tensor<3072x768xf32>) -> tensor<480x768xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<768xf32>) -> tensor<1x768xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x768xf32>) -> tensor<768xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<768xf32>) -> tensor<480x768xf32>
    %5 = mhlo.add %0, %4 : tensor<480x768xf32>
    return %5 : tensor<480x768xf32>
  }
  func.func private @aten.convolution_overrideable.2401(%arg0: tensor<16x3x256x256xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<16x64x128x128xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>) -> tensor<16x64x128x128xf32>
    return %0 : tensor<16x64x128x128xf32>
  }
  func.func private @aten.native_batch_norm.2406(%arg0: tensor<16x64x128x128xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,64,128,128]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %4 : tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.relu.2426(%arg0: tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x64x128x128xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x64x128x128xf32>
    return %2 : tensor<16x64x128x128xf32>
  }
  func.func private @aten.max_pool2d.2501(%arg0: tensor<16x64x128x128xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<16x64x64x64xui32>> {
    %0 = mhlo.constant dense<0> : tensor<ui32>
    %1 = mhlo.constant dense<4194304> : tensor<ui32>
    %2 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %3 = "mhlo.pad"(%arg0, %2) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<16x64x128x128xf32>, tensor<f32>) -> tensor<16x64x130x130xf32>
    %4 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %5 = "mhlo.reduce_window"(%3, %4) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %16 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%16) : (tensor<f32>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<16x64x130x130xf32>, tensor<f32>) -> tensor<16x64x64x64xf32>
    %6 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<16384xui32>
    %7 = "mhlo.reshape"(%6) : (tensor<16384xui32>) -> tensor<128x128xui32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<128x128xui32>) -> tensor<16x64x128x128xui32>
    %9 = mhlo.constant dense<4294967295> : tensor<ui32>
    %10 = "mhlo.pad"(%8, %9) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<16x64x128x128xui32>, tensor<ui32>) -> tensor<16x64x130x130xui32>
    %11 = mhlo.constant dense<0> : tensor<ui32>
    %12 = "mhlo.broadcast_in_dim"(%11) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<ui32>) -> tensor<4194304xui32>
    %13:6 = mhlo.while(%iterArg = %0, %iterArg_0 = %1, %iterArg_1 = %3, %iterArg_2 = %5, %iterArg_3 = %10, %iterArg_4 = %12) : tensor<ui32>, tensor<ui32>, tensor<16x64x130x130xf32>, tensor<16x64x64x64xf32>, tensor<16x64x130x130xui32>, tensor<4194304xui32>
     cond {
      %16 = "mhlo.compare"(%iterArg, %iterArg_0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
      "mhlo.return"(%16) : (tensor<i1>) -> ()
    } do {
      %16 = mhlo.constant dense<262144> : tensor<ui32>
      %17 = mhlo.remainder %iterArg, %16 : tensor<ui32>
      %18 = mhlo.constant dense<4096> : tensor<ui32>
      %19 = mhlo.remainder %17, %18 : tensor<ui32>
      %20 = mhlo.constant dense<64> : tensor<ui32>
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
      %38 = "mhlo.dynamic_slice"(%iterArg_1, %28, %31, %34, %37) {slice_sizes = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<16x64x130x130xf32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x3x3xf32>
      %39 = "mhlo.dynamic_slice"(%iterArg_2, %27, %30, %33, %36) {slice_sizes = dense<1> : tensor<4xi64>} : (tensor<16x64x64x64xf32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x1x1xf32>
      %40 = mhlo.constant dense<0xFF800000> : tensor<f32>
      %41 = "mhlo.select_and_scatter"(%38, %39, %40) ({
      ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
        %51 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
        "mhlo.return"(%51) : (tensor<i1>) -> ()
      }, {
      ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
        %51 = mhlo.maximum %arg1, %arg2 : tensor<f32>
        "mhlo.return"(%51) : (tensor<f32>) -> ()
      }) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x1x3x3xf32>, tensor<1x1x1x1xf32>, tensor<f32>) -> tensor<1x1x3x3xf32>
      %42 = "mhlo.broadcast_in_dim"(%40) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x1x3x3xf32>
      %43 = "mhlo.compare"(%41, %42) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xi1>
      %44 = "mhlo.dynamic_slice"(%iterArg_3, %28, %31, %34, %37) {slice_sizes = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<16x64x130x130xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x3x3xui32>
      %45 = mhlo.constant dense<4294967295> : tensor<ui32>
      %46 = "mhlo.broadcast_in_dim"(%45) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<ui32>) -> tensor<1x1x3x3xui32>
      %47 = "mhlo.select"(%43, %44, %46) : (tensor<1x1x3x3xi1>, tensor<1x1x3x3xui32>, tensor<1x1x3x3xui32>) -> tensor<1x1x3x3xui32>
      %48 = "mhlo.reduce_window"(%47, %45) ({
      ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>):
        %51 = mhlo.minimum %arg1, %arg2 : tensor<ui32>
        "mhlo.return"(%51) : (tensor<ui32>) -> ()
      }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x1x3x3xui32>, tensor<ui32>) -> tensor<1x1x1x1xui32>
      %49 = "mhlo.reshape"(%48) : (tensor<1x1x1x1xui32>) -> tensor<1xui32>
      %50 = "mhlo.dynamic_update_slice"(%iterArg_4, %49, %iterArg) : (tensor<4194304xui32>, tensor<1xui32>, tensor<ui32>) -> tensor<4194304xui32>
      "mhlo.return"(%25, %iterArg_0, %iterArg_1, %iterArg_2, %iterArg_3, %50) : (tensor<ui32>, tensor<ui32>, tensor<16x64x130x130xf32>, tensor<16x64x64x64xf32>, tensor<16x64x130x130xui32>, tensor<4194304xui32>) -> ()
    }
    %14 = "mhlo.reshape"(%13#5) : (tensor<4194304xui32>) -> tensor<16x64x64x64xui32>
    %15 = "mhlo.tuple"(%5, %14) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, u32[16,64,64,64]{3,2,1,0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xui32>) -> tuple<tensor<16x64x64x64xf32>, tensor<16x64x64x64xui32>>
    return %15 : tuple<tensor<16x64x64x64xf32>, tensor<16x64x64x64xui32>>
  }
  func.func private @aten.convolution_overrideable.2571(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<256x64x1x1xf32>) -> tensor<16x256x64x64xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tensor<16x256x64x64xf32>
    return %0 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.native_batch_norm.2576(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %4 : tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_overrideable.2529(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<64x64x1x1xf32>) -> tensor<16x64x64x64xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>) -> tensor<16x64x64x64xf32>
    return %0 : tensor<16x64x64x64xf32>
  }
  func.func private @aten.native_batch_norm.2534(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %4 : tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.relu.2554(%arg0: tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x64x64x64xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x64x64x64xf32>
    return %2 : tensor<16x64x64x64xf32>
  }
  func.func private @aten.convolution_overrideable.2560(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<64x64x3x3xf32>) -> tensor<16x64x64x64xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>) -> tensor<16x64x64x64xf32>
    return %0 : tensor<16x64x64x64xf32>
  }
  func.func private @aten.expand.2390(%arg0: tensor<f32>) -> tensor<16x256x64x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x64x64xf32>
    return %3 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.mul.2596(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x256x64x64xf32>
    return %0 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.add.2607(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x256x64x64xf32>
    return %0 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.relu.2612(%arg0: tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x256x64x64xf32>
    return %2 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.convolution_overrideable.2618(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<64x256x1x1xf32>) -> tensor<16x64x64x64xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>) -> tensor<16x64x64x64xf32>
    return %0 : tensor<16x64x64x64xf32>
  }
  func.func private @aten.convolution_overrideable.2760(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<512x256x1x1xf32>) -> tensor<16x512x32x32xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>) -> tensor<16x512x32x32xf32>
    return %0 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.native_batch_norm.2735(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %4 : tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_overrideable.2668(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<128x256x1x1xf32>) -> tensor<16x128x64x64xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>) -> tensor<16x128x64x64xf32>
    return %0 : tensor<16x128x64x64xf32>
  }
  func.func private @aten.native_batch_norm.2673(%arg0: tensor<16x128x64x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,128,64,64]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %4 : tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.relu.2693(%arg0: tensor<16x128x64x64xf32>) -> tensor<16x128x64x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x128x64x64xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x128x64x64xf32>
    return %2 : tensor<16x128x64x64xf32>
  }
  func.func private @aten.convolution_overrideable.2699(%arg0: tensor<16x128x64x64xf32>, %arg1: tensor<128x128x3x3xf32>) -> tensor<16x128x32x32xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>) -> tensor<16x128x32x32xf32>
    return %0 : tensor<16x128x32x32xf32>
  }
  func.func private @aten.native_batch_norm.2704(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %4 : tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.relu.2724(%arg0: tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x128x32x32xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x128x32x32xf32>
    return %2 : tensor<16x128x32x32xf32>
  }
  func.func private @aten.convolution_overrideable.2730(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<512x128x1x1xf32>) -> tensor<16x512x32x32xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tensor<16x512x32x32xf32>
    return %0 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.expand.2376(%arg0: tensor<f32>) -> tensor<16x512x32x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x32x32xf32>
    return %3 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.mul.2755(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x512x32x32xf32>
    return %0 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.add.2770(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x512x32x32xf32>
    return %0 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.relu.2775(%arg0: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x512x32x32xf32>
    return %2 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.convolution_overrideable.2781(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<128x512x1x1xf32>) -> tensor<16x128x32x32xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>) -> tensor<16x128x32x32xf32>
    return %0 : tensor<16x128x32x32xf32>
  }
  func.func private @aten.convolution_overrideable.2792(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<128x128x3x3xf32>) -> tensor<16x128x32x32xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>) -> tensor<16x128x32x32xf32>
    return %0 : tensor<16x128x32x32xf32>
  }
  func.func private @aten.convolution_overrideable.2950(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<1024x512x1x1xf32>) -> tensor<16x1024x16x16xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>) -> tensor<16x1024x16x16xf32>
    return %0 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.native_batch_norm.2925(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<1024xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<1024xf32>) -> tensor<1024xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
    return %4 : tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>>
  }
  func.func private @aten.convolution_overrideable.2858(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<256x512x1x1xf32>) -> tensor<16x256x32x32xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>) -> tensor<16x256x32x32xf32>
    return %0 : tensor<16x256x32x32xf32>
  }
  func.func private @aten.native_batch_norm.2863(%arg0: tensor<16x256x32x32xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,256,32,32]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %4 : tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.relu.2883(%arg0: tensor<16x256x32x32xf32>) -> tensor<16x256x32x32xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x32x32xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x256x32x32xf32>
    return %2 : tensor<16x256x32x32xf32>
  }
  func.func private @aten.convolution_overrideable.2889(%arg0: tensor<16x256x32x32xf32>, %arg1: tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32>
    return %0 : tensor<16x256x16x16xf32>
  }
  func.func private @aten.native_batch_norm.2894(%arg0: tensor<16x256x16x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %4 : tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.relu.2914(%arg0: tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x16x16xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x256x16x16xf32>
    return %2 : tensor<16x256x16x16xf32>
  }
  func.func private @aten.convolution_overrideable.2920(%arg0: tensor<16x256x16x16xf32>, %arg1: tensor<1024x256x1x1xf32>) -> tensor<16x1024x16x16xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tensor<16x1024x16x16xf32>
    return %0 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.expand.2358(%arg0: tensor<f32>) -> tensor<16x1024x16x16xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    return %3 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.mul.2945(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x1024x16x16xf32>
    return %0 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.add.2960(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x1024x16x16xf32>
    return %0 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.relu.2965(%arg0: tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x1024x16x16xf32>
    return %2 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.convolution_overrideable.2971(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<256x1024x1x1xf32>) -> tensor<16x256x16x16xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tensor<16x256x16x16xf32>
    return %0 : tensor<16x256x16x16xf32>
  }
  func.func private @aten.convolution_overrideable.2982(%arg0: tensor<16x256x16x16xf32>, %arg1: tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tensor<16x256x16x16xf32>
    return %0 : tensor<16x256x16x16xf32>
  }
  func.func private @aten.convolution_overrideable.3186(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<2048x1024x1x1xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>) -> tensor<16x2048x8x8xf32>
    return %0 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.native_batch_norm.3161(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>) -> (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<2048xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<2048xf32>) -> tensor<2048xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>
    return %4 : tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>>
  }
  func.func private @aten.convolution_overrideable.3094(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<512x1024x1x1xf32>) -> tensor<16x512x16x16xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>) -> tensor<16x512x16x16xf32>
    return %0 : tensor<16x512x16x16xf32>
  }
  func.func private @aten.native_batch_norm.3099(%arg0: tensor<16x512x16x16xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,512,16,16]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %4 : tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.relu.3119(%arg0: tensor<16x512x16x16xf32>) -> tensor<16x512x16x16xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x16x16xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x512x16x16xf32>
    return %2 : tensor<16x512x16x16xf32>
  }
  func.func private @aten.convolution_overrideable.3125(%arg0: tensor<16x512x16x16xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<16x512x8x8xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>) -> tensor<16x512x8x8xf32>
    return %0 : tensor<16x512x8x8xf32>
  }
  func.func private @aten.native_batch_norm.3130(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>)
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2 = mhlo.add %batch_var, %1 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.tuple"(%output, %batch_mean, %batch_var, %3) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %4 : tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.relu.3150(%arg0: tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x8x8xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x512x8x8xf32>
    return %2 : tensor<16x512x8x8xf32>
  }
  func.func private @aten.convolution_overrideable.3156(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<2048x512x1x1xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>) -> tensor<16x2048x8x8xf32>
    return %0 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.expand.2346(%arg0: tensor<f32>) -> tensor<16x2048x8x8xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    return %3 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.mul.3181(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x2048x8x8xf32>
    return %0 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.add.3196(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x2048x8x8xf32>
    return %0 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.relu.3201(%arg0: tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<16x2048x8x8xf32>
    return %2 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.convolution_overrideable.3207(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<512x2048x1x1xf32>) -> tensor<16x512x8x8xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>) -> tensor<16x512x8x8xf32>
    return %0 : tensor<16x512x8x8xf32>
  }
  func.func private @aten.convolution_overrideable.3218(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<16x512x8x8xf32> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>) -> tensor<16x512x8x8xf32>
    return %0 : tensor<16x512x8x8xf32>
  }
  func.func private @aten.view.3261(%arg0: tensor<16x2048x8x8xf32>) -> tensor<2x8x2048x8x8xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<16x2048x8x8xf32>) -> tensor<2x8x2048x8x8xf32>
    return %0 : tensor<2x8x2048x8x8xf32>
  }
  func.func private @aten.permute.3265(%arg0: tensor<2x8x2048x8x8xf32>) -> tensor<2x2048x8x8x8xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3, 4]> : tensor<5xi64>, xla_shape = "f32[2,2048,8,8,8]{4,3,1,2,0}"} : (tensor<2x8x2048x8x8xf32>) -> tensor<2x2048x8x8x8xf32>
    return %0 : tensor<2x2048x8x8x8xf32>
  }
  func.func private @aten.mean.3273(%arg0: tensor<2x2048x8x8x8xf32>) -> tensor<2x2048xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [2, 3, 4] : (tensor<2x2048x8x8x8xf32>, tensor<f32>) -> tensor<2x2048xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %13 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%13) : (tensor<f32>) -> ()
    }
    %2 = mhlo.constant dense<512> : tensor<i64>
    %3 = mhlo.constant dense<0> : tensor<i64>
    %4 = "mhlo.compare"(%2, %3) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %5 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %6 = "mhlo.convert"(%2) : (tensor<i64>) -> tensor<f32>
    %7 = mhlo.divide %5, %6 : tensor<f32>
    %8 = mhlo.constant dense<0x7FC00000> : tensor<f32>
    %9 = "mhlo.select"(%4, %7, %8) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %10 = "mhlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x2048xf32>
    %11 = mhlo.multiply %1, %10 : tensor<2x2048xf32>
    %12 = "mhlo.convert"(%11) : (tensor<2x2048xf32>) -> tensor<2x2048xf32>
    return %12 : tensor<2x2048xf32>
  }
  func.func private @aten.view.2326(%arg0: tensor<2x1x768xf32>) -> tensor<2x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x1x768xf32>) -> tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
  func.func private @aten.addmm.2330(%arg0: tensor<2x768xf32>, %arg1: tensor<768x768xf32>, %arg2: tensor<768xf32>) -> tensor<2x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x768xf32>, tensor<768x768xf32>) -> tensor<2x768xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<768xf32>) -> tensor<1x768xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x768xf32>) -> tensor<768xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<768xf32>) -> tensor<2x768xf32>
    %5 = mhlo.add %0, %4 : tensor<2x768xf32>
    return %5 : tensor<2x768xf32>
  }
  func.func private @aten.tanh.2341(%arg0: tensor<2x768xf32>) -> tensor<2x768xf32> {
    %0 = "mhlo.tanh"(%arg0) : (tensor<2x768xf32>) -> tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
  func.func private @aten.cat.3289(%arg0: tensor<2x2048xf32>, %arg1: tensor<2x768xf32>) -> tensor<2x2816xf32> {
    %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<2x2048xf32>, tensor<2x768xf32>) -> tensor<2x2816xf32>
    return %0 : tensor<2x2816xf32>
  }
  func.func private @aten.permute.748(%arg0: tensor<256x2816xf32>) -> tensor<2816x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[2816,256]{0,1}"} : (tensor<256x2816xf32>) -> tensor<2816x256xf32>
    return %0 : tensor<2816x256xf32>
  }
  func.func private @aten.addmm.3294(%arg0: tensor<2x2816xf32>, %arg1: tensor<2816x256xf32>, %arg2: tensor<256xf32>) -> tensor<2x256xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x2816xf32>, tensor<2816x256xf32>) -> tensor<2x256xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<256xf32>) -> tensor<1x256xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x256xf32>) -> tensor<1x256xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x256xf32>) -> tensor<256xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x256xf32>
    %5 = mhlo.add %0, %4 : tensor<2x256xf32>
    return %5 : tensor<2x256xf32>
  }
  func.func private @aten.relu.3305(%arg0: tensor<2x256xf32>) -> tensor<2x256xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x256xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<2x256xf32>
    return %2 : tensor<2x256xf32>
  }
  func.func private @aten.permute.744(%arg0: tensor<19x256xf32>) -> tensor<256x19xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[256,19]{0,1}"} : (tensor<19x256xf32>) -> tensor<256x19xf32>
    return %0 : tensor<256x19xf32>
  }
  func.func private @aten.addmm.3311(%arg0: tensor<2x256xf32>, %arg1: tensor<256x19xf32>, %arg2: tensor<19xf32>) -> tensor<2x19xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x256xf32>, tensor<256x19xf32>) -> tensor<2x19xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<19xf32>) -> tensor<1x19xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x19xf32>) -> tensor<1x19xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x19xf32>) -> tensor<19xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<19xf32>) -> tensor<2x19xf32>
    %5 = mhlo.add %0, %4 : tensor<2x19xf32>
    return %5 : tensor<2x19xf32>
  }
  func.func private @aten.permute.3326(%arg0: tensor<128x2816xf32>) -> tensor<2816x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[2816,128]{0,1}"} : (tensor<128x2816xf32>) -> tensor<2816x128xf32>
    return %0 : tensor<2816x128xf32>
  }
  func.func private @aten.addmm.3330(%arg0: tensor<2x2816xf32>, %arg1: tensor<2816x128xf32>, %arg2: tensor<128xf32>) -> tensor<2x128xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x2816xf32>, tensor<2816x128xf32>) -> tensor<2x128xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x128xf32>) -> tensor<128xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<2x128xf32>
    %5 = mhlo.add %0, %4 : tensor<2x128xf32>
    return %5 : tensor<2x128xf32>
  }
  func.func private @aten.relu.3341(%arg0: tensor<2x128xf32>) -> tensor<2x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<2x128xf32>
    return %2 : tensor<2x128xf32>
  }
  func.func private @aten.permute.3322(%arg0: tensor<19x128xf32>) -> tensor<128x19xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[128,19]{0,1}"} : (tensor<19x128xf32>) -> tensor<128x19xf32>
    return %0 : tensor<128x19xf32>
  }
  func.func private @aten.addmm.3347(%arg0: tensor<2x128xf32>, %arg1: tensor<128x19xf32>, %arg2: tensor<19xf32>) -> tensor<2x19xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x128xf32>, tensor<128x19xf32>) -> tensor<2x19xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<19xf32>) -> tensor<1x19xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x19xf32>) -> tensor<1x19xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x19xf32>) -> tensor<19xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<19xf32>) -> tensor<2x19xf32>
    %5 = mhlo.add %0, %4 : tensor<2x19xf32>
    return %5 : tensor<2x19xf32>
  }
  func.func private @aten.permute.3362(%arg0: tensor<256x2048xf32>) -> tensor<2048x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[2048,256]{0,1}"} : (tensor<256x2048xf32>) -> tensor<2048x256xf32>
    return %0 : tensor<2048x256xf32>
  }
  func.func private @aten.addmm.3366(%arg0: tensor<2x2048xf32>, %arg1: tensor<2048x256xf32>, %arg2: tensor<256xf32>) -> tensor<2x256xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x2048xf32>, tensor<2048x256xf32>) -> tensor<2x256xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<256xf32>) -> tensor<1x256xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x256xf32>) -> tensor<1x256xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x256xf32>) -> tensor<256xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x256xf32>
    %5 = mhlo.add %0, %4 : tensor<2x256xf32>
    return %5 : tensor<2x256xf32>
  }
  func.func private @aten.permute.3358(%arg0: tensor<2x256xf32>) -> tensor<256x2xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    return %0 : tensor<256x2xf32>
  }
  func.func private @aten.addmm.3378(%arg0: tensor<2x256xf32>, %arg1: tensor<256x2xf32>, %arg2: tensor<2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x256xf32>, tensor<256x2xf32>) -> tensor<2x2xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<2xf32>) -> tensor<1x2xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x2xf32>) -> tensor<2xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2xf32>) -> tensor<2x2xf32>
    %5 = mhlo.add %0, %4 : tensor<2x2xf32>
    return %5 : tensor<2x2xf32>
  }
  func.func private @aten.permute.3390(%arg0: tensor<256x768xf32>) -> tensor<768x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[768,256]{0,1}"} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    return %0 : tensor<768x256xf32>
  }
  func.func private @aten.addmm.3394(%arg0: tensor<2x768xf32>, %arg1: tensor<768x256xf32>, %arg2: tensor<256xf32>) -> tensor<2x256xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x768xf32>, tensor<768x256xf32>) -> tensor<2x256xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<256xf32>) -> tensor<1x256xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x256xf32>) -> tensor<1x256xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x256xf32>) -> tensor<256xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x256xf32>
    %5 = mhlo.add %0, %4 : tensor<2x256xf32>
    return %5 : tensor<2x256xf32>
  }
  func.func private @aten.view.3408(%arg0: tensor<1x19xi64>) -> tensor<19xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x19xi64>) -> tensor<19xi64>
    return %0 : tensor<19xi64>
  }
  func.func private @aten.ge.3415(%arg0: tensor<19xi64>, %arg1: tensor<i64>) -> tensor<19xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<19xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<19xi64>, tensor<19xi64>) -> tensor<19xi1>
    return %1 : tensor<19xi1>
  }
  func.func private @aten.view.3422(%arg0: tensor<480xf32>) -> tensor<2x240x1xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<480xf32>) -> tensor<2x240x1xf32>
    return %0 : tensor<2x240x1xf32>
  }
}

