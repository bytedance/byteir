// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
!tuple = tuple<tensor<19xf32>, tensor<19x256xf32>, tensor<19xf32>, tensor<19x128xf32>, tensor<2xf32>, tensor<2x256xf32>, tensor<2xf32>, tensor<2x256xf32>, tensor<256xf32>, tensor<256x2816xf32>, tensor<128xf32>, tensor<128x2816xf32>, tensor<256xf32>, tensor<256x2048xf32>, tensor<256xf32>, tensor<256x768xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<512x768xf32>, tensor<2x768xf32>, tensor<21128x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768x768xf32>>
module @IrToMhlo.4124 {
  func.func @main(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<24x240x240xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<480x768xf32>, %arg5: tensor<16x2048x8x8xf32>, %arg6: tensor<2x2048xf32>, %arg7: tensor<2x2816xf32>, %arg8: tensor<2816x128xf32>, %arg9: tensor<768x3072xf32>, %arg10: tensor<2x240x1xf32>, %arg11: tensor<2x240x1xf32>, %arg12: tensor<2x128xf32>, %arg13: tensor<480x768xf32>, %arg14: tensor<16x2048x8x8xf32>, %arg15: tensor<1024xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<24x240x240xf32>, %arg18: tensor<16x1024x16x16xf32>, %arg19: tensor<16x512x8x8xf32>, %arg20: tensor<480x768xf32>, %arg21: tensor<16x1024x16x16xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<768x3072xf32>, %arg25: tensor<16x256x16x16xf32>, %arg26: tensor<2x240x1xf32>, %arg27: tensor<2x240x1xf32>, %arg28: tensor<2x240x3072xf32>, %arg29: tensor<480x768xf32>, %arg30: tensor<256xf32>, %arg31: tensor<256xf32>, %arg32: tensor<16x256x16x16xf32>, %arg33: tensor<3072x768xf32>, %arg34: tensor<480x3072xf32>, %arg35: tensor<2x240x768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x3072xf32>, %arg40: tensor<768xf32>, %arg41: tensor<2x240x1xf32>, %arg42: tensor<2x240x1xf32>, %arg43: tensor<768xf32>, %arg44: tensor<2048xf32>, %arg45: tensor<768xf32>, %arg46: tensor<2048xf32>, %arg47: tensor<2x240x3072xf32>, %arg48: tensor<480x768xf32>, %arg49: tensor<16x2048x8x8xf32>, %arg50: tensor<3072x768xf32>, %arg51: tensor<16x512x8x8xf32>, %arg52: tensor<16x512x8x8xf32>, %arg53: tensor<768xf32>, %arg54: tensor<480x3072xf32>, %arg55: tensor<768xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<2x240x768xf32>, %arg59: tensor<16x512x8x8xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768xf32>, %arg62: tensor<2x240x1xf32>, %arg63: tensor<2x240x1xf32>, %arg64: tensor<768x768xf32>, %arg65: tensor<16x512x8x8xf32>, %arg66: tensor<480x768xf32>, %arg67: tensor<512xf32>, %arg68: tensor<768x768xf32>, %arg69: tensor<512xf32>, %arg70: tensor<2x240x1xf32>, %arg71: tensor<2x240x1xf32>, %arg72: tensor<480x768xf32>, %arg73: tensor<24x240x64xf32>, %arg74: tensor<2x240x3072xf32>, %arg75: tensor<768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<3072x768xf32>, %arg78: tensor<480x3072xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<2x240x768xf32>, %arg82: tensor<2x240x1xf32>, %arg83: tensor<2x240x1xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<480x768xf32>, %arg87: tensor<768xf32>, %arg88: tensor<768x768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768xf32>, %arg92: tensor<480x768xf32>, %arg93: tensor<2x240x3072xf32>, %arg94: tensor<2x240xi64>, %arg95: tensor<2x240x1xf32>, %arg96: tensor<2x240x1xf32>, %arg97: tensor<768x768xf32>, %arg98: tensor<16x128x32x32xf32>, %arg99: tensor<2x12x240x240xf32>, %arg100: tensor<480x768xf32>, %arg101: tensor<3072x768xf32>, %arg102: tensor<24x64x240xf32>, %arg103: tensor<480x768xf32>, %arg104: tensor<2x240x768xf32>, %arg105: tensor<480x3072xf32>, %arg106: tensor<768x768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<2x240x768xf32>, %arg109: tensor<2x240x1xf32>, %arg110: tensor<2x240x1xf32>, %arg111: tensor<768x768xf32>, %arg112: tensor<768x768xf32>, %arg113: tensor<480x768xf32>, %arg114: tensor<480x768xf32>, %arg115: tensor<768x768xf32>, %arg116: tensor<2x240x768xf32>, %arg117: tensor<480x768xf32>, %arg118: tensor<768x768xf32>, %arg119: tensor<480x768xf32>, %arg120: tensor<24x240x64xf32>, %arg121: tensor<2x12x240x240xf32>, %arg122: tensor<24x240x240xf32>, %arg123: tensor<480x768xf32>, %arg124: tensor<24x240x64xf32>, %arg125: tensor<768x768xf32>, %arg126: tensor<24x240x64xf32>, %arg127: tensor<2x240x768xf32>, %arg128: tensor<24x64x240xf32>, %arg129: tensor<24x240x64xf32>, %arg130: tensor<24x240x64xf32>, %arg131: tensor<2x12x240x240xf32>, %arg132: tensor<1024xf32>, %arg133: tensor<1024xf32>, %arg134: tensor<768x768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<16x1024x16x16xf32>, %arg137: tensor<128xf32>, %arg138: tensor<128xf32>, %arg139: tensor<16x128x32x32xf32>, %arg140: tensor<480x768xf32>, %arg141: tensor<16x256x16x16xf32>, %arg142: tensor<256xf32>, %arg143: tensor<256xf32>, %arg144: tensor<64x3x7x7xf32>, %arg145: tensor<16x256x16x16xf32>, %arg146: tensor<128xf32>, %arg147: tensor<128xf32>, %arg148: tensor<16x128x32x32xf32>, %arg149: tensor<24x64x240xf32>, %arg150: tensor<16x256x16x16xf32>, %arg151: tensor<16x1024x16x16xf32>, %arg152: tensor<16x1024x16x16xf32>, %arg153: tensor<256xf32>, %arg154: tensor<256xf32>, %arg155: tensor<16x256x16x16xf32>, %arg156: tensor<16x256x32x32xf32>, %arg157: tensor<512xf32>, %arg158: tensor<512xf32>, %arg159: tensor<2x240x768xf32>, %arg160: tensor<16x512x32x32xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<64xf32>, %arg163: tensor<1024xf32>, %arg164: tensor<64xf32>, %arg165: tensor<16x1024x16x16xf32>, %arg166: tensor<128xf32>, %arg167: tensor<16x1024x16x16xf32>, %arg168: tensor<128xf32>, %arg169: tensor<16x128x32x32xf32>, %arg170: tensor<24x240x240xf32>, %arg171: tensor<16x128x32x32xf32>, %arg172: tensor<16x256x16x16xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256xf32>, %arg175: tensor<128xf32>, %arg176: tensor<16x256x16x16xf32>, %arg177: tensor<128xf32>, %arg178: tensor<16x128x32x32xf32>, %arg179: tensor<480x768xf32>, %arg180: tensor<16x128x32x32xf32>, %arg181: tensor<16x256x16x16xf32>, %arg182: tensor<256xf32>, %arg183: tensor<768x3072xf32>, %arg184: tensor<256xf32>, %arg185: tensor<16x512x32x32xf32>, %arg186: tensor<2x240x1xf32>, %arg187: tensor<512xf32>, %arg188: tensor<16x256x16x16xf32>, %arg189: tensor<2x240x1xf32>, %arg190: tensor<512xf32>, %arg191: tensor<16x512x32x32xf32>, %arg192: tensor<480x768xf32>, %arg193: tensor<2x240x3072xf32>, %arg194: tensor<128xf32>, %arg195: tensor<2x240x1xf32>, %arg196: tensor<512xf32>, %arg197: tensor<128x512x1x1xf32>, %arg198: tensor<2x240x1xf32>, %arg199: tensor<512x128x1x1xf32>, %arg200: tensor<128xf32>, %arg201: tensor<768x768xf32>, %arg202: tensor<128x128x3x3xf32>, %arg203: tensor<480x768xf32>, %arg204: tensor<512xf32>, %arg205: tensor<512x128x1x1xf32>, %arg206: tensor<256xf32>, %arg207: tensor<768x768xf32>, %arg208: tensor<256x512x1x1xf32>, %arg209: tensor<256xf32>, %arg210: tensor<480x768xf32>, %arg211: tensor<256x256x3x3xf32>, %arg212: tensor<1024xf32>, %arg213: tensor<1024x256x1x1xf32>, %arg214: tensor<768x768xf32>, %arg215: tensor<1024xf32>, %arg216: tensor<1024x512x1x1xf32>, %arg217: tensor<256x1024x1x1xf32>, %arg218: tensor<480x768xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256x1024x1x1xf32>, %arg221: tensor<256xf32>, %arg222: tensor<256x256x3x3xf32>, %arg223: tensor<1024xf32>, %arg224: tensor<2x12x240x240xf32>, %arg225: tensor<1024x256x1x1xf32>, %arg226: tensor<24x64x240xf32>, %arg227: tensor<256xf32>, %arg228: tensor<2x12x240x240xf32>, %arg229: tensor<16x256x64x64xf32>, %arg230: tensor<24x240x64xf32>, %arg231: tensor<256xf32>, %arg232: tensor<64xf32>, %arg233: tensor<2x12x240x240xf32>, %arg234: tensor<64xf32>, %arg235: tensor<16x64x64x64xf32>, %arg236: tensor<24x64x240xf32>, %arg237: tensor<256xf32>, %arg238: tensor<24x240x64xf32>, %arg239: tensor<16x256x64x64xf32>, %arg240: tensor<2x12x240x240xf32>, %arg241: tensor<2x240x768xf32>, %arg242: tensor<16x512x32x32xf32>, %arg243: tensor<512xf32>, %arg244: tensor<512xf32>, %arg245: tensor<768x3072xf32>, %arg246: tensor<2x240x1xf32>, %arg247: tensor<2x240x1xf32>, %arg248: tensor<2x240x3072xf32>, %arg249: tensor<480x768xf32>, %arg250: tensor<24x240x64xf32>, %arg251: tensor<3072x768xf32>, %arg252: tensor<480x3072xf32>, %arg253: tensor<2x12x240x240xf32>, %arg254: tensor<2x240x768xf32>, %arg255: tensor<2x240x1xf32>, %arg256: tensor<2x240x1xf32>, %arg257: tensor<768x768xf32>, %arg258: tensor<480x768xf32>, %arg259: tensor<2x240x768xf32>, %arg260: tensor<768x768xf32>, %arg261: tensor<24x64x240xf32>, %arg262: tensor<2x240x768xf32>, %arg263: tensor<24x240x240xf32>, %arg264: tensor<24x240x64xf32>, %arg265: tensor<768x768xf32>, %arg266: tensor<256xf32>, %arg267: tensor<512xf32>, %arg268: tensor<768x768xf32>, %arg269: tensor<128x19xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<2816x256xf32>, %arg275: tensor<256xf32>, %arg276: tensor<256xf32>, %arg277: tensor<2x256xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<1024xf32>, %arg280: tensor<480x768xf32>, %arg281: tensor<512xf32>, %arg282: tensor<480x768xf32>, %arg283: tensor<512xf32>, %arg284: tensor<256x19xf32>, %arg285: tensor<512xf32>, %arg286: tensor<512xf32>, %arg287: tensor<2048xf32>, %arg288: tensor<2048x256xf32>, %arg289: tensor<2048xf32>, %arg290: tensor<2048xf32>, %arg291: tensor<2048xf32>, %arg292: tensor<768x768xf32>, %arg293: tensor<2x256xf32>, %arg294: tensor<512xf32>, %arg295: tensor<512xf32>, %arg296: tensor<480x768xf32>, %arg297: tensor<512xf32>, %arg298: tensor<512xf32>, %arg299: tensor<256x2xf32>, %arg300: tensor<2048xf32>, %arg301: tensor<2048xf32>, %arg302: tensor<512xf32>, %arg303: tensor<1024xf32>, %arg304: tensor<128xf32>, %arg305: tensor<128xf32>, %arg306: tensor<128xf32>, %arg307: tensor<128xf32>, %arg308: tensor<512xf32>, %arg309: tensor<512xf32>, %arg310: tensor<128xf32>, %arg311: tensor<128xf32>, %arg312: tensor<128xf32>, %arg313: tensor<128xf32>, %arg314: tensor<512xf32>, %arg315: tensor<512xf32>, %arg316: tensor<128xf32>, %arg317: tensor<128xf32>, %arg318: tensor<128xf32>, %arg319: tensor<128xf32>, %arg320: tensor<512xf32>, %arg321: tensor<512xf32>, %arg322: tensor<256xf32>, %arg323: tensor<256xf32>, %arg324: tensor<256xf32>, %arg325: tensor<256xf32>, %arg326: tensor<480x3072xf32>, %arg327: tensor<64xf32>, %arg328: tensor<64xf32>, %arg329: tensor<2x240x1xf32>, %arg330: tensor<16x64x64x64xf32>, %arg331: tensor<2x240x1xf32>, %arg332: tensor<768x768xf32>, %arg333: tensor<480x768xf32>, %arg334: tensor<16x64x64x64xf32>, %arg335: tensor<768x768xf32>, %arg336: tensor<256xf32>, %arg337: tensor<256xf32>, %arg338: tensor<480x768xf32>, %arg339: tensor<16x256x64x64xf32>, %arg340: tensor<768x768xf32>, %arg341: tensor<2x240x768xf32>, %arg342: tensor<16x256x64x64xf32>, %arg343: tensor<480x768xf32>, %arg344: tensor<64xf32>, %arg345: tensor<64xf32>, %arg346: tensor<16x64x64x64xf32>, %arg347: tensor<24x240x64xf32>, %arg348: tensor<16x512x8x8xf32>, %arg349: tensor<2048xf32>, %arg350: tensor<2048xf32>, %arg351: tensor<16x2048x8x8xf32>, %arg352: tensor<512xf32>, %arg353: tensor<512xf32>, %arg354: tensor<16x512x8x8xf32>, %arg355: tensor<16x2048x8x8xf32>, %arg356: tensor<512xf32>, %arg357: tensor<512xf32>, %arg358: tensor<16x1024x16x16xf32>, %arg359: tensor<1024xf32>, %arg360: tensor<64xf32>, %arg361: tensor<24x240x240xf32>, %arg362: tensor<1024xf32>, %arg363: tensor<1024xf32>, %arg364: tensor<1024xf32>, %arg365: tensor<768x768xf32>, %arg366: tensor<16x512x16x16xf32>, %arg367: tensor<16x1024x16x16xf32>, %arg368: tensor<16x1024x16x16xf32>, %arg369: tensor<768xf32>, %arg370: tensor<480x768xf32>, %arg371: tensor<768xf32>, %arg372: tensor<16x2048x8x8xf32>, %arg373: tensor<480x768xf32>, %arg374: tensor<480x768xf32>, %arg375: tensor<16x256x16x16xf32>, %arg376: tensor<768xf32>, %arg377: tensor<16x512x8x8xf32>, %arg378: tensor<768xf32>, %arg379: tensor<256xf32>, %arg380: tensor<24x240x240xf32>, %arg381: tensor<256xf32>, %arg382: tensor<2048xf32>, %arg383: tensor<16x2048x8x8xf32>, %arg384: tensor<768x3072xf32>, %arg385: tensor<16x256x16x16xf32>, %arg386: tensor<2048xf32>, %arg387: tensor<2x240x1xf32>, %arg388: tensor<2x240x1xf32>, %arg389: tensor<2x240x3072xf32>, %arg390: tensor<480x768xf32>, %arg391: tensor<512xf32>, %arg392: tensor<24x64x240xf32>, %arg393: tensor<512xf32>, %arg394: tensor<16x512x16x16xf32>, %arg395: tensor<768xf32>, %arg396: tensor<768xf32>, %arg397: tensor<256xf32>, %arg398: tensor<256xf32>, %arg399: tensor<16x256x16x16xf32>, %arg400: tensor<768x3072xf32>, %arg401: tensor<3072x768xf32>, %arg402: tensor<2x240x1xf32>, %arg403: tensor<64xf32>, %arg404: tensor<480x3072xf32>, %arg405: tensor<2x12x240x240xf32>, %arg406: tensor<2x240x1xf32>, %arg407: tensor<64xf32>, %arg408: tensor<2x240x768xf32>, %arg409: tensor<512xf32>, %arg410: tensor<64xf32>, %arg411: tensor<480x768xf32>, %arg412: tensor<24x240x64xf32>, %arg413: tensor<2x240x768xf32>, %arg414: tensor<512xf32>, %arg415: tensor<64xf32>, %arg416: tensor<16x512x8x8xf32>, %arg417: tensor<512x1024x1x1xf32>, %arg418: tensor<2x240x3072xf32>, %arg419: tensor<256xf32>, %arg420: tensor<256x256x3x3xf32>, %arg421: tensor<24x240x240xf32>, %arg422: tensor<1024xf32>, %arg423: tensor<1024x256x1x1xf32>, %arg424: tensor<3072x768xf32>, %arg425: tensor<256xf32>, %arg426: tensor<768xf32>, %arg427: tensor<480x3072xf32>, %arg428: tensor<256x1024x1x1xf32>, %arg429: tensor<768xf32>, %arg430: tensor<256xf32>, %arg431: tensor<256x256x3x3xf32>, %arg432: tensor<768xf32>, %arg433: tensor<2x240x768xf32>, %arg434: tensor<768xf32>, %arg435: tensor<1024xf32>, %arg436: tensor<2x240x1xf32>, %arg437: tensor<1024x256x1x1xf32>, %arg438: tensor<480x768xf32>, %arg439: tensor<2x240x1xf32>, %arg440: tensor<256xf32>, %arg441: tensor<256x1024x1x1xf32>, %arg442: tensor<2x768xf32>, %arg443: tensor<256xf32>, %arg444: tensor<768x3072xf32>, %arg445: tensor<256x256x3x3xf32>, %arg446: tensor<2x240x1xf32>, %arg447: tensor<1024xf32>, %arg448: tensor<768x768xf32>, %arg449: tensor<2x240x1xf32>, %arg450: tensor<1024x256x1x1xf32>, %arg451: tensor<768xf32>, %arg452: tensor<768xf32>, %arg453: tensor<16x64x128x128xf32>, %arg454: tensor<256xf32>, %arg455: tensor<480x768xf32>, %arg456: tensor<256x1024x1x1xf32>, %arg457: tensor<768xf32>, %arg458: tensor<2x768xf32>, %arg459: tensor<256xf32>, %arg460: tensor<768xf32>, %arg461: tensor<256x256x3x3xf32>, %arg462: tensor<2x240x3072xf32>, %arg463: tensor<1024xf32>, %arg464: tensor<1024x256x1x1xf32>, %arg465: tensor<512xf32>, %arg466: tensor<768xf32>, %arg467: tensor<480x768xf32>, %arg468: tensor<768x256xf32>, %arg469: tensor<768xf32>, %arg470: tensor<2x240x1xf32>, %arg471: tensor<2x240x1xf32>, %arg472: tensor<768xf32>, %arg473: tensor<768x768xf32>, %arg474: tensor<768x768xf32>, %arg475: tensor<768xf32>, %arg476: tensor<480x768xf32>, %arg477: tensor<2x256xf32>, %arg478: tensor<768x768xf32>, %arg479: tensor<480x768xf32>, %arg480: tensor<2x240x768xf32>, %arg481: tensor<24x64x240xf32>, %arg482: tensor<480x768xf32>, %arg483: tensor<768x768xf32>, %arg484: tensor<768x768xf32>, %arg485: tensor<256x2xf32>, %arg486: tensor<768xf32>, %arg487: tensor<480x768xf32>, %arg488: tensor<768x768xf32>, %arg489: tensor<24x240x240xf32>, %arg490: tensor<768xf32>, %arg491: tensor<768xf32>, %arg492: tensor<768x768xf32>, %arg493: tensor<24x64x240xf32>, %arg494: tensor<768xf32>, %arg495: tensor<480x768xf32>, %arg496: tensor<24x240x64xf32>, %arg497: tensor<768xf32>, %arg498: tensor<2x240x768xf32>, %arg499: tensor<24x240x64xf32>, %arg500: tensor<768xf32>, %arg501: tensor<24x64x240xf32>, %arg502: tensor<24x240x64xf32>, %arg503: tensor<24x240x64xf32>, %arg504: tensor<1024xf32>, %arg505: tensor<1024xf32>, %arg506: tensor<768x3072xf32>, %arg507: tensor<16x1024x16x16xf32>, %arg508: tensor<2x240x1xf32>, %arg509: tensor<2x240x1xf32>, %arg510: tensor<1024xf32>, %arg511: tensor<1024xf32>, %arg512: tensor<2x240x3072xf32>, %arg513: tensor<480x768xf32>, %arg514: tensor<480x3072xf32>, %arg515: tensor<16x256x16x16xf32>, %arg516: tensor<16x256x16x16xf32>, %arg517: tensor<256xf32>, %arg518: tensor<256xf32>, %arg519: tensor<256xf32>, %arg520: tensor<256xf32>, %arg521: tensor<16x256x32x32xf32>, %arg522: tensor<16x256x16x16xf32>, %arg523: tensor<3072x768xf32>, %arg524: tensor<16x256x16x16xf32>, %arg525: tensor<2x240x768xf32>, %arg526: tensor<2x240x1xf32>, %arg527: tensor<256xf32>, %arg528: tensor<16x1024x16x16xf32>, %arg529: tensor<256xf32>, %arg530: tensor<2x240x1xf32>, %arg531: tensor<256xf32>, %arg532: tensor<256xf32>, %arg533: tensor<768x768xf32>, %arg534: tensor<16x256x16x16xf32>, %arg535: tensor<16x256x16x16xf32>, %arg536: tensor<480x768xf32>, %arg537: tensor<16x1024x16x16xf32>, %arg538: tensor<16x256x16x16xf32>, %arg539: tensor<768x768xf32>, %arg540: tensor<16x256x16x16xf32>, %arg541: tensor<480x768xf32>, %arg542: tensor<256xf32>, %arg543: tensor<64xf32>, %arg544: tensor<3072x768xf32>, %arg545: tensor<256x64x1x1xf32>, %arg546: tensor<64x64x3x3xf32>, %arg547: tensor<480x3072xf32>, %arg548: tensor<128xf32>, %arg549: tensor<128x256x1x1xf32>, %arg550: tensor<2x240x768xf32>, %arg551: tensor<128xf32>, %arg552: tensor<128x128x3x3xf32>, %arg553: tensor<2x240x1xf32>, %arg554: tensor<512xf32>, %arg555: tensor<2x240x1xf32>, %arg556: tensor<512x128x1x1xf32>, %arg557: tensor<768x768xf32>, %arg558: tensor<512xf32>, %arg559: tensor<480x768xf32>, %arg560: tensor<512x256x1x1xf32>, %arg561: tensor<480x768xf32>, %arg562: tensor<2x240x768xf32>, %arg563: tensor<128xf32>, %arg564: tensor<128x512x1x1xf32>, %arg565: tensor<768x768xf32>, %arg566: tensor<128xf32>, %arg567: tensor<128x128x3x3xf32>, %arg568: tensor<512xf32>, %arg569: tensor<512x128x1x1xf32>, %arg570: tensor<768x768xf32>, %arg571: tensor<128xf32>, %arg572: tensor<128x512x1x1xf32>, %arg573: tensor<128xf32>, %arg574: tensor<480x768xf32>, %arg575: tensor<24x240x64xf32>, %arg576: tensor<128x128x3x3xf32>, %arg577: tensor<2x12x240x240xf32>, %arg578: tensor<2x12x240x240xf32>, %arg579: tensor<64x64x3x3xf32>, %arg580: tensor<64x256x1x1xf32>, %arg581: tensor<24x240x64xf32>, %arg582: tensor<64xf32>, %arg583: tensor<24x240x240xf32>, %arg584: tensor<24x240x64xf32>, %arg585: tensor<256x64x1x1xf32>, %arg586: tensor<256xf32>, %arg587: tensor<64x64x3x3xf32>, %arg588: tensor<64xf32>, %arg589: tensor<64x256x1x1xf32>, %arg590: tensor<64xf32>, %arg591: tensor<480x768xf32>, %arg592: tensor<256x64x1x1xf32>, %arg593: tensor<256xf32>, %arg594: tensor<256xf32>, %arg595: tensor<256x64x1x1xf32>, %arg596: tensor<768x3072xf32>, %arg597: tensor<768x768xf32>, %arg598: tensor<768x768xf32>, %arg599: tensor<128xf32>, %arg600: tensor<128xf32>, %arg601: tensor<480x768xf32>, %arg602: tensor<16x128x64x64xf32>, %arg603: tensor<768x768xf32>, %arg604: tensor<16x128x32x32xf32>, %arg605: tensor<480x768xf32>, %arg606: tensor<128xf32>, %arg607: tensor<128xf32>, %arg608: tensor<16x128x32x32xf32>, %arg609: tensor<64xf32>, %arg610: tensor<64x64x1x1xf32>, %arg611: tensor<16x128x32x32xf32>, %arg612: tensor<16x512x32x32xf32>, %arg613: tensor<512xf32>, %arg614: tensor<24x64x240xf32>, %arg615: tensor<2x12x240x240xf32>, %arg616: tensor<512xf32>, %arg617: tensor<16x512x32x32xf32>, %arg618: tensor<512xf32>, %arg619: tensor<512xf32>, %arg620: tensor<128xf32>, %arg621: tensor<512xf32>, %arg622: tensor<128xf32>, %arg623: tensor<16x128x32x32xf32>, %arg624: tensor<2048xf32>, %arg625: tensor<2048xf32>, %arg626: tensor<16x3x256x256xf32>, %arg627: tensor<2x240xi64>, %arg628: tensor<128xf32>, %arg629: tensor<128xf32>, %arg630: tensor<16x128x32x32xf32>, %arg631: tensor<16x128x32x32xf32>, %arg632: tensor<512xf32>, %arg633: tensor<512xf32>, %arg634: tensor<16x512x32x32xf32>, %arg635: tensor<16x512x32x32xf32>, %arg636: tensor<16x128x64x64xf32>, %arg637: tensor<64xf32>, %arg638: tensor<64xf32>, %arg639: tensor<16x64x64x64xf32>, %arg640: tensor<16x64x64x64xf32>, %arg641: tensor<256xf32>, %arg642: tensor<256xf32>, %arg643: tensor<16x256x64x64xf32>, %arg644: tensor<16x64x64x64xf32>, %arg645: tensor<16x128x32x32xf32>, %arg646: tensor<16x512x32x32xf32>, %arg647: tensor<64xf32>, %arg648: tensor<24x240x64xf32>, %arg649: tensor<64xf32>, %arg650: tensor<1024xf32>, %arg651: tensor<256xf32>, %arg652: tensor<480x768xf32>, %arg653: tensor<1024xf32>, %arg654: tensor<1024xf32>, %arg655: tensor<256xf32>, %arg656: tensor<256xf32>, %arg657: tensor<256xf32>, %arg658: tensor<256xf32>, %arg659: tensor<768x3072xf32>, %arg660: tensor<1024xf32>, %arg661: tensor<1024xf32>, %arg662: tensor<2x240x1xf32>, %arg663: tensor<2x240x1xf32>, %arg664: tensor<256xf32>, %arg665: tensor<256xf32>, %arg666: tensor<256xf32>, %arg667: tensor<480x768xf32>, %arg668: tensor<256xf32>, %arg669: tensor<2x240x3072xf32>, %arg670: tensor<1024xf32>, %arg671: tensor<1024xf32>, %arg672: tensor<256xf32>, %arg673: tensor<256xf32>, %arg674: tensor<3072x768xf32>, %arg675: tensor<256xf32>, %arg676: tensor<256xf32>, %arg677: tensor<480x3072xf32>, %arg678: tensor<1024xf32>, %arg679: tensor<1024xf32>, %arg680: tensor<24x240x64xf32>, %arg681: tensor<256xf32>, %arg682: tensor<2x240x768xf32>, %arg683: tensor<256xf32>, %arg684: tensor<24x64x240xf32>, %arg685: tensor<480x768xf32>, %arg686: tensor<768x768xf32>, %arg687: tensor<24x240x240xf32>, %arg688: tensor<24x240x64xf32>, %arg689: tensor<64xf32>, %arg690: tensor<64xf32>, %arg691: tensor<16x256x64x64xf32>, %arg692: tensor<16x64x128x128xf32>, %arg693: tensor<16x256x64x64xf32>, %arg694: tensor<16x64x64x64xf32>, %arg695: tensor<16x64x64x64xf32>, %arg696: tensor<256xf32>, %arg697: tensor<256xf32>, %arg698: tensor<16x64x64x64xf32>, %arg699: tensor<16x64x64x64xf32>, %arg700: tensor<16x64x64x64xf32>, %arg701: tensor<64xf32>, %arg702: tensor<64xf32>, %arg703: tensor<16x64x64x64xf32>, %arg704: tensor<1x240xi64>, %arg705: tensor<768x768xf32>, %arg706: tensor<24x240x64xf32>, %arg707: tensor<2x240x768xf32>, %arg708: tensor<24x240x64xf32>, %arg709: tensor<2x240x1xf32>, %arg710: tensor<2x240x1xf32>, %arg711: tensor<64xf32>, %arg712: tensor<512xf32>, %arg713: tensor<3072x768xf32>, %arg714: tensor<24x240x240xf32>, %arg715: tensor<256xf32>, %arg716: tensor<256xf32>, %arg717: tensor<480x3072xf32>, %arg718: tensor<256xf32>, %arg719: tensor<256xf32>, %arg720: tensor<2x240x768xf32>, %arg721: tensor<64xf32>, %arg722: tensor<768xf32>, %arg723: tensor<64xf32>, %arg724: tensor<2x240x1xf32>, %arg725: tensor<768xf32>, %arg726: tensor<2x240x1xf32>, %arg727: tensor<64xf32>, %arg728: tensor<768x768xf32>, %arg729: tensor<64xf32>, %arg730: tensor<480x768xf32>, %arg731: tensor<768xf32>, %arg732: tensor<256xf32>, %arg733: tensor<480x768xf32>, %arg734: tensor<768xf32>, %arg735: tensor<256xf32>, %arg736: tensor<768x3072xf32>, %arg737: tensor<480x768xf32>, %arg738: tensor<64xf32>, %arg739: tensor<64xf32>, %arg740: tensor<768x768xf32>, %arg741: tensor<2x240x1xf32>, %arg742: tensor<64xf32>, %arg743: tensor<2x240x1xf32>, %arg744: tensor<64xf32>, %arg745: tensor<256xf32>, %arg746: tensor<480x768xf32>, %arg747: tensor<256xf32>, %arg748: tensor<768xf32>, %arg749: tensor<128xf32>, %arg750: tensor<768x768xf32>, %arg751: tensor<2x240x3072xf32>, %arg752: tensor<768xf32>, %arg753: tensor<128xf32>, %arg754: tensor<2x240x768xf32>, %arg755: tensor<128xf32>, %arg756: tensor<768xf32>, %arg757: tensor<128xf32>, %arg758: tensor<768x768xf32>, %arg759: tensor<768xf32>, %arg760: tensor<16x256x16x16xf32>, %arg761: tensor<512xf32>, %arg762: tensor<480x768xf32>, %arg763: tensor<3072x768xf32>, %arg764: tensor<512xf32>, %arg765: tensor<512xf32>, %arg766: tensor<512x512x3x3xf32>, %arg767: tensor<2048xf32>, %arg768: tensor<2048x512x1x1xf32>, %arg769: tensor<2048xf32>, %arg770: tensor<2048x1024x1x1xf32>, %arg771: tensor<512xf32>, %arg772: tensor<512x2048x1x1xf32>, %arg773: tensor<512xf32>, %arg774: tensor<512x512x3x3xf32>, %arg775: tensor<2048xf32>, %arg776: tensor<2048x512x1x1xf32>, %arg777: tensor<512xf32>, %arg778: tensor<512x2048x1x1xf32>, %arg779: tensor<512xf32>, %arg780: tensor<512x512x3x3xf32>, %arg781: tensor<2048xf32>, %arg782: tensor<2048x512x1x1xf32>, %arg783: tensor<2x19xf32>, %arg784: tensor<2x19xf32>, %arg785: tensor<2x2xf32>, %arg786: tensor<2x2xf32>, %arg787: tensor<19xi64>, %arg788: tensor<19xi64>, %arg789: tensor<19xi64>, %arg790: tensor<19xi64>, %arg791: tensor<0xf32>) -> !tuple {
    %0 = call @aten.permute.978(%arg299) {xla_shape = "f32[2,256]{0,1}"} : (tensor<256x2xf32>) -> tensor<2x256xf32>
    %1 = call @aten.mm.982(%arg785, %0) : (tensor<2x2xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    %2 = call @aten.threshold_backward.888(%1, %arg293) : (tensor<2x256xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    %3 = call @aten.permute.1136(%arg288) {xla_shape = "f32[256,2048]{0,1}"} : (tensor<2048x256xf32>) -> tensor<256x2048xf32>
    %4 = call @aten.mm.1140(%2, %3) : (tensor<2x256xf32>, tensor<256x2048xf32>) -> tensor<2x2048xf32>
    %5 = call @aten.permute.879(%arg284) {xla_shape = "f32[19,256]{0,1}"} : (tensor<256x19xf32>) -> tensor<19x256xf32>
    %6 = call @aten.mm.883(%arg783, %5) : (tensor<2x19xf32>, tensor<19x256xf32>) -> tensor<2x256xf32>
    %7 = call @aten.threshold_backward.888(%6, %arg277) : (tensor<2x256xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    %8 = call @aten.permute.1116(%arg274) {xla_shape = "f32[256,2816]{0,1}"} : (tensor<2816x256xf32>) -> tensor<256x2816xf32>
    %9 = call @aten.mm.1120(%7, %8) : (tensor<2x256xf32>, tensor<256x2816xf32>) -> tensor<2x2816xf32>
    %10 = call @aten.permute.927(%arg269) {xla_shape = "f32[19,128]{0,1}"} : (tensor<128x19xf32>) -> tensor<19x128xf32>
    %11 = call @aten.mm.931(%arg784, %10) : (tensor<2x19xf32>, tensor<19x128xf32>) -> tensor<2x128xf32>
    %12 = call @aten.threshold_backward.936(%11, %arg12) : (tensor<2x128xf32>, tensor<2x128xf32>) -> tensor<2x128xf32>
    %13 = call @aten.permute.1102(%arg8) {xla_shape = "f32[128,2816]{0,1}"} : (tensor<2816x128xf32>) -> tensor<128x2816xf32>
    %14 = call @aten.mm.1106(%12, %13) : (tensor<2x128xf32>, tensor<128x2816xf32>) -> tensor<2x2816xf32>
    %15 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %16 = call @aten.expand.1095(%15) : (tensor<f32>) -> tensor<2x2816xf32>
    %17 = call @aten.mul.1111(%14, %16) : (tensor<2x2816xf32>, tensor<2x2816xf32>) -> tensor<2x2816xf32>
    %18 = call @aten.add.1125(%9, %17) : (tensor<2x2816xf32>, tensor<2x2816xf32>) -> tensor<2x2816xf32>
    %19 = "mhlo.slice"(%18) {limit_indices = dense<[2, 2048]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2816xf32>) -> tensor<2x2048xf32>
    %20 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %21 = call @aten.expand.1087(%20) : (tensor<f32>) -> tensor<2x2048xf32>
    %22 = call @aten.mul.1131(%19, %21) : (tensor<2x2048xf32>, tensor<2x2048xf32>) -> tensor<2x2048xf32>
    %23 = call @aten.add.1145(%4, %22) : (tensor<2x2048xf32>, tensor<2x2048xf32>) -> tensor<2x2048xf32>
    %24 = call @aten.view.1150(%23) : (tensor<2x2048xf32>) -> tensor<2x2048x1xf32>
    %25 = call @aten.view.1154(%24) : (tensor<2x2048x1xf32>) -> tensor<2x2048x1x1xf32>
    %26 = call @aten.view.1158(%25) : (tensor<2x2048x1x1xf32>) -> tensor<2x2048x1x1x1xf32>
    %27 = call @aten.expand.1162(%26) : (tensor<2x2048x1x1x1xf32>) -> tensor<2x2048x8x8x8xf32>
    %28 = mhlo.constant dense<5.120000e+02> : tensor<f32>
    %29 = call @aten.div.1168(%27, %28) : (tensor<2x2048x8x8x8xf32>, tensor<f32>) -> tensor<2x2048x8x8x8xf32>
    %30 = call @aten.permute.1174(%29) {xla_shape = "f32[2,8,2048,8,8]{4,3,1,2,0}"} : (tensor<2x2048x8x8x8xf32>) -> tensor<2x8x2048x8x8xf32>
    %31 = call @aten.view.1178(%30) : (tensor<2x8x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %32 = call @aten.threshold_backward.1182(%31, %arg5) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %33 = call @aten.native_batch_norm_backward.1192(%32, %arg355, %arg781, %arg2, %arg3) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>
    %34 = "mhlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<16x2048x8x8xf32>
    %35 = call @aten.convolution_backward_overrideable.1218(%34, %arg0, %arg782) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[2048,512,1,1]{0,1,3,2}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>
    %36 = "mhlo.get_tuple_element"(%35) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %37 = "mhlo.get_tuple_element"(%35) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<16x512x8x8xf32>
    %38 = call @aten.threshold_backward.1234(%37, %arg0) : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %39 = call @aten.native_batch_norm_backward.1244(%38, %arg348, %arg779, %arg356, %arg357) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>
    %40 = "mhlo.get_tuple_element"(%39) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %41 = call @aten.convolution_backward_overrideable.1270(%40, %arg354, %arg780) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512,512,3,3]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>
    %42 = "mhlo.get_tuple_element"(%41) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %43 = "mhlo.get_tuple_element"(%41) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %44 = call @aten.threshold_backward.1234(%43, %arg354) : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %45 = call @aten.native_batch_norm_backward.1244(%44, %arg51, %arg777, %arg352, %arg353) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>
    %46 = "mhlo.get_tuple_element"(%45) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %47 = call @aten.convolution_backward_overrideable.1295(%46, %arg351, %arg778) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[512,2048,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>
    %48 = "mhlo.get_tuple_element"(%47) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %49 = "mhlo.get_tuple_element"(%47) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>) -> tensor<16x2048x8x8xf32>
    %50 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %51 = call @aten.expand.1076(%50) : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    %52 = call @aten.mul.1311(%49, %51) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %53 = call @aten.add.1316(%32, %52) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %54 = call @aten.threshold_backward.1182(%53, %arg351) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %55 = call @aten.native_batch_norm_backward.1192(%54, %arg383, %arg775, %arg349, %arg350) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>
    %56 = "mhlo.get_tuple_element"(%55) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<16x2048x8x8xf32>
    %57 = call @aten.convolution_backward_overrideable.1218(%56, %arg65, %arg776) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[2048,512,1,1]{0,1,3,2}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>
    %58 = "mhlo.get_tuple_element"(%57) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %59 = "mhlo.get_tuple_element"(%57) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<16x512x8x8xf32>
    %60 = call @aten.threshold_backward.1234(%59, %arg65) : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %61 = call @aten.native_batch_norm_backward.1244(%60, %arg52, %arg773, %arg67, %arg69) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>
    %62 = "mhlo.get_tuple_element"(%61) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %63 = call @aten.convolution_backward_overrideable.1270(%62, %arg59, %arg774) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512,512,3,3]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>
    %64 = "mhlo.get_tuple_element"(%63) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %65 = "mhlo.get_tuple_element"(%63) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %66 = call @aten.threshold_backward.1234(%65, %arg59) : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %67 = call @aten.native_batch_norm_backward.1244(%66, %arg377, %arg771, %arg56, %arg57) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>
    %68 = "mhlo.get_tuple_element"(%67) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %69 = call @aten.convolution_backward_overrideable.1295(%68, %arg49, %arg772) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[512,2048,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>
    %70 = "mhlo.get_tuple_element"(%69) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %71 = "mhlo.get_tuple_element"(%69) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>) -> tensor<16x2048x8x8xf32>
    %72 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %73 = call @aten.expand.1076(%72) : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    %74 = call @aten.mul.1311(%71, %73) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %75 = call @aten.add.1316(%54, %74) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %76 = call @aten.threshold_backward.1182(%75, %arg49) : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    %77 = call @aten.native_batch_norm_backward.1192(%76, %arg14, %arg769, %arg382, %arg386) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>
    %78 = "mhlo.get_tuple_element"(%77) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<16x2048x8x8xf32>
    %79 = call @aten.convolution_backward_overrideable.1359(%78, %arg368, %arg770) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[2048,1024,1,1]{0,1,3,2}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>>
    %80 = "mhlo.get_tuple_element"(%79) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %81 = call @aten.native_batch_norm_backward.1192(%76, %arg372, %arg767, %arg44, %arg46) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>
    %82 = "mhlo.get_tuple_element"(%81) {index = 0 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<16x2048x8x8xf32>
    %83 = call @aten.convolution_backward_overrideable.1218(%82, %arg19, %arg768) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[2048,512,1,1]{0,1,3,2}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>
    %84 = "mhlo.get_tuple_element"(%83) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %85 = "mhlo.get_tuple_element"(%83) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<16x512x8x8xf32>
    %86 = call @aten.threshold_backward.1234(%85, %arg19) : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    %87 = call @aten.native_batch_norm_backward.1244(%86, %arg416, %arg765, %arg409, %arg414) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>
    %88 = "mhlo.get_tuple_element"(%87) {index = 0 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x8x8xf32>
    %89 = call @aten.convolution_backward_overrideable.1397(%88, %arg394, %arg766) {xla_shape = "(f32[16,512,16,16]{3,2,1,0}, f32[512,512,3,3]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>
    %90 = "mhlo.get_tuple_element"(%89) {index = 2 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %91 = "mhlo.get_tuple_element"(%89) {index = 0 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<16x512x16x16xf32>
    %92 = call @aten.threshold_backward.1413(%91, %arg394) : (tensor<16x512x16x16xf32>, tensor<16x512x16x16xf32>) -> tensor<16x512x16x16xf32>
    %93 = call @aten.native_batch_norm_backward.1423(%92, %arg366, %arg465, %arg391, %arg393) {xla_shape = "(f32[16,512,16,16]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x16x16xf32>, tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>>
    %94 = "mhlo.get_tuple_element"(%93) {index = 0 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x16x16xf32>
    %95 = call @aten.convolution_backward_overrideable.1449(%94, %arg368, %arg417) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[512,1024,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>>
    %96 = "mhlo.get_tuple_element"(%95) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %97 = "mhlo.get_tuple_element"(%95) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>>) -> tensor<16x1024x16x16xf32>
    %98 = "mhlo.get_tuple_element"(%79) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>>) -> tensor<16x1024x16x16xf32>
    %99 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %100 = call @aten.expand.1058(%99) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %101 = call @aten.mul.1375(%98, %100) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %102 = call @aten.add.1465(%97, %101) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %103 = call @aten.threshold_backward.1470(%102, %arg368) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %104 = call @aten.native_batch_norm_backward.1480(%103, %arg21, %arg463, %arg362, %arg364) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %105 = "mhlo.get_tuple_element"(%104) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %106 = call @aten.convolution_backward_overrideable.1506(%105, %arg32, %arg464) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[1024,256,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>
    %107 = "mhlo.get_tuple_element"(%106) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %108 = "mhlo.get_tuple_element"(%106) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<16x256x16x16xf32>
    %109 = call @aten.threshold_backward.1522(%108, %arg32) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %110 = call @aten.native_batch_norm_backward.1532(%109, %arg181, %arg459, %arg30, %arg31) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %111 = "mhlo.get_tuple_element"(%110) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %112 = call @aten.convolution_backward_overrideable.1558(%111, %arg25, %arg461) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
    %113 = "mhlo.get_tuple_element"(%112) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %114 = "mhlo.get_tuple_element"(%112) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %115 = call @aten.threshold_backward.1522(%114, %arg25) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %116 = call @aten.native_batch_norm_backward.1532(%115, %arg375, %arg454, %arg22, %arg23) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %117 = "mhlo.get_tuple_element"(%116) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %118 = call @aten.convolution_backward_overrideable.1583(%117, %arg18, %arg456) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>
    %119 = "mhlo.get_tuple_element"(%118) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %120 = "mhlo.get_tuple_element"(%118) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<16x1024x16x16xf32>
    %121 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %122 = call @aten.expand.1058(%121) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %123 = call @aten.mul.1375(%120, %122) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %124 = call @aten.add.1465(%103, %123) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %125 = call @aten.threshold_backward.1470(%124, %arg18) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %126 = call @aten.native_batch_norm_backward.1480(%125, %arg358, %arg447, %arg15, %arg16) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %127 = "mhlo.get_tuple_element"(%126) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %128 = call @aten.convolution_backward_overrideable.1506(%127, %arg399, %arg450) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[1024,256,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>
    %129 = "mhlo.get_tuple_element"(%128) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %130 = "mhlo.get_tuple_element"(%128) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<16x256x16x16xf32>
    %131 = call @aten.threshold_backward.1522(%130, %arg399) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %132 = call @aten.native_batch_norm_backward.1532(%131, %arg540, %arg443, %arg397, %arg398) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %133 = "mhlo.get_tuple_element"(%132) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %134 = call @aten.convolution_backward_overrideable.1558(%133, %arg385, %arg445) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
    %135 = "mhlo.get_tuple_element"(%134) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %136 = "mhlo.get_tuple_element"(%134) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %137 = call @aten.threshold_backward.1522(%136, %arg385) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %138 = call @aten.native_batch_norm_backward.1532(%137, %arg172, %arg440, %arg379, %arg381) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %139 = "mhlo.get_tuple_element"(%138) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %140 = call @aten.convolution_backward_overrideable.1583(%139, %arg367, %arg441) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>
    %141 = "mhlo.get_tuple_element"(%140) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %142 = "mhlo.get_tuple_element"(%140) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<16x1024x16x16xf32>
    %143 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %144 = call @aten.expand.1058(%143) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %145 = call @aten.mul.1375(%142, %144) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %146 = call @aten.add.1465(%125, %145) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %147 = call @aten.threshold_backward.1470(%146, %arg367) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %148 = call @aten.native_batch_norm_backward.1480(%147, %arg165, %arg435, %arg359, %arg363) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %149 = "mhlo.get_tuple_element"(%148) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %150 = call @aten.convolution_backward_overrideable.1506(%149, %arg188, %arg437) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[1024,256,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>
    %151 = "mhlo.get_tuple_element"(%150) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %152 = "mhlo.get_tuple_element"(%150) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<16x256x16x16xf32>
    %153 = call @aten.threshold_backward.1522(%152, %arg188) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %154 = call @aten.native_batch_norm_backward.1532(%153, %arg538, %arg430, %arg182, %arg184) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %155 = "mhlo.get_tuple_element"(%154) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %156 = call @aten.convolution_backward_overrideable.1558(%155, %arg176, %arg431) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
    %157 = "mhlo.get_tuple_element"(%156) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %158 = "mhlo.get_tuple_element"(%156) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %159 = call @aten.threshold_backward.1522(%158, %arg176) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %160 = call @aten.native_batch_norm_backward.1532(%159, %arg516, %arg425, %arg173, %arg174) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %161 = "mhlo.get_tuple_element"(%160) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %162 = call @aten.convolution_backward_overrideable.1583(%161, %arg167, %arg428) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>
    %163 = "mhlo.get_tuple_element"(%162) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %164 = "mhlo.get_tuple_element"(%162) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<16x1024x16x16xf32>
    %165 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %166 = call @aten.expand.1058(%165) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %167 = call @aten.mul.1375(%164, %166) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %168 = call @aten.add.1465(%147, %167) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %169 = call @aten.threshold_backward.1470(%168, %arg167) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %170 = call @aten.native_batch_norm_backward.1480(%169, %arg528, %arg422, %arg161, %arg163) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %171 = "mhlo.get_tuple_element"(%170) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %172 = call @aten.convolution_backward_overrideable.1506(%171, %arg535, %arg423) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[1024,256,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>
    %173 = "mhlo.get_tuple_element"(%172) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %174 = "mhlo.get_tuple_element"(%172) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<16x256x16x16xf32>
    %175 = call @aten.threshold_backward.1522(%174, %arg535) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %176 = call @aten.native_batch_norm_backward.1532(%175, %arg141, %arg419, %arg529, %arg532) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %177 = "mhlo.get_tuple_element"(%176) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %178 = call @aten.convolution_backward_overrideable.1558(%177, %arg522, %arg420) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
    %179 = "mhlo.get_tuple_element"(%178) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %180 = "mhlo.get_tuple_element"(%178) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %181 = call @aten.threshold_backward.1522(%180, %arg522) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %182 = call @aten.native_batch_norm_backward.1532(%181, %arg150, %arg227, %arg518, %arg520) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %183 = "mhlo.get_tuple_element"(%182) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %184 = call @aten.convolution_backward_overrideable.1583(%183, %arg507, %arg217) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>
    %185 = "mhlo.get_tuple_element"(%184) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %186 = "mhlo.get_tuple_element"(%184) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<16x1024x16x16xf32>
    %187 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %188 = call @aten.expand.1058(%187) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %189 = call @aten.mul.1375(%186, %188) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %190 = call @aten.add.1465(%169, %189) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %191 = call @aten.threshold_backward.1470(%190, %arg507) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %192 = call @aten.native_batch_norm_backward.1480(%191, %arg152, %arg223, %arg504, %arg505) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %193 = "mhlo.get_tuple_element"(%192) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %194 = call @aten.convolution_backward_overrideable.1506(%193, %arg155, %arg225) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[1024,256,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>
    %195 = "mhlo.get_tuple_element"(%194) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %196 = "mhlo.get_tuple_element"(%194) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<16x256x16x16xf32>
    %197 = call @aten.threshold_backward.1522(%196, %arg155) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %198 = call @aten.native_batch_norm_backward.1532(%197, %arg524, %arg221, %arg153, %arg154) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %199 = "mhlo.get_tuple_element"(%198) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %200 = call @aten.convolution_backward_overrideable.1558(%199, %arg145, %arg222) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
    %201 = "mhlo.get_tuple_element"(%200) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %202 = "mhlo.get_tuple_element"(%200) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %203 = call @aten.threshold_backward.1522(%202, %arg145) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %204 = call @aten.native_batch_norm_backward.1532(%203, %arg515, %arg219, %arg142, %arg143) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %205 = "mhlo.get_tuple_element"(%204) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %206 = call @aten.convolution_backward_overrideable.1583(%205, %arg136, %arg220) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>
    %207 = "mhlo.get_tuple_element"(%206) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %208 = "mhlo.get_tuple_element"(%206) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<16x1024x16x16xf32>
    %209 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %210 = call @aten.expand.1058(%209) : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %211 = call @aten.mul.1375(%208, %210) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %212 = call @aten.add.1465(%191, %211) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %213 = call @aten.threshold_backward.1470(%212, %arg136) : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    %214 = call @aten.native_batch_norm_backward.1480(%213, %arg151, %arg215, %arg510, %arg511) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %215 = "mhlo.get_tuple_element"(%214) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %216 = call @aten.convolution_backward_overrideable.1726(%215, %arg160, %arg216) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[1024,512,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>>
    %217 = "mhlo.get_tuple_element"(%216) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %218 = call @aten.native_batch_norm_backward.1480(%213, %arg537, %arg212, %arg132, %arg133) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
    %219 = "mhlo.get_tuple_element"(%218) {index = 0 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<16x1024x16x16xf32>
    %220 = call @aten.convolution_backward_overrideable.1506(%219, %arg534, %arg213) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[1024,256,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>
    %221 = "mhlo.get_tuple_element"(%220) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %222 = "mhlo.get_tuple_element"(%220) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<16x256x16x16xf32>
    %223 = call @aten.threshold_backward.1522(%222, %arg534) : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    %224 = call @aten.native_batch_norm_backward.1532(%223, %arg760, %arg209, %arg527, %arg531) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    %225 = "mhlo.get_tuple_element"(%224) {index = 0 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x16x16xf32>
    %226 = call @aten.convolution_backward_overrideable.1764(%225, %arg521, %arg211) {xla_shape = "(f32[16,256,32,32]{3,2,1,0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
    %227 = "mhlo.get_tuple_element"(%226) {index = 2 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %228 = "mhlo.get_tuple_element"(%226) {index = 0 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<16x256x32x32xf32>
    %229 = call @aten.threshold_backward.1780(%228, %arg521) : (tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>) -> tensor<16x256x32x32xf32>
    %230 = call @aten.native_batch_norm_backward.1790(%229, %arg156, %arg206, %arg517, %arg519) {xla_shape = "(f32[16,256,32,32]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>>
    %231 = "mhlo.get_tuple_element"(%230) {index = 0 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x32x32xf32>
    %232 = call @aten.convolution_backward_overrideable.1816(%231, %arg160, %arg208) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[256,512,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x32x32xf32>, tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>>
    %233 = "mhlo.get_tuple_element"(%232) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %234 = "mhlo.get_tuple_element"(%232) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>>) -> tensor<16x512x32x32xf32>
    %235 = "mhlo.get_tuple_element"(%216) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>>) -> tensor<16x512x32x32xf32>
    %236 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %237 = call @aten.expand.1044(%236) : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %238 = call @aten.mul.1742(%235, %237) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %239 = call @aten.add.1832(%234, %238) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %240 = call @aten.threshold_backward.1837(%239, %arg160) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %241 = call @aten.native_batch_norm_backward.1847(%240, %arg634, %arg204, %arg157, %arg158) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>
    %242 = "mhlo.get_tuple_element"(%241) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %243 = call @aten.convolution_backward_overrideable.1873(%242, %arg148, %arg205) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[512,128,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>
    %244 = "mhlo.get_tuple_element"(%243) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %245 = "mhlo.get_tuple_element"(%243) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<16x128x32x32xf32>
    %246 = call @aten.threshold_backward.1889(%245, %arg148) : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %247 = call @aten.native_batch_norm_backward.1899(%246, %arg171, %arg200, %arg146, %arg147) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
    %248 = "mhlo.get_tuple_element"(%247) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %249 = call @aten.convolution_backward_overrideable.1925(%248, %arg139, %arg202) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128,128,3,3]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>
    %250 = "mhlo.get_tuple_element"(%249) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %251 = "mhlo.get_tuple_element"(%249) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %252 = call @aten.threshold_backward.1889(%251, %arg139) : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %253 = call @aten.native_batch_norm_backward.1899(%252, %arg631, %arg194, %arg137, %arg138) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
    %254 = "mhlo.get_tuple_element"(%253) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %255 = call @aten.convolution_backward_overrideable.1950(%254, %arg635, %arg197) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[128,512,1,1]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>
    %256 = "mhlo.get_tuple_element"(%255) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %257 = "mhlo.get_tuple_element"(%255) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<16x512x32x32xf32>
    %258 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %259 = call @aten.expand.1044(%258) : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %260 = call @aten.mul.1742(%257, %259) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %261 = call @aten.add.1832(%240, %260) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %262 = call @aten.threshold_backward.1837(%261, %arg635) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %263 = call @aten.native_batch_norm_backward.1847(%262, %arg185, %arg196, %arg632, %arg633) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>
    %264 = "mhlo.get_tuple_element"(%263) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %265 = call @aten.convolution_backward_overrideable.1873(%264, %arg630, %arg199) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[512,128,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>
    %266 = "mhlo.get_tuple_element"(%265) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %267 = "mhlo.get_tuple_element"(%265) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<16x128x32x32xf32>
    %268 = call @aten.threshold_backward.1889(%267, %arg630) : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %269 = call @aten.native_batch_norm_backward.1899(%268, %arg604, %arg573, %arg628, %arg629) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
    %270 = "mhlo.get_tuple_element"(%269) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %271 = call @aten.convolution_backward_overrideable.1925(%270, %arg623, %arg576) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128,128,3,3]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>
    %272 = "mhlo.get_tuple_element"(%271) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %273 = "mhlo.get_tuple_element"(%271) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %274 = call @aten.threshold_backward.1889(%273, %arg623) : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %275 = call @aten.native_batch_norm_backward.1899(%274, %arg180, %arg571, %arg620, %arg622) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
    %276 = "mhlo.get_tuple_element"(%275) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %277 = call @aten.convolution_backward_overrideable.1950(%276, %arg191, %arg572) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[128,512,1,1]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>
    %278 = "mhlo.get_tuple_element"(%277) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %279 = "mhlo.get_tuple_element"(%277) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<16x512x32x32xf32>
    %280 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %281 = call @aten.expand.1044(%280) : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %282 = call @aten.mul.1742(%279, %281) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %283 = call @aten.add.1832(%262, %282) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %284 = call @aten.threshold_backward.1837(%283, %arg191) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %285 = call @aten.native_batch_norm_backward.1847(%284, %arg612, %arg568, %arg187, %arg190) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>
    %286 = "mhlo.get_tuple_element"(%285) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %287 = call @aten.convolution_backward_overrideable.1873(%286, %arg178, %arg569) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[512,128,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>
    %288 = "mhlo.get_tuple_element"(%287) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %289 = "mhlo.get_tuple_element"(%287) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<16x128x32x32xf32>
    %290 = call @aten.threshold_backward.1889(%289, %arg178) : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %291 = call @aten.native_batch_norm_backward.1899(%290, %arg98, %arg566, %arg175, %arg177) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
    %292 = "mhlo.get_tuple_element"(%291) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %293 = call @aten.convolution_backward_overrideable.1925(%292, %arg169, %arg567) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128,128,3,3]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>
    %294 = "mhlo.get_tuple_element"(%293) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %295 = "mhlo.get_tuple_element"(%293) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %296 = call @aten.threshold_backward.1889(%295, %arg169) : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %297 = call @aten.native_batch_norm_backward.1899(%296, %arg611, %arg563, %arg166, %arg168) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
    %298 = "mhlo.get_tuple_element"(%297) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %299 = call @aten.convolution_backward_overrideable.1950(%298, %arg617, %arg564) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[128,512,1,1]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>
    %300 = "mhlo.get_tuple_element"(%299) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %301 = "mhlo.get_tuple_element"(%299) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<16x512x32x32xf32>
    %302 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %303 = call @aten.expand.1044(%302) : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %304 = call @aten.mul.1742(%301, %303) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %305 = call @aten.add.1832(%284, %304) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %306 = call @aten.threshold_backward.1837(%305, %arg617) : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    %307 = call @aten.native_batch_norm_backward.1847(%306, %arg646, %arg558, %arg243, %arg244) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>
    %308 = "mhlo.get_tuple_element"(%307) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %309 = call @aten.convolution_backward_overrideable.2035(%308, %arg239, %arg560) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[512,256,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>>
    %310 = "mhlo.get_tuple_element"(%309) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %311 = call @aten.native_batch_norm_backward.1847(%306, %arg242, %arg554, %arg613, %arg616) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>
    %312 = "mhlo.get_tuple_element"(%311) {index = 0 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<16x512x32x32xf32>
    %313 = call @aten.convolution_backward_overrideable.1873(%312, %arg608, %arg556) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[512,128,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>
    %314 = "mhlo.get_tuple_element"(%313) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %315 = "mhlo.get_tuple_element"(%313) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<16x128x32x32xf32>
    %316 = call @aten.threshold_backward.1889(%315, %arg608) : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    %317 = call @aten.native_batch_norm_backward.1899(%316, %arg645, %arg551, %arg606, %arg607) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
    %318 = "mhlo.get_tuple_element"(%317) {index = 0 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x32x32xf32>
    %319 = call @aten.convolution_backward_overrideable.2073(%318, %arg602, %arg552) {xla_shape = "(f32[16,128,64,64]{3,2,1,0}, f32[128,128,3,3]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>
    %320 = "mhlo.get_tuple_element"(%319) {index = 2 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %321 = "mhlo.get_tuple_element"(%319) {index = 0 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<16x128x64x64xf32>
    %322 = call @aten.threshold_backward.2089(%321, %arg602) : (tensor<16x128x64x64xf32>, tensor<16x128x64x64xf32>) -> tensor<16x128x64x64xf32>
    %323 = call @aten.native_batch_norm_backward.2099(%322, %arg636, %arg548, %arg599, %arg600) {xla_shape = "(f32[16,128,64,64]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x64x64xf32>, tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>>
    %324 = "mhlo.get_tuple_element"(%323) {index = 0 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<16x128x64x64xf32>
    %325 = call @aten.convolution_backward_overrideable.2125(%324, %arg239, %arg549) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[128,256,1,1]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x64x64xf32>, tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>>
    %326 = "mhlo.get_tuple_element"(%325) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %327 = "mhlo.get_tuple_element"(%325) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>>) -> tensor<16x256x64x64xf32>
    %328 = "mhlo.get_tuple_element"(%309) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>>) -> tensor<16x256x64x64xf32>
    %329 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %330 = call @aten.expand.1032(%329) : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %331 = call @aten.mul.2051(%328, %330) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %332 = call @aten.add.2141(%327, %331) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %333 = call @aten.threshold_backward.2146(%332, %arg239) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %334 = call @aten.native_batch_norm_backward.2156(%333, %arg229, %arg542, %arg231, %arg237) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>
    %335 = "mhlo.get_tuple_element"(%334) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x64x64xf32>
    %336 = call @aten.convolution_backward_overrideable.2182(%335, %arg235, %arg545) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[256,64,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>
    %337 = "mhlo.get_tuple_element"(%336) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %338 = "mhlo.get_tuple_element"(%336) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<16x64x64x64xf32>
    %339 = call @aten.threshold_backward.2198(%338, %arg235) : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %340 = call @aten.native_batch_norm_backward.2208(%339, %arg640, %arg543, %arg232, %arg234) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>
    %341 = "mhlo.get_tuple_element"(%340) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %342 = call @aten.convolution_backward_overrideable.2234(%341, %arg644, %arg546) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64,64,3,3]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>
    %343 = "mhlo.get_tuple_element"(%342) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %344 = "mhlo.get_tuple_element"(%342) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %345 = call @aten.threshold_backward.2198(%344, %arg644) : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %346 = call @aten.native_batch_norm_backward.2208(%345, %arg334, %arg582, %arg647, %arg649) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>
    %347 = "mhlo.get_tuple_element"(%346) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %348 = call @aten.convolution_backward_overrideable.2259(%347, %arg643, %arg580) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[64,256,1,1]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>
    %349 = "mhlo.get_tuple_element"(%348) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %350 = "mhlo.get_tuple_element"(%348) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>) -> tensor<16x256x64x64xf32>
    %351 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %352 = call @aten.expand.1032(%351) : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %353 = call @aten.mul.2051(%350, %352) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %354 = call @aten.add.2141(%333, %353) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %355 = call @aten.threshold_backward.2146(%354, %arg643) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %356 = call @aten.native_batch_norm_backward.2156(%355, %arg342, %arg586, %arg641, %arg642) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>
    %357 = "mhlo.get_tuple_element"(%356) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x64x64xf32>
    %358 = call @aten.convolution_backward_overrideable.2182(%357, %arg639, %arg585) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[256,64,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>
    %359 = "mhlo.get_tuple_element"(%358) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %360 = "mhlo.get_tuple_element"(%358) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<16x64x64x64xf32>
    %361 = call @aten.threshold_backward.2198(%360, %arg639) : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %362 = call @aten.native_batch_norm_backward.2208(%361, %arg698, %arg588, %arg637, %arg638) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>
    %363 = "mhlo.get_tuple_element"(%362) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %364 = call @aten.convolution_backward_overrideable.2234(%363, %arg346, %arg587) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64,64,3,3]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>
    %365 = "mhlo.get_tuple_element"(%364) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %366 = "mhlo.get_tuple_element"(%364) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %367 = call @aten.threshold_backward.2198(%366, %arg346) : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %368 = call @aten.native_batch_norm_backward.2208(%367, %arg699, %arg590, %arg344, %arg345) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>
    %369 = "mhlo.get_tuple_element"(%368) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %370 = call @aten.convolution_backward_overrideable.2259(%369, %arg339, %arg589) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[64,256,1,1]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>
    %371 = "mhlo.get_tuple_element"(%370) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %372 = "mhlo.get_tuple_element"(%370) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>) -> tensor<16x256x64x64xf32>
    %373 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %374 = call @aten.expand.1032(%373) : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %375 = call @aten.mul.2051(%372, %374) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %376 = call @aten.add.2141(%355, %375) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %377 = call @aten.threshold_backward.2146(%376, %arg339) : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    %378 = call @aten.native_batch_norm_backward.2156(%377, %arg693, %arg594, %arg696, %arg697) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>
    %379 = "mhlo.get_tuple_element"(%378) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x64x64xf32>
    %380 = call @aten.convolution_backward_overrideable.2182(%379, %arg694, %arg592) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[256,64,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>
    %381 = "mhlo.get_tuple_element"(%380) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %382 = call @aten.native_batch_norm_backward.2156(%377, %arg691, %arg593, %arg336, %arg337) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>
    %383 = "mhlo.get_tuple_element"(%382) {index = 0 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<16x256x64x64xf32>
    %384 = call @aten.convolution_backward_overrideable.2182(%383, %arg330, %arg595) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[256,64,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>
    %385 = "mhlo.get_tuple_element"(%384) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %386 = "mhlo.get_tuple_element"(%384) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<16x64x64x64xf32>
    %387 = call @aten.threshold_backward.2198(%386, %arg330) : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %388 = call @aten.native_batch_norm_backward.2208(%387, %arg700, %arg609, %arg327, %arg328) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>
    %389 = "mhlo.get_tuple_element"(%388) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %390 = call @aten.convolution_backward_overrideable.2234(%389, %arg703, %arg579) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64,64,3,3]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>
    %391 = "mhlo.get_tuple_element"(%390) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %392 = "mhlo.get_tuple_element"(%390) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %393 = call @aten.threshold_backward.2198(%392, %arg703) : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %394 = call @aten.native_batch_norm_backward.2208(%393, %arg695, %arg164, %arg701, %arg702) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>
    %395 = "mhlo.get_tuple_element"(%394) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %396 = call @aten.convolution_backward_overrideable.2346(%395, %arg694, %arg610) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64,64,1,1]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>>
    %397 = "mhlo.get_tuple_element"(%396) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %398 = "mhlo.get_tuple_element"(%396) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>>) -> tensor<16x64x64x64xf32>
    %399 = "mhlo.get_tuple_element"(%380) {index = 0 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<16x64x64x64xf32>
    %400 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %401 = call @aten.expand.1024(%400) : (tensor<f32>) -> tensor<16x64x64x64xf32>
    %402 = call @aten.mul.2315(%399, %401) : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %403 = call @aten.add.2362(%398, %402) : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    %404 = call @aten.max_pool2d_with_indices_backward.2375(%403, %arg692) : (tensor<16x64x64x64xf32>, tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32>
    %405 = call @aten.threshold_backward.2381(%404, %arg692) : (tensor<16x64x128x128xf32>, tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32>
    %406 = call @aten.native_batch_norm_backward.2391(%405, %arg453, %arg162, %arg689, %arg690) {xla_shape = "(f32[16,64,128,128]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x128x128xf32>, tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>>
    %407 = "mhlo.get_tuple_element"(%406) {index = 0 : i32} : (tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<16x64x128x128xf32>
    %408 = call @aten.convolution_backward_overrideable.2417(%407, %arg626, %arg144) {xla_shape = "(f32[16,3,256,256]{3,2,1,0}, f32[64,3,7,7]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x128x128xf32>, tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>) -> tuple<tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>>
    %409 = "mhlo.get_tuple_element"(%408) {index = 0 : i32} : (tuple<tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>>) -> tensor<16x3x256x256xf32>
    %410 = "mhlo.get_tuple_element"(%408) {index = 2 : i32} : (tuple<tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %411 = call @aten.sum.797(%arg783) : (tensor<2x19xf32>) -> tensor<1x19xf32>
    %412 = call @aten.view.804(%411) : (tensor<1x19xf32>) -> tensor<19xf32>
    %413 = call @aten.permute.808(%arg783) {xla_shape = "f32[19,2]{0,1}"} : (tensor<2x19xf32>) -> tensor<19x2xf32>
    %414 = call @aten.mm.812(%413, %arg277) : (tensor<19x2xf32>, tensor<2x256xf32>) -> tensor<19x256xf32>
    %415 = call @aten.permute.817(%414) {xla_shape = "f32[256,19]{0,1}"} : (tensor<19x256xf32>) -> tensor<256x19xf32>
    %416 = call @aten.permute.821(%415) : (tensor<256x19xf32>) -> tensor<19x256xf32>
    %417 = call @aten.sum.797(%arg784) : (tensor<2x19xf32>) -> tensor<1x19xf32>
    %418 = call @aten.view.804(%417) : (tensor<1x19xf32>) -> tensor<19xf32>
    %419 = call @aten.permute.808(%arg784) {xla_shape = "f32[19,2]{0,1}"} : (tensor<2x19xf32>) -> tensor<19x2xf32>
    %420 = call @aten.mm.828(%419, %arg12) : (tensor<19x2xf32>, tensor<2x128xf32>) -> tensor<19x128xf32>
    %421 = call @aten.permute.833(%420) {xla_shape = "f32[128,19]{0,1}"} : (tensor<19x128xf32>) -> tensor<128x19xf32>
    %422 = call @aten.permute.837(%421) : (tensor<128x19xf32>) -> tensor<19x128xf32>
    %423 = call @aten.sum.845(%arg785) : (tensor<2x2xf32>) -> tensor<1x2xf32>
    %424 = call @aten.view.852(%423) : (tensor<1x2xf32>) -> tensor<2xf32>
    %425 = call @aten.permute.856(%arg785) {xla_shape = "f32[2,2]{0,1}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %426 = call @aten.mm.860(%425, %arg293) : (tensor<2x2xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    %427 = call @aten.permute.865(%426) {xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    %428 = call @aten.permute.869(%427) : (tensor<256x2xf32>) -> tensor<2x256xf32>
    %429 = call @aten.sum.845(%arg786) : (tensor<2x2xf32>) -> tensor<1x2xf32>
    %430 = call @aten.view.852(%429) : (tensor<1x2xf32>) -> tensor<2xf32>
    %431 = call @aten.permute.856(%arg786) {xla_shape = "f32[2,2]{0,1}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %432 = call @aten.mm.860(%431, %arg477) : (tensor<2x2xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    %433 = call @aten.permute.865(%432) {xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    %434 = call @aten.permute.869(%433) : (tensor<256x2xf32>) -> tensor<2x256xf32>
    %435 = call @aten.sum.902(%7) : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %436 = call @aten.view.909(%435) : (tensor<1x256xf32>) -> tensor<256xf32>
    %437 = call @aten.permute.865(%7) {xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    %438 = call @aten.mm.914(%437, %arg7) : (tensor<256x2xf32>, tensor<2x2816xf32>) -> tensor<256x2816xf32>
    %439 = call @aten.permute.919(%438) {xla_shape = "f32[2816,256]{0,1}"} : (tensor<256x2816xf32>) -> tensor<2816x256xf32>
    %440 = call @aten.permute.923(%439) : (tensor<2816x256xf32>) -> tensor<256x2816xf32>
    %441 = call @aten.sum.950(%12) : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %442 = call @aten.view.957(%441) : (tensor<1x128xf32>) -> tensor<128xf32>
    %443 = call @aten.permute.961(%12) {xla_shape = "f32[128,2]{0,1}"} : (tensor<2x128xf32>) -> tensor<128x2xf32>
    %444 = call @aten.mm.965(%443, %arg7) : (tensor<128x2xf32>, tensor<2x2816xf32>) -> tensor<128x2816xf32>
    %445 = call @aten.permute.970(%444) {xla_shape = "f32[2816,128]{0,1}"} : (tensor<128x2816xf32>) -> tensor<2816x128xf32>
    %446 = call @aten.permute.974(%445) : (tensor<2816x128xf32>) -> tensor<128x2816xf32>
    %447 = call @aten.sum.902(%2) : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %448 = call @aten.view.909(%447) : (tensor<1x256xf32>) -> tensor<256xf32>
    %449 = call @aten.permute.865(%2) {xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    %450 = call @aten.mm.991(%449, %arg6) : (tensor<256x2xf32>, tensor<2x2048xf32>) -> tensor<256x2048xf32>
    %451 = call @aten.permute.996(%450) {xla_shape = "f32[2048,256]{0,1}"} : (tensor<256x2048xf32>) -> tensor<2048x256xf32>
    %452 = call @aten.permute.1000(%451) : (tensor<2048x256xf32>) -> tensor<256x2048xf32>
    %453 = call @aten.permute.978(%arg485) {xla_shape = "f32[2,256]{0,1}"} : (tensor<256x2xf32>) -> tensor<2x256xf32>
    %454 = call @aten.mm.982(%arg786, %453) : (tensor<2x2xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    %455 = call @aten.threshold_backward.888(%454, %arg477) : (tensor<2x256xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    %456 = call @aten.sum.902(%455) : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %457 = call @aten.view.909(%456) : (tensor<1x256xf32>) -> tensor<256xf32>
    %458 = call @aten.permute.865(%455) {xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    %459 = call @aten.mm.1010(%458, %arg458) : (tensor<256x2xf32>, tensor<2x768xf32>) -> tensor<256x768xf32>
    %460 = call @aten.permute.1015(%459) {xla_shape = "f32[768,256]{0,1}"} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    %461 = call @aten.permute.1019(%460) : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %462 = "mhlo.get_tuple_element"(%406) {index = 2 : i32} : (tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %463 = "mhlo.get_tuple_element"(%406) {index = 1 : i32} : (tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %464 = "mhlo.get_tuple_element"(%408) {index = 1 : i32, xla_shape = "f32[64,3,7,7]{0,1,3,2}"} : (tuple<tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>>) -> tensor<64x3x7x7xf32>
    %465 = "mhlo.get_tuple_element"(%394) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %466 = "mhlo.get_tuple_element"(%394) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %467 = "mhlo.get_tuple_element"(%396) {index = 1 : i32, xla_shape = "f32[64,64,1,1]{0,1,3,2}"} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>>) -> tensor<64x64x1x1xf32>
    %468 = "mhlo.get_tuple_element"(%388) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %469 = "mhlo.get_tuple_element"(%388) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %470 = "mhlo.get_tuple_element"(%390) {index = 1 : i32, xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<64x64x3x3xf32>
    %471 = "mhlo.get_tuple_element"(%382) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %472 = "mhlo.get_tuple_element"(%382) {index = 1 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %473 = "mhlo.get_tuple_element"(%384) {index = 1 : i32, xla_shape = "f32[256,64,1,1]{0,1,3,2}"} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<256x64x1x1xf32>
    %474 = "mhlo.get_tuple_element"(%378) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %475 = "mhlo.get_tuple_element"(%378) {index = 1 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %476 = "mhlo.get_tuple_element"(%380) {index = 1 : i32, xla_shape = "f32[256,64,1,1]{0,1,3,2}"} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<256x64x1x1xf32>
    %477 = "mhlo.get_tuple_element"(%368) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %478 = "mhlo.get_tuple_element"(%368) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %479 = "mhlo.get_tuple_element"(%370) {index = 1 : i32, xla_shape = "f32[64,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>) -> tensor<64x256x1x1xf32>
    %480 = "mhlo.get_tuple_element"(%362) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %481 = "mhlo.get_tuple_element"(%362) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %482 = "mhlo.get_tuple_element"(%364) {index = 1 : i32, xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<64x64x3x3xf32>
    %483 = "mhlo.get_tuple_element"(%356) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %484 = "mhlo.get_tuple_element"(%356) {index = 1 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %485 = "mhlo.get_tuple_element"(%358) {index = 1 : i32, xla_shape = "f32[256,64,1,1]{0,1,3,2}"} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<256x64x1x1xf32>
    %486 = "mhlo.get_tuple_element"(%346) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %487 = "mhlo.get_tuple_element"(%346) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %488 = "mhlo.get_tuple_element"(%348) {index = 1 : i32, xla_shape = "f32[64,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>) -> tensor<64x256x1x1xf32>
    %489 = "mhlo.get_tuple_element"(%340) {index = 2 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %490 = "mhlo.get_tuple_element"(%340) {index = 1 : i32} : (tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %491 = "mhlo.get_tuple_element"(%342) {index = 1 : i32, xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>) -> tensor<64x64x3x3xf32>
    %492 = "mhlo.get_tuple_element"(%334) {index = 2 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %493 = "mhlo.get_tuple_element"(%334) {index = 1 : i32} : (tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %494 = "mhlo.get_tuple_element"(%336) {index = 1 : i32, xla_shape = "f32[256,64,1,1]{0,1,3,2}"} : (tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>) -> tensor<256x64x1x1xf32>
    %495 = "mhlo.get_tuple_element"(%323) {index = 2 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %496 = "mhlo.get_tuple_element"(%323) {index = 1 : i32} : (tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %497 = "mhlo.get_tuple_element"(%325) {index = 1 : i32, xla_shape = "f32[128,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>>) -> tensor<128x256x1x1xf32>
    %498 = "mhlo.get_tuple_element"(%317) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %499 = "mhlo.get_tuple_element"(%317) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %500 = "mhlo.get_tuple_element"(%319) {index = 1 : i32, xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tuple<tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<128x128x3x3xf32>
    %501 = "mhlo.get_tuple_element"(%311) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %502 = "mhlo.get_tuple_element"(%311) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %503 = "mhlo.get_tuple_element"(%313) {index = 1 : i32, xla_shape = "f32[512,128,1,1]{0,1,3,2}"} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<512x128x1x1xf32>
    %504 = "mhlo.get_tuple_element"(%307) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %505 = "mhlo.get_tuple_element"(%307) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %506 = "mhlo.get_tuple_element"(%309) {index = 1 : i32, xla_shape = "f32[512,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>>) -> tensor<512x256x1x1xf32>
    %507 = "mhlo.get_tuple_element"(%297) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %508 = "mhlo.get_tuple_element"(%297) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %509 = "mhlo.get_tuple_element"(%299) {index = 1 : i32, xla_shape = "f32[128,512,1,1]{0,1,3,2}"} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<128x512x1x1xf32>
    %510 = "mhlo.get_tuple_element"(%291) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %511 = "mhlo.get_tuple_element"(%291) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %512 = "mhlo.get_tuple_element"(%293) {index = 1 : i32, xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<128x128x3x3xf32>
    %513 = "mhlo.get_tuple_element"(%285) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %514 = "mhlo.get_tuple_element"(%285) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %515 = "mhlo.get_tuple_element"(%287) {index = 1 : i32, xla_shape = "f32[512,128,1,1]{0,1,3,2}"} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<512x128x1x1xf32>
    %516 = "mhlo.get_tuple_element"(%275) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %517 = "mhlo.get_tuple_element"(%275) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %518 = "mhlo.get_tuple_element"(%277) {index = 1 : i32, xla_shape = "f32[128,512,1,1]{0,1,3,2}"} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<128x512x1x1xf32>
    %519 = "mhlo.get_tuple_element"(%269) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %520 = "mhlo.get_tuple_element"(%269) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %521 = "mhlo.get_tuple_element"(%271) {index = 1 : i32, xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<128x128x3x3xf32>
    %522 = "mhlo.get_tuple_element"(%263) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %523 = "mhlo.get_tuple_element"(%263) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %524 = "mhlo.get_tuple_element"(%265) {index = 1 : i32, xla_shape = "f32[512,128,1,1]{0,1,3,2}"} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<512x128x1x1xf32>
    %525 = "mhlo.get_tuple_element"(%253) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %526 = "mhlo.get_tuple_element"(%253) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %527 = "mhlo.get_tuple_element"(%255) {index = 1 : i32, xla_shape = "f32[128,512,1,1]{0,1,3,2}"} : (tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>) -> tensor<128x512x1x1xf32>
    %528 = "mhlo.get_tuple_element"(%247) {index = 2 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %529 = "mhlo.get_tuple_element"(%247) {index = 1 : i32} : (tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %530 = "mhlo.get_tuple_element"(%249) {index = 1 : i32, xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>) -> tensor<128x128x3x3xf32>
    %531 = "mhlo.get_tuple_element"(%241) {index = 2 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %532 = "mhlo.get_tuple_element"(%241) {index = 1 : i32} : (tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %533 = "mhlo.get_tuple_element"(%243) {index = 1 : i32, xla_shape = "f32[512,128,1,1]{0,1,3,2}"} : (tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>) -> tensor<512x128x1x1xf32>
    %534 = "mhlo.get_tuple_element"(%230) {index = 2 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %535 = "mhlo.get_tuple_element"(%230) {index = 1 : i32} : (tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %536 = "mhlo.get_tuple_element"(%232) {index = 1 : i32, xla_shape = "f32[256,512,1,1]{0,1,3,2}"} : (tuple<tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>>) -> tensor<256x512x1x1xf32>
    %537 = "mhlo.get_tuple_element"(%224) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %538 = "mhlo.get_tuple_element"(%224) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %539 = "mhlo.get_tuple_element"(%226) {index = 1 : i32, xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256x256x3x3xf32>
    %540 = "mhlo.get_tuple_element"(%218) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %541 = "mhlo.get_tuple_element"(%218) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %542 = "mhlo.get_tuple_element"(%220) {index = 1 : i32, xla_shape = "f32[1024,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024x256x1x1xf32>
    %543 = "mhlo.get_tuple_element"(%214) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %544 = "mhlo.get_tuple_element"(%214) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %545 = "mhlo.get_tuple_element"(%216) {index = 1 : i32, xla_shape = "f32[1024,512,1,1]{0,1,3,2}"} : (tuple<tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>>) -> tensor<1024x512x1x1xf32>
    %546 = "mhlo.get_tuple_element"(%204) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %547 = "mhlo.get_tuple_element"(%204) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %548 = "mhlo.get_tuple_element"(%206) {index = 1 : i32, xla_shape = "f32[256,1024,1,1]{0,1,3,2}"} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256x1024x1x1xf32>
    %549 = "mhlo.get_tuple_element"(%198) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %550 = "mhlo.get_tuple_element"(%198) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %551 = "mhlo.get_tuple_element"(%200) {index = 1 : i32, xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256x256x3x3xf32>
    %552 = "mhlo.get_tuple_element"(%192) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %553 = "mhlo.get_tuple_element"(%192) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %554 = "mhlo.get_tuple_element"(%194) {index = 1 : i32, xla_shape = "f32[1024,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024x256x1x1xf32>
    %555 = "mhlo.get_tuple_element"(%182) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %556 = "mhlo.get_tuple_element"(%182) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %557 = "mhlo.get_tuple_element"(%184) {index = 1 : i32, xla_shape = "f32[256,1024,1,1]{0,1,3,2}"} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256x1024x1x1xf32>
    %558 = "mhlo.get_tuple_element"(%176) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %559 = "mhlo.get_tuple_element"(%176) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %560 = "mhlo.get_tuple_element"(%178) {index = 1 : i32, xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256x256x3x3xf32>
    %561 = "mhlo.get_tuple_element"(%170) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %562 = "mhlo.get_tuple_element"(%170) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %563 = "mhlo.get_tuple_element"(%172) {index = 1 : i32, xla_shape = "f32[1024,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024x256x1x1xf32>
    %564 = "mhlo.get_tuple_element"(%160) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %565 = "mhlo.get_tuple_element"(%160) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %566 = "mhlo.get_tuple_element"(%162) {index = 1 : i32, xla_shape = "f32[256,1024,1,1]{0,1,3,2}"} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256x1024x1x1xf32>
    %567 = "mhlo.get_tuple_element"(%154) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %568 = "mhlo.get_tuple_element"(%154) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %569 = "mhlo.get_tuple_element"(%156) {index = 1 : i32, xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256x256x3x3xf32>
    %570 = "mhlo.get_tuple_element"(%148) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %571 = "mhlo.get_tuple_element"(%148) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %572 = "mhlo.get_tuple_element"(%150) {index = 1 : i32, xla_shape = "f32[1024,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024x256x1x1xf32>
    %573 = "mhlo.get_tuple_element"(%138) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %574 = "mhlo.get_tuple_element"(%138) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %575 = "mhlo.get_tuple_element"(%140) {index = 1 : i32, xla_shape = "f32[256,1024,1,1]{0,1,3,2}"} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256x1024x1x1xf32>
    %576 = "mhlo.get_tuple_element"(%132) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %577 = "mhlo.get_tuple_element"(%132) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %578 = "mhlo.get_tuple_element"(%134) {index = 1 : i32, xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256x256x3x3xf32>
    %579 = "mhlo.get_tuple_element"(%126) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %580 = "mhlo.get_tuple_element"(%126) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %581 = "mhlo.get_tuple_element"(%128) {index = 1 : i32, xla_shape = "f32[1024,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024x256x1x1xf32>
    %582 = "mhlo.get_tuple_element"(%116) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %583 = "mhlo.get_tuple_element"(%116) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %584 = "mhlo.get_tuple_element"(%118) {index = 1 : i32, xla_shape = "f32[256,1024,1,1]{0,1,3,2}"} : (tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>) -> tensor<256x1024x1x1xf32>
    %585 = "mhlo.get_tuple_element"(%110) {index = 2 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %586 = "mhlo.get_tuple_element"(%110) {index = 1 : i32} : (tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %587 = "mhlo.get_tuple_element"(%112) {index = 1 : i32, xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>) -> tensor<256x256x3x3xf32>
    %588 = "mhlo.get_tuple_element"(%104) {index = 2 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %589 = "mhlo.get_tuple_element"(%104) {index = 1 : i32} : (tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>) -> tensor<1024xf32>
    %590 = "mhlo.get_tuple_element"(%106) {index = 1 : i32, xla_shape = "f32[1024,256,1,1]{0,1,3,2}"} : (tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>) -> tensor<1024x256x1x1xf32>
    %591 = "mhlo.get_tuple_element"(%93) {index = 2 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %592 = "mhlo.get_tuple_element"(%93) {index = 1 : i32} : (tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %593 = "mhlo.get_tuple_element"(%95) {index = 1 : i32, xla_shape = "f32[512,1024,1,1]{0,1,3,2}"} : (tuple<tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>>) -> tensor<512x1024x1x1xf32>
    %594 = "mhlo.get_tuple_element"(%87) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %595 = "mhlo.get_tuple_element"(%87) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %596 = "mhlo.get_tuple_element"(%89) {index = 1 : i32, xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tuple<tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<512x512x3x3xf32>
    %597 = "mhlo.get_tuple_element"(%81) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %598 = "mhlo.get_tuple_element"(%81) {index = 1 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %599 = "mhlo.get_tuple_element"(%83) {index = 1 : i32, xla_shape = "f32[2048,512,1,1]{0,1,3,2}"} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<2048x512x1x1xf32>
    %600 = "mhlo.get_tuple_element"(%77) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %601 = "mhlo.get_tuple_element"(%77) {index = 1 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %602 = "mhlo.get_tuple_element"(%79) {index = 1 : i32, xla_shape = "f32[2048,1024,1,1]{0,1,3,2}"} : (tuple<tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>>) -> tensor<2048x1024x1x1xf32>
    %603 = "mhlo.get_tuple_element"(%67) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %604 = "mhlo.get_tuple_element"(%67) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %605 = "mhlo.get_tuple_element"(%69) {index = 1 : i32, xla_shape = "f32[512,2048,1,1]{0,1,3,2}"} : (tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>) -> tensor<512x2048x1x1xf32>
    %606 = "mhlo.get_tuple_element"(%61) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %607 = "mhlo.get_tuple_element"(%61) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %608 = "mhlo.get_tuple_element"(%63) {index = 1 : i32, xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<512x512x3x3xf32>
    %609 = "mhlo.get_tuple_element"(%55) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %610 = "mhlo.get_tuple_element"(%55) {index = 1 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %611 = "mhlo.get_tuple_element"(%57) {index = 1 : i32, xla_shape = "f32[2048,512,1,1]{0,1,3,2}"} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<2048x512x1x1xf32>
    %612 = "mhlo.get_tuple_element"(%45) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %613 = "mhlo.get_tuple_element"(%45) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %614 = "mhlo.get_tuple_element"(%47) {index = 1 : i32, xla_shape = "f32[512,2048,1,1]{0,1,3,2}"} : (tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>) -> tensor<512x2048x1x1xf32>
    %615 = "mhlo.get_tuple_element"(%39) {index = 2 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %616 = "mhlo.get_tuple_element"(%39) {index = 1 : i32} : (tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %617 = "mhlo.get_tuple_element"(%41) {index = 1 : i32, xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>) -> tensor<512x512x3x3xf32>
    %618 = "mhlo.get_tuple_element"(%33) {index = 2 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %619 = "mhlo.get_tuple_element"(%33) {index = 1 : i32} : (tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>) -> tensor<2048xf32>
    %620 = "mhlo.get_tuple_element"(%35) {index = 1 : i32, xla_shape = "f32[2048,512,1,1]{0,1,3,2}"} : (tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>) -> tensor<2048x512x1x1xf32>
    %621 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %622 = "mhlo.broadcast_in_dim"(%621) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %623 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %624 = "mhlo.broadcast_in_dim"(%623) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %625 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %626 = call @aten.expand.2522(%625) : (tensor<f32>) -> tensor<512x768xf32>
    %627 = call @aten.view.2463(%arg704) : (tensor<1x240xi64>) -> tensor<240xi64>
    %628 = mhlo.constant dense<0> : tensor<i64>
    %629 = call @aten.lt.2504(%627, %628) : (tensor<240xi64>, tensor<i64>) -> tensor<240xi1>
    %630 = mhlo.constant dense<512> : tensor<i64>
    %631 = call @aten.expand.2491(%630) : (tensor<i64>) -> tensor<240xi64>
    %632 = call @aten.add.2498(%627, %631) : (tensor<240xi64>, tensor<240xi64>) -> tensor<240xi64>
    %633 = call @aten.where.2510(%629, %632, %627) : (tensor<240xi1>, tensor<240xi64>, tensor<240xi64>) -> tensor<240xi64>
    %634 = call @aten.stack.2516(%633) : (tensor<240xi64>) -> tensor<240x1xi64>
    %635 = mhlo.constant dense<-1.000000e+00> : tensor<f64>
    %636 = call @aten.ne.2467(%627, %635) : (tensor<240xi64>, tensor<f64>) -> tensor<240xi1>
    %637 = call @aten.view.2474(%636) : (tensor<240xi1>) -> tensor<240x1xi1>
    %638 = call @aten.expand.2478(%637) : (tensor<240x1xi1>) -> tensor<240x768xi1>
    %639 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %640 = "mhlo.broadcast_in_dim"(%639) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %641 = call @aten.sum.2451(%640) : (tensor<2x240x768xf32>) -> tensor<1x240x768xf32>
    %642 = call @aten.view.2458(%641) : (tensor<1x240x768xf32>) -> tensor<240x768xf32>
    %643 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %644 = call @aten.expand.2438(%643) : (tensor<f32>) -> tensor<240x768xf32>
    %645 = call @aten.where.2484(%638, %642, %644) : (tensor<240x768xi1>, tensor<240x768xf32>, tensor<240x768xf32>) -> tensor<240x768xf32>
    %646 = call @aten.index_put.2533(%626, %634, %645) : (tensor<512x768xf32>, tensor<240x1xi64>, tensor<240x768xf32>) -> tensor<512x768xf32>
    %647 = call @aten.permute.2540(%646) : (tensor<512x768xf32>) -> tensor<512x768xf32>
    %648 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %649 = call @aten.expand.2616(%648) : (tensor<f32>) -> tensor<2x768xf32>
    %650 = call @aten.view.2557(%arg94) : (tensor<2x240xi64>) -> tensor<480xi64>
    %651 = mhlo.constant dense<0> : tensor<i64>
    %652 = call @aten.lt.2598(%650, %651) : (tensor<480xi64>, tensor<i64>) -> tensor<480xi1>
    %653 = mhlo.constant dense<2> : tensor<i64>
    %654 = call @aten.expand.2585(%653) : (tensor<i64>) -> tensor<480xi64>
    %655 = call @aten.add.2592(%650, %654) : (tensor<480xi64>, tensor<480xi64>) -> tensor<480xi64>
    %656 = call @aten.where.2604(%652, %655, %650) : (tensor<480xi1>, tensor<480xi64>, tensor<480xi64>) -> tensor<480xi64>
    %657 = call @aten.stack.2610(%656) : (tensor<480xi64>) -> tensor<480x1xi64>
    %658 = mhlo.constant dense<-1.000000e+00> : tensor<f64>
    %659 = call @aten.ne.2561(%650, %658) : (tensor<480xi64>, tensor<f64>) -> tensor<480xi1>
    %660 = call @aten.view.2568(%659) : (tensor<480xi1>) -> tensor<480x1xi1>
    %661 = call @aten.expand.2572(%660) : (tensor<480x1xi1>) -> tensor<480x768xi1>
    %662 = call @aten.view.2552(%640) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %663 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %664 = call @aten.expand.2545(%663) : (tensor<f32>) -> tensor<480x768xf32>
    %665 = call @aten.where.2578(%661, %662, %664) : (tensor<480x768xi1>, tensor<480x768xf32>, tensor<480x768xf32>) -> tensor<480x768xf32>
    %666 = call @aten.index_put.2627(%649, %657, %665) : (tensor<2x768xf32>, tensor<480x1xi64>, tensor<480x768xf32>) -> tensor<2x768xf32>
    %667 = call @aten.permute.2634(%666) : (tensor<2x768xf32>) -> tensor<2x768xf32>
    %668 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %669 = call @aten.expand.2655(%668) : (tensor<f32>) -> tensor<21128x768xf32>
    %670 = call @aten.view.2557(%arg627) : (tensor<2x240xi64>) -> tensor<480xi64>
    %671 = mhlo.constant dense<0> : tensor<i64>
    %672 = call @aten.lt.2598(%670, %671) : (tensor<480xi64>, tensor<i64>) -> tensor<480xi1>
    %673 = mhlo.constant dense<21128> : tensor<i64>
    %674 = call @aten.expand.2585(%673) : (tensor<i64>) -> tensor<480xi64>
    %675 = call @aten.add.2592(%670, %674) : (tensor<480xi64>, tensor<480xi64>) -> tensor<480xi64>
    %676 = call @aten.where.2604(%672, %675, %670) : (tensor<480xi1>, tensor<480xi64>, tensor<480xi64>) -> tensor<480xi64>
    %677 = call @aten.stack.2610(%676) : (tensor<480xi64>) -> tensor<480x1xi64>
    %678 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %679 = call @aten.ne.2561(%670, %678) : (tensor<480xi64>, tensor<f64>) -> tensor<480xi1>
    %680 = call @aten.view.2568(%679) : (tensor<480xi1>) -> tensor<480x1xi1>
    %681 = call @aten.expand.2572(%680) : (tensor<480x1xi1>) -> tensor<480x768xi1>
    %682 = call @aten.view.2552(%640) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %683 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %684 = call @aten.expand.2545(%683) : (tensor<f32>) -> tensor<480x768xf32>
    %685 = call @aten.where.2578(%681, %682, %684) : (tensor<480x768xi1>, tensor<480x768xf32>, tensor<480x768xf32>) -> tensor<480x768xf32>
    %686 = call @aten.index_put.2666(%669, %677, %685) : (tensor<21128x768xf32>, tensor<480x1xi64>, tensor<480x768xf32>) -> tensor<21128x768xf32>
    %687 = call @aten.permute.2673(%686) : (tensor<21128x768xf32>) -> tensor<21128x768xf32>
    %688 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %689 = "mhlo.broadcast_in_dim"(%688) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %690 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %691 = "mhlo.broadcast_in_dim"(%690) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %692 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %693 = "mhlo.broadcast_in_dim"(%692) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %694 = call @aten.view.2552(%693) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %695 = call @aten.sum.2688(%694) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %696 = call @aten.view.2695(%695) : (tensor<1x768xf32>) -> tensor<768xf32>
    %697 = call @aten.view.2552(%693) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %698 = call @aten.permute.2700(%697) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %699 = call @aten.mm.2704(%698, %arg438) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %700 = call @aten.permute.2709(%699) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %701 = call @aten.permute.2713(%700) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %702 = call @aten.permute.2718(%arg230) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %703 = call @aten.permute.2709(%arg265) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %704 = call @aten.mm.2723(%694, %703) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %705 = call @aten.view.2728(%704) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %706 = call @aten.view.2732(%705) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %707 = call @aten.permute.2736(%706) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %708 = call @aten.view.2740(%707) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %709 = call @aten.permute.2718(%arg238) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %710 = call @aten.matmul.2744(%708, %709) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %711 = call @aten.view.2749(%710) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %712 = call @aten._softmax_backward_data.2757(%711, %arg228) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %713 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %714 = call @aten.div.2767(%712, %713) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %715 = call @aten.view.2773(%714) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %716 = call @aten.matmul.2778(%702, %715) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %717 = call @aten.view.2783(%716) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %718 = call @aten.permute.2787(%717) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %719 = call @aten.permute.2791(%718) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %720 = call @aten.view.2795(%719) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %721 = call @aten.view.2552(%720) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %722 = call @aten.sum.2688(%721) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %723 = call @aten.view.2695(%722) : (tensor<1x768xf32>) -> tensor<768xf32>
    %724 = call @aten.view.2552(%720) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %725 = call @aten.permute.2700(%724) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %726 = call @aten.mm.2704(%725, %arg282) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %727 = call @aten.permute.2709(%726) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %728 = call @aten.permute.2713(%727) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %729 = call @aten.permute.2807(%arg236) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %730 = call @aten.matmul.2811(%715, %729) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %731 = call @aten.view.2816(%730) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %732 = call @aten.permute.2820(%731) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %733 = call @aten.view.2824(%732) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %734 = call @aten.view.2552(%733) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %735 = call @aten.sum.2688(%734) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %736 = call @aten.view.2695(%735) : (tensor<1x768xf32>) -> tensor<768xf32>
    %737 = call @aten.view.2824(%732) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %738 = call @aten.view.2552(%737) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %739 = call @aten.permute.2700(%738) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %740 = call @aten.mm.2704(%739, %arg280) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %741 = call @aten.permute.2709(%740) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %742 = call @aten.permute.2713(%741) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %743 = call @aten.permute.2837(%arg421) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %744 = call @aten.matmul.2841(%743, %708) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %745 = call @aten.view.2816(%744) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %746 = call @aten.permute.2820(%745) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %747 = call @aten.view.2824(%746) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %748 = call @aten.view.2552(%747) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %749 = call @aten.sum.2688(%748) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %750 = call @aten.view.2695(%749) : (tensor<1x768xf32>) -> tensor<768xf32>
    %751 = call @aten.view.2824(%746) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %752 = call @aten.view.2552(%751) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %753 = call @aten.permute.2700(%752) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %754 = call @aten.mm.2704(%753, %arg296) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %755 = call @aten.permute.2709(%754) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %756 = call @aten.permute.2713(%755) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %757 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %758 = "mhlo.broadcast_in_dim"(%757) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %759 = call @aten.view.2552(%758) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %760 = call @aten.permute.2858(%arg713) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %761 = call @aten.mm.2865(%759, %760) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %762 = call @aten.view.2870(%761) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %763 = call @aten.gelu_backward.2874(%762, %arg462) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %764 = call @aten.view.2960(%763) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %765 = call @aten.sum.2968(%764) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %766 = call @aten.view.2975(%765) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %767 = call @aten.view.2960(%763) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %768 = call @aten.permute.2980(%767) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %769 = call @aten.mm.2984(%768, %arg455) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %770 = call @aten.permute.2858(%769) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %771 = call @aten.permute.2990(%770) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %772 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %773 = "mhlo.broadcast_in_dim"(%772) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %774 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %775 = "mhlo.broadcast_in_dim"(%774) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %776 = call @aten.sum.2688(%759) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %777 = call @aten.view.2695(%776) : (tensor<1x768xf32>) -> tensor<768xf32>
    %778 = call @aten.view.2552(%758) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %779 = call @aten.permute.2700(%778) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %780 = call @aten.mm.3002(%779, %arg717) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %781 = call @aten.permute.3007(%780) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %782 = call @aten.permute.3011(%781) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %783 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %784 = "mhlo.broadcast_in_dim"(%783) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %785 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %786 = "mhlo.broadcast_in_dim"(%785) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %787 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %788 = "mhlo.broadcast_in_dim"(%787) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %789 = call @aten.view.2552(%788) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %790 = call @aten.sum.2688(%789) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %791 = call @aten.view.2695(%790) : (tensor<1x768xf32>) -> tensor<768xf32>
    %792 = call @aten.view.2552(%788) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %793 = call @aten.permute.2700(%792) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %794 = call @aten.mm.2704(%793, %arg652) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %795 = call @aten.permute.2709(%794) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %796 = call @aten.permute.2713(%795) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %797 = call @aten.permute.2718(%arg706) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %798 = call @aten.permute.2709(%arg758) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %799 = call @aten.mm.2723(%789, %798) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %800 = call @aten.view.2728(%799) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %801 = call @aten.view.2732(%800) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %802 = call @aten.permute.2736(%801) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %803 = call @aten.view.2740(%802) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %804 = call @aten.permute.2718(%arg264) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %805 = call @aten.matmul.2744(%803, %804) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %806 = call @aten.view.2749(%805) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %807 = call @aten._softmax_backward_data.2757(%806, %arg240) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %808 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %809 = call @aten.div.2767(%807, %808) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %810 = call @aten.view.2773(%809) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %811 = call @aten.matmul.2778(%797, %810) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %812 = call @aten.view.2783(%811) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %813 = call @aten.permute.2787(%812) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %814 = call @aten.permute.2791(%813) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %815 = call @aten.view.2795(%814) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %816 = call @aten.view.2552(%815) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %817 = call @aten.sum.2688(%816) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %818 = call @aten.view.2695(%817) : (tensor<1x768xf32>) -> tensor<768xf32>
    %819 = call @aten.view.2552(%815) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %820 = call @aten.permute.2700(%819) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %821 = call @aten.mm.2704(%820, %arg737) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %822 = call @aten.permute.2709(%821) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %823 = call @aten.permute.2713(%822) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %824 = call @aten.permute.2807(%arg261) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %825 = call @aten.matmul.2811(%810, %824) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %826 = call @aten.view.2816(%825) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %827 = call @aten.permute.2820(%826) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %828 = call @aten.view.2824(%827) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %829 = call @aten.view.2552(%828) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %830 = call @aten.sum.2688(%829) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %831 = call @aten.view.2695(%830) : (tensor<1x768xf32>) -> tensor<768xf32>
    %832 = call @aten.view.2824(%827) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %833 = call @aten.view.2552(%832) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %834 = call @aten.permute.2700(%833) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %835 = call @aten.mm.2704(%834, %arg733) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %836 = call @aten.permute.2709(%835) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %837 = call @aten.permute.2713(%836) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %838 = call @aten.permute.2837(%arg263) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %839 = call @aten.matmul.2841(%838, %803) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %840 = call @aten.view.2816(%839) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %841 = call @aten.permute.2820(%840) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %842 = call @aten.view.2824(%841) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %843 = call @aten.view.2552(%842) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %844 = call @aten.sum.2688(%843) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %845 = call @aten.view.2695(%844) : (tensor<1x768xf32>) -> tensor<768xf32>
    %846 = call @aten.view.2824(%841) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %847 = call @aten.view.2552(%846) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %848 = call @aten.permute.2700(%847) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %849 = call @aten.mm.2704(%848, %arg762) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %850 = call @aten.permute.2709(%849) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %851 = call @aten.permute.2713(%850) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %852 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %853 = "mhlo.broadcast_in_dim"(%852) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %854 = call @aten.view.2552(%853) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %855 = call @aten.permute.2858(%arg674) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %856 = call @aten.mm.2865(%854, %855) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %857 = call @aten.view.2870(%856) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %858 = call @aten.gelu_backward.2874(%857, %arg669) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %859 = call @aten.view.2960(%858) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %860 = call @aten.sum.2968(%859) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %861 = call @aten.view.2975(%860) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %862 = call @aten.view.2960(%858) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %863 = call @aten.permute.2980(%862) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %864 = call @aten.mm.2984(%863, %arg667) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %865 = call @aten.permute.2858(%864) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %866 = call @aten.permute.2990(%865) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %867 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %868 = "mhlo.broadcast_in_dim"(%867) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %869 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %870 = "mhlo.broadcast_in_dim"(%869) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %871 = call @aten.sum.2688(%854) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %872 = call @aten.view.2695(%871) : (tensor<1x768xf32>) -> tensor<768xf32>
    %873 = call @aten.view.2552(%853) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %874 = call @aten.permute.2700(%873) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %875 = call @aten.mm.3002(%874, %arg677) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %876 = call @aten.permute.3007(%875) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %877 = call @aten.permute.3011(%876) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %878 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %879 = "mhlo.broadcast_in_dim"(%878) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %880 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %881 = "mhlo.broadcast_in_dim"(%880) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %882 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %883 = "mhlo.broadcast_in_dim"(%882) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %884 = call @aten.view.2552(%883) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %885 = call @aten.sum.2688(%884) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %886 = call @aten.view.2695(%885) : (tensor<1x768xf32>) -> tensor<768xf32>
    %887 = call @aten.view.2552(%883) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %888 = call @aten.permute.2700(%887) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %889 = call @aten.mm.2704(%888, %arg20) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %890 = call @aten.permute.2709(%889) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %891 = call @aten.permute.2713(%890) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %892 = call @aten.permute.2718(%arg126) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %893 = call @aten.permute.2709(%arg478) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %894 = call @aten.mm.2723(%884, %893) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %895 = call @aten.view.2728(%894) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %896 = call @aten.view.2732(%895) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %897 = call @aten.permute.2736(%896) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %898 = call @aten.view.2740(%897) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %899 = call @aten.permute.2718(%arg499) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %900 = call @aten.matmul.2744(%898, %899) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %901 = call @aten.view.2749(%900) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %902 = call @aten._softmax_backward_data.2757(%901, %arg577) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %903 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %904 = call @aten.div.2767(%902, %903) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %905 = call @aten.view.2773(%904) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %906 = call @aten.matmul.2778(%892, %905) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %907 = call @aten.view.2783(%906) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %908 = call @aten.permute.2787(%907) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %909 = call @aten.permute.2791(%908) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %910 = call @aten.view.2795(%909) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %911 = call @aten.view.2552(%910) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %912 = call @aten.sum.2688(%911) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %913 = call @aten.view.2695(%912) : (tensor<1x768xf32>) -> tensor<768xf32>
    %914 = call @aten.view.2552(%910) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %915 = call @aten.permute.2700(%914) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %916 = call @aten.mm.2704(%915, %arg467) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %917 = call @aten.permute.2709(%916) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %918 = call @aten.permute.2713(%917) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %919 = call @aten.permute.2807(%arg493) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %920 = call @aten.matmul.2811(%905, %919) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %921 = call @aten.view.2816(%920) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %922 = call @aten.permute.2820(%921) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %923 = call @aten.view.2824(%922) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %924 = call @aten.view.2552(%923) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %925 = call @aten.sum.2688(%924) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %926 = call @aten.view.2695(%925) : (tensor<1x768xf32>) -> tensor<768xf32>
    %927 = call @aten.view.2824(%922) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %928 = call @aten.view.2552(%927) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %929 = call @aten.permute.2700(%928) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %930 = call @aten.mm.2704(%929, %arg66) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %931 = call @aten.permute.2709(%930) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %932 = call @aten.permute.2713(%931) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %933 = call @aten.permute.2837(%arg17) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %934 = call @aten.matmul.2841(%933, %898) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %935 = call @aten.view.2816(%934) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %936 = call @aten.permute.2820(%935) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %937 = call @aten.view.2824(%936) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %938 = call @aten.view.2552(%937) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %939 = call @aten.sum.2688(%938) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %940 = call @aten.view.2695(%939) : (tensor<1x768xf32>) -> tensor<768xf32>
    %941 = call @aten.view.2824(%936) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %942 = call @aten.view.2552(%941) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %943 = call @aten.permute.2700(%942) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %944 = call @aten.mm.2704(%943, %arg482) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %945 = call @aten.permute.2709(%944) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %946 = call @aten.permute.2713(%945) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %947 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %948 = "mhlo.broadcast_in_dim"(%947) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %949 = call @aten.view.2552(%948) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %950 = call @aten.permute.2858(%arg33) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %951 = call @aten.mm.2865(%949, %950) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %952 = call @aten.view.2870(%951) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %953 = call @aten.gelu_backward.2874(%952, %arg28) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %954 = call @aten.view.2960(%953) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %955 = call @aten.sum.2968(%954) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %956 = call @aten.view.2975(%955) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %957 = call @aten.view.2960(%953) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %958 = call @aten.permute.2980(%957) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %959 = call @aten.mm.2984(%958, %arg29) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %960 = call @aten.permute.2858(%959) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %961 = call @aten.permute.2990(%960) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %962 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %963 = "mhlo.broadcast_in_dim"(%962) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %964 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %965 = "mhlo.broadcast_in_dim"(%964) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %966 = call @aten.sum.2688(%949) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %967 = call @aten.view.2695(%966) : (tensor<1x768xf32>) -> tensor<768xf32>
    %968 = call @aten.view.2552(%948) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %969 = call @aten.permute.2700(%968) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %970 = call @aten.mm.3002(%969, %arg34) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %971 = call @aten.permute.3007(%970) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %972 = call @aten.permute.3011(%971) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %973 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %974 = "mhlo.broadcast_in_dim"(%973) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %975 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %976 = "mhlo.broadcast_in_dim"(%975) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %977 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %978 = "mhlo.broadcast_in_dim"(%977) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %979 = call @aten.view.2552(%978) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %980 = call @aten.sum.2688(%979) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %981 = call @aten.view.2695(%980) : (tensor<1x768xf32>) -> tensor<768xf32>
    %982 = call @aten.view.2552(%978) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %983 = call @aten.permute.2700(%982) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %984 = call @aten.mm.2704(%983, %arg4) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %985 = call @aten.permute.2709(%984) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %986 = call @aten.permute.2713(%985) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %987 = call @aten.permute.2718(%arg502) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %988 = call @aten.permute.2709(%arg483) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %989 = call @aten.mm.2723(%979, %988) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %990 = call @aten.view.2728(%989) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %991 = call @aten.view.2732(%990) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %992 = call @aten.permute.2736(%991) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %993 = call @aten.view.2740(%992) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %994 = call @aten.permute.2718(%arg503) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %995 = call @aten.matmul.2744(%993, %994) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %996 = call @aten.view.2749(%995) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %997 = call @aten._softmax_backward_data.2757(%996, %arg131) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %998 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %999 = call @aten.div.2767(%997, %998) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1000 = call @aten.view.2773(%999) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1001 = call @aten.matmul.2778(%987, %1000) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1002 = call @aten.view.2783(%1001) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1003 = call @aten.permute.2787(%1002) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1004 = call @aten.permute.2791(%1003) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1005 = call @aten.view.2795(%1004) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1006 = call @aten.view.2552(%1005) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1007 = call @aten.sum.2688(%1006) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1008 = call @aten.view.2695(%1007) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1009 = call @aten.view.2552(%1005) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1010 = call @aten.permute.2700(%1009) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1011 = call @aten.mm.2704(%1010, %arg479) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1012 = call @aten.permute.2709(%1011) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1013 = call @aten.permute.2713(%1012) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1014 = call @aten.permute.2807(%arg501) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1015 = call @aten.matmul.2811(%1000, %1014) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1016 = call @aten.view.2816(%1015) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1017 = call @aten.permute.2820(%1016) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1018 = call @aten.view.2824(%1017) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1019 = call @aten.view.2552(%1018) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1020 = call @aten.sum.2688(%1019) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1021 = call @aten.view.2695(%1020) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1022 = call @aten.view.2824(%1017) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1023 = call @aten.view.2552(%1022) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1024 = call @aten.permute.2700(%1023) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1025 = call @aten.mm.2704(%1024, %arg476) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1026 = call @aten.permute.2709(%1025) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1027 = call @aten.permute.2713(%1026) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1028 = call @aten.permute.2837(%arg1) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1029 = call @aten.matmul.2841(%1028, %993) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1030 = call @aten.view.2816(%1029) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1031 = call @aten.permute.2820(%1030) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1032 = call @aten.view.2824(%1031) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1033 = call @aten.view.2552(%1032) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1034 = call @aten.sum.2688(%1033) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1035 = call @aten.view.2695(%1034) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1036 = call @aten.view.2824(%1031) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1037 = call @aten.view.2552(%1036) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1038 = call @aten.permute.2700(%1037) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1039 = call @aten.mm.2704(%1038, %arg495) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1040 = call @aten.permute.2709(%1039) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1041 = call @aten.permute.2713(%1040) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1042 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %1043 = "mhlo.broadcast_in_dim"(%1042) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1044 = call @aten.view.2552(%1043) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1045 = call @aten.permute.2858(%arg424) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1046 = call @aten.mm.2865(%1044, %1045) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1047 = call @aten.view.2870(%1046) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1048 = call @aten.gelu_backward.2874(%1047, %arg418) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1049 = call @aten.view.2960(%1048) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1050 = call @aten.sum.2968(%1049) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1051 = call @aten.view.2975(%1050) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1052 = call @aten.view.2960(%1048) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1053 = call @aten.permute.2980(%1052) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1054 = call @aten.mm.2984(%1053, %arg13) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1055 = call @aten.permute.2858(%1054) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1056 = call @aten.permute.2990(%1055) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1057 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %1058 = "mhlo.broadcast_in_dim"(%1057) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1059 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %1060 = "mhlo.broadcast_in_dim"(%1059) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1061 = call @aten.sum.2688(%1044) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1062 = call @aten.view.2695(%1061) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1063 = call @aten.view.2552(%1043) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1064 = call @aten.permute.2700(%1063) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1065 = call @aten.mm.3002(%1064, %arg427) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1066 = call @aten.permute.3007(%1065) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1067 = call @aten.permute.3011(%1066) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1068 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %1069 = "mhlo.broadcast_in_dim"(%1068) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1070 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %1071 = "mhlo.broadcast_in_dim"(%1070) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1072 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %1073 = "mhlo.broadcast_in_dim"(%1072) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1074 = call @aten.view.2552(%1073) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1075 = call @aten.sum.2688(%1074) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1076 = call @aten.view.2695(%1075) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1077 = call @aten.view.2552(%1073) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1078 = call @aten.permute.2700(%1077) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1079 = call @aten.mm.2704(%1078, %arg591) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1080 = call @aten.permute.2709(%1079) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1081 = call @aten.permute.2713(%1080) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1082 = call @aten.permute.2718(%arg648) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1083 = call @aten.permute.2709(%arg201) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1084 = call @aten.mm.2723(%1074, %1083) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1085 = call @aten.view.2728(%1084) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1086 = call @aten.view.2732(%1085) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1087 = call @aten.permute.2736(%1086) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1088 = call @aten.view.2740(%1087) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1089 = call @aten.permute.2718(%arg581) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1090 = call @aten.matmul.2744(%1088, %1089) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1091 = call @aten.view.2749(%1090) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1092 = call @aten._softmax_backward_data.2757(%1091, %arg233) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1093 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1094 = call @aten.div.2767(%1092, %1093) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1095 = call @aten.view.2773(%1094) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1096 = call @aten.matmul.2778(%1082, %1095) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1097 = call @aten.view.2783(%1096) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1098 = call @aten.permute.2787(%1097) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1099 = call @aten.permute.2791(%1098) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1100 = call @aten.view.2795(%1099) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1101 = call @aten.view.2552(%1100) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1102 = call @aten.sum.2688(%1101) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1103 = call @aten.view.2695(%1102) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1104 = call @aten.view.2552(%1100) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1105 = call @aten.permute.2700(%1104) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1106 = call @aten.mm.2704(%1105, %arg210) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1107 = call @aten.permute.2709(%1106) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1108 = call @aten.permute.2713(%1107) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1109 = call @aten.permute.2807(%arg226) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1110 = call @aten.matmul.2811(%1095, %1109) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1111 = call @aten.view.2816(%1110) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1112 = call @aten.permute.2820(%1111) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1113 = call @aten.view.2824(%1112) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1114 = call @aten.view.2552(%1113) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1115 = call @aten.sum.2688(%1114) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1116 = call @aten.view.2695(%1115) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1117 = call @aten.view.2824(%1112) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1118 = call @aten.view.2552(%1117) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1119 = call @aten.permute.2700(%1118) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1120 = call @aten.mm.2704(%1119, %arg203) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1121 = call @aten.permute.2709(%1120) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1122 = call @aten.permute.2713(%1121) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1123 = call @aten.permute.2837(%arg583) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1124 = call @aten.matmul.2841(%1123, %1088) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1125 = call @aten.view.2816(%1124) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1126 = call @aten.permute.2820(%1125) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1127 = call @aten.view.2824(%1126) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1128 = call @aten.view.2552(%1127) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1129 = call @aten.sum.2688(%1128) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1130 = call @aten.view.2695(%1129) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1131 = call @aten.view.2824(%1126) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1132 = call @aten.view.2552(%1131) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1133 = call @aten.permute.2700(%1132) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1134 = call @aten.mm.2704(%1133, %arg218) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1135 = call @aten.permute.2709(%1134) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1136 = call @aten.permute.2713(%1135) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1137 = mhlo.constant dense<0xFFFE1610> : tensor<f32>
    %1138 = "mhlo.broadcast_in_dim"(%1137) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1139 = call @aten.view.2552(%1138) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1140 = call @aten.permute.2858(%arg77) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1141 = call @aten.mm.2865(%1139, %1140) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1142 = call @aten.view.2870(%1141) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1143 = call @aten.gelu_backward.2874(%1142, %arg74) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1144 = call @aten.view.2960(%1143) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1145 = call @aten.sum.2968(%1144) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1146 = call @aten.view.2975(%1145) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1147 = call @aten.view.2960(%1143) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1148 = call @aten.permute.2980(%1147) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1149 = call @aten.mm.2984(%1148, %arg72) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1150 = call @aten.permute.2858(%1149) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1151 = call @aten.permute.2990(%1150) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1152 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1153 = "mhlo.broadcast_in_dim"(%1152) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1154 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1155 = "mhlo.broadcast_in_dim"(%1154) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1156 = call @aten.sum.2688(%1139) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1157 = call @aten.view.2695(%1156) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1158 = call @aten.view.2552(%1138) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1159 = call @aten.permute.2700(%1158) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1160 = call @aten.mm.3002(%1159, %arg78) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1161 = call @aten.permute.3007(%1160) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1162 = call @aten.permute.3011(%1161) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1163 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1164 = "mhlo.broadcast_in_dim"(%1163) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1165 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1166 = "mhlo.broadcast_in_dim"(%1165) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1167 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1168 = "mhlo.broadcast_in_dim"(%1167) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1169 = call @aten.view.2552(%1168) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1170 = call @aten.sum.2688(%1169) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1171 = call @aten.view.2695(%1170) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1172 = call @aten.view.2552(%1168) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1173 = call @aten.permute.2700(%1172) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1174 = call @aten.mm.2704(%1173, %arg730) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1175 = call @aten.permute.2709(%1174) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1176 = call @aten.permute.2713(%1175) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1177 = call @aten.permute.2718(%arg680) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1178 = call @aten.permute.2709(%arg88) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1179 = call @aten.mm.2723(%1169, %1178) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1180 = call @aten.view.2728(%1179) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1181 = call @aten.view.2732(%1180) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1182 = call @aten.permute.2736(%1181) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1183 = call @aten.view.2740(%1182) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1184 = call @aten.permute.2718(%arg584) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1185 = call @aten.matmul.2744(%1183, %1184) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1186 = call @aten.view.2749(%1185) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1187 = call @aten._softmax_backward_data.2757(%1186, %arg405) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1188 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1189 = call @aten.div.2767(%1187, %1188) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1190 = call @aten.view.2773(%1189) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1191 = call @aten.matmul.2778(%1177, %1190) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1192 = call @aten.view.2783(%1191) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1193 = call @aten.permute.2787(%1192) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1194 = call @aten.permute.2791(%1193) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1195 = call @aten.view.2795(%1194) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1196 = call @aten.view.2552(%1195) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1197 = call @aten.sum.2688(%1196) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1198 = call @aten.view.2695(%1197) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1199 = call @aten.view.2552(%1195) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1200 = call @aten.permute.2700(%1199) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1201 = call @aten.mm.2704(%1200, %arg92) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1202 = call @aten.permute.2709(%1201) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1203 = call @aten.permute.2713(%1202) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1204 = call @aten.permute.2807(%arg392) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1205 = call @aten.matmul.2811(%1190, %1204) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1206 = call @aten.view.2816(%1205) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1207 = call @aten.permute.2820(%1206) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1208 = call @aten.view.2824(%1207) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1209 = call @aten.view.2552(%1208) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1210 = call @aten.sum.2688(%1209) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1211 = call @aten.view.2695(%1210) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1212 = call @aten.view.2824(%1207) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1213 = call @aten.view.2552(%1212) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1214 = call @aten.permute.2700(%1213) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1215 = call @aten.mm.2704(%1214, %arg86) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1216 = call @aten.permute.2709(%1215) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1217 = call @aten.permute.2713(%1216) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1218 = call @aten.permute.2837(%arg714) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1219 = call @aten.matmul.2841(%1218, %1183) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1220 = call @aten.view.2816(%1219) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1221 = call @aten.permute.2820(%1220) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1222 = call @aten.view.2824(%1221) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1223 = call @aten.view.2552(%1222) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1224 = call @aten.sum.2688(%1223) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1225 = call @aten.view.2695(%1224) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1226 = call @aten.view.2824(%1221) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1227 = call @aten.view.2552(%1226) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1228 = call @aten.permute.2700(%1227) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1229 = call @aten.mm.2704(%1228, %arg370) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1230 = call @aten.permute.2709(%1229) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1231 = call @aten.permute.2713(%1230) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1232 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1233 = "mhlo.broadcast_in_dim"(%1232) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1234 = call @aten.view.2552(%1233) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1235 = call @aten.permute.2858(%arg763) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1236 = call @aten.mm.2865(%1234, %1235) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1237 = call @aten.view.2870(%1236) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1238 = call @aten.gelu_backward.2874(%1237, %arg751) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1239 = call @aten.view.2960(%1238) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1240 = call @aten.sum.2968(%1239) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1241 = call @aten.view.2975(%1240) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1242 = call @aten.view.2960(%1238) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1243 = call @aten.permute.2980(%1242) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1244 = call @aten.mm.2984(%1243, %arg746) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1245 = call @aten.permute.2858(%1244) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1246 = call @aten.permute.2990(%1245) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1247 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1248 = "mhlo.broadcast_in_dim"(%1247) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1249 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1250 = "mhlo.broadcast_in_dim"(%1249) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1251 = call @aten.sum.2688(%1234) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1252 = call @aten.view.2695(%1251) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1253 = call @aten.view.2552(%1233) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1254 = call @aten.permute.2700(%1253) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1255 = call @aten.mm.3002(%1254, %arg326) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1256 = call @aten.permute.3007(%1255) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1257 = call @aten.permute.3011(%1256) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1258 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1259 = "mhlo.broadcast_in_dim"(%1258) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1260 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1261 = "mhlo.broadcast_in_dim"(%1260) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1262 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1263 = "mhlo.broadcast_in_dim"(%1262) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1264 = call @aten.view.2552(%1263) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1265 = call @aten.sum.2688(%1264) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1266 = call @aten.view.2695(%1265) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1267 = call @aten.view.2552(%1263) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1268 = call @aten.permute.2700(%1267) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1269 = call @aten.mm.2704(%1268, %arg685) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1270 = call @aten.permute.2709(%1269) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1271 = call @aten.permute.2713(%1270) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1272 = call @aten.permute.2718(%arg73) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1273 = call @aten.permute.2709(%arg686) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1274 = call @aten.mm.2723(%1264, %1273) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1275 = call @aten.view.2728(%1274) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1276 = call @aten.view.2732(%1275) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1277 = call @aten.permute.2736(%1276) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1278 = call @aten.view.2740(%1277) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1279 = call @aten.permute.2718(%arg688) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1280 = call @aten.matmul.2744(%1278, %1279) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1281 = call @aten.view.2749(%1280) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1282 = call @aten._softmax_backward_data.2757(%1281, %arg224) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1283 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1284 = call @aten.div.2767(%1282, %1283) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1285 = call @aten.view.2773(%1284) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1286 = call @aten.matmul.2778(%1272, %1285) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1287 = call @aten.view.2783(%1286) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1288 = call @aten.permute.2787(%1287) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1289 = call @aten.permute.2791(%1288) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1290 = call @aten.view.2795(%1289) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1291 = call @aten.view.2552(%1290) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1292 = call @aten.sum.2688(%1291) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1293 = call @aten.view.2695(%1292) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1294 = call @aten.view.2552(%1290) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1295 = call @aten.permute.2700(%1294) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1296 = call @aten.mm.2704(%1295, %arg338) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1297 = call @aten.permute.2709(%1296) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1298 = call @aten.permute.2713(%1297) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1299 = call @aten.permute.2807(%arg684) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1300 = call @aten.matmul.2811(%1285, %1299) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1301 = call @aten.view.2816(%1300) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1302 = call @aten.permute.2820(%1301) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1303 = call @aten.view.2824(%1302) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1304 = call @aten.view.2552(%1303) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1305 = call @aten.sum.2688(%1304) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1306 = call @aten.view.2695(%1305) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1307 = call @aten.view.2824(%1302) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1308 = call @aten.view.2552(%1307) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1309 = call @aten.permute.2700(%1308) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1310 = call @aten.mm.2704(%1309, %arg333) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1311 = call @aten.permute.2709(%1310) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1312 = call @aten.permute.2713(%1311) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1313 = call @aten.permute.2837(%arg687) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1314 = call @aten.matmul.2841(%1313, %1278) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1315 = call @aten.view.2816(%1314) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1316 = call @aten.permute.2820(%1315) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1317 = call @aten.view.2824(%1316) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1318 = call @aten.view.2552(%1317) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1319 = call @aten.sum.2688(%1318) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1320 = call @aten.view.2695(%1319) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1321 = call @aten.view.2824(%1316) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1322 = call @aten.view.2552(%1321) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1323 = call @aten.permute.2700(%1322) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1324 = call @aten.mm.2704(%1323, %arg343) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1325 = call @aten.permute.2709(%1324) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1326 = call @aten.permute.2713(%1325) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1327 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1328 = "mhlo.broadcast_in_dim"(%1327) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1329 = call @aten.view.2552(%1328) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1330 = call @aten.permute.2858(%arg251) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1331 = call @aten.mm.2865(%1329, %1330) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1332 = call @aten.view.2870(%1331) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1333 = call @aten.gelu_backward.2874(%1332, %arg248) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1334 = call @aten.view.2960(%1333) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1335 = call @aten.sum.2968(%1334) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1336 = call @aten.view.2975(%1335) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1337 = call @aten.view.2960(%1333) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1338 = call @aten.permute.2980(%1337) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1339 = call @aten.mm.2984(%1338, %arg249) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1340 = call @aten.permute.2858(%1339) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1341 = call @aten.permute.2990(%1340) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1342 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1343 = "mhlo.broadcast_in_dim"(%1342) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1344 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1345 = "mhlo.broadcast_in_dim"(%1344) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1346 = call @aten.sum.2688(%1329) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1347 = call @aten.view.2695(%1346) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1348 = call @aten.view.2552(%1328) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1349 = call @aten.permute.2700(%1348) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1350 = call @aten.mm.3002(%1349, %arg252) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1351 = call @aten.permute.3007(%1350) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1352 = call @aten.permute.3011(%1351) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1353 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1354 = "mhlo.broadcast_in_dim"(%1353) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1355 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1356 = "mhlo.broadcast_in_dim"(%1355) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1357 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1358 = "mhlo.broadcast_in_dim"(%1357) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1359 = call @aten.view.2552(%1358) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1360 = call @aten.sum.2688(%1359) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1361 = call @aten.view.2695(%1360) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1362 = call @aten.view.2552(%1358) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1363 = call @aten.permute.2700(%1362) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1364 = call @aten.mm.2704(%1363, %arg179) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1365 = call @aten.permute.2709(%1364) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1366 = call @aten.permute.2713(%1365) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1367 = call @aten.permute.2718(%arg575) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1368 = call @aten.permute.2709(%arg597) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1369 = call @aten.mm.2723(%1359, %1368) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1370 = call @aten.view.2728(%1369) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1371 = call @aten.view.2732(%1370) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1372 = call @aten.permute.2736(%1371) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1373 = call @aten.view.2740(%1372) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1374 = call @aten.permute.2718(%arg347) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1375 = call @aten.matmul.2744(%1373, %1374) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1376 = call @aten.view.2749(%1375) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1377 = call @aten._softmax_backward_data.2757(%1376, %arg615) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1378 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1379 = call @aten.div.2767(%1377, %1378) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1380 = call @aten.view.2773(%1379) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1381 = call @aten.matmul.2778(%1367, %1380) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1382 = call @aten.view.2783(%1381) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1383 = call @aten.permute.2787(%1382) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1384 = call @aten.permute.2791(%1383) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1385 = call @aten.view.2795(%1384) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1386 = call @aten.view.2552(%1385) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1387 = call @aten.sum.2688(%1386) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1388 = call @aten.view.2695(%1387) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1389 = call @aten.view.2552(%1385) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1390 = call @aten.permute.2700(%1389) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1391 = call @aten.mm.2704(%1390, %arg601) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1392 = call @aten.permute.2709(%1391) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1393 = call @aten.permute.2713(%1392) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1394 = call @aten.permute.2807(%arg614) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1395 = call @aten.matmul.2811(%1380, %1394) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1396 = call @aten.view.2816(%1395) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1397 = call @aten.permute.2820(%1396) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1398 = call @aten.view.2824(%1397) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1399 = call @aten.view.2552(%1398) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1400 = call @aten.sum.2688(%1399) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1401 = call @aten.view.2695(%1400) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1402 = call @aten.view.2824(%1397) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1403 = call @aten.view.2552(%1402) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1404 = call @aten.permute.2700(%1403) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1405 = call @aten.mm.2704(%1404, %arg258) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1406 = call @aten.permute.2709(%1405) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1407 = call @aten.permute.2713(%1406) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1408 = call @aten.permute.2837(%arg170) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1409 = call @aten.matmul.2841(%1408, %1373) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1410 = call @aten.view.2816(%1409) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1411 = call @aten.permute.2820(%1410) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1412 = call @aten.view.2824(%1411) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1413 = call @aten.view.2552(%1412) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1414 = call @aten.sum.2688(%1413) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1415 = call @aten.view.2695(%1414) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1416 = call @aten.view.2824(%1411) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1417 = call @aten.view.2552(%1416) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1418 = call @aten.permute.2700(%1417) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1419 = call @aten.mm.2704(%1418, %arg605) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1420 = call @aten.permute.2709(%1419) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1421 = call @aten.permute.2713(%1420) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1422 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1423 = "mhlo.broadcast_in_dim"(%1422) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1424 = call @aten.view.2552(%1423) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1425 = call @aten.permute.2858(%arg544) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1426 = call @aten.mm.2865(%1424, %1425) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1427 = call @aten.view.2870(%1426) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1428 = call @aten.gelu_backward.2874(%1427, %arg193) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1429 = call @aten.view.2960(%1428) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1430 = call @aten.sum.2968(%1429) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1431 = call @aten.view.2975(%1430) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1432 = call @aten.view.2960(%1428) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1433 = call @aten.permute.2980(%1432) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1434 = call @aten.mm.2984(%1433, %arg192) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1435 = call @aten.permute.2858(%1434) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1436 = call @aten.permute.2990(%1435) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1437 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1438 = "mhlo.broadcast_in_dim"(%1437) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1439 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1440 = "mhlo.broadcast_in_dim"(%1439) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1441 = call @aten.sum.2688(%1424) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1442 = call @aten.view.2695(%1441) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1443 = call @aten.view.2552(%1423) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1444 = call @aten.permute.2700(%1443) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1445 = call @aten.mm.3002(%1444, %arg547) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1446 = call @aten.permute.3007(%1445) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1447 = call @aten.permute.3011(%1446) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1448 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1449 = "mhlo.broadcast_in_dim"(%1448) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1450 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1451 = "mhlo.broadcast_in_dim"(%1450) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1452 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1453 = "mhlo.broadcast_in_dim"(%1452) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1454 = call @aten.view.2552(%1453) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1455 = call @aten.sum.2688(%1454) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1456 = call @aten.view.2695(%1455) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1457 = call @aten.view.2552(%1453) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1458 = call @aten.permute.2700(%1457) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1459 = call @aten.mm.2704(%1458, %arg119) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1460 = call @aten.permute.2709(%1459) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1461 = call @aten.permute.2713(%1460) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1462 = call @aten.permute.2718(%arg250) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1463 = call @aten.permute.2709(%arg115) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1464 = call @aten.mm.2723(%1454, %1463) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1465 = call @aten.view.2728(%1464) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1466 = call @aten.view.2732(%1465) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1467 = call @aten.permute.2736(%1466) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1468 = call @aten.view.2740(%1467) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1469 = call @aten.permute.2718(%arg120) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1470 = call @aten.matmul.2744(%1468, %1469) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1471 = call @aten.view.2749(%1470) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1472 = call @aten._softmax_backward_data.2757(%1471, %arg121) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1473 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1474 = call @aten.div.2767(%1472, %1473) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1475 = call @aten.view.2773(%1474) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1476 = call @aten.matmul.2778(%1462, %1475) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1477 = call @aten.view.2783(%1476) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1478 = call @aten.permute.2787(%1477) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1479 = call @aten.permute.2791(%1478) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1480 = call @aten.view.2795(%1479) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1481 = call @aten.view.2552(%1480) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1482 = call @aten.sum.2688(%1481) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1483 = call @aten.view.2695(%1482) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1484 = call @aten.view.2552(%1480) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1485 = call @aten.permute.2700(%1484) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1486 = call @aten.mm.2704(%1485, %arg561) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1487 = call @aten.permute.2709(%1486) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1488 = call @aten.permute.2713(%1487) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1489 = call @aten.permute.2807(%arg102) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1490 = call @aten.matmul.2811(%1475, %1489) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1491 = call @aten.view.2816(%1490) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1492 = call @aten.permute.2820(%1491) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1493 = call @aten.view.2824(%1492) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1494 = call @aten.view.2552(%1493) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1495 = call @aten.sum.2688(%1494) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1496 = call @aten.view.2695(%1495) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1497 = call @aten.view.2824(%1492) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1498 = call @aten.view.2552(%1497) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1499 = call @aten.permute.2700(%1498) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1500 = call @aten.mm.2704(%1499, %arg559) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1501 = call @aten.permute.2709(%1500) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1502 = call @aten.permute.2713(%1501) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1503 = call @aten.permute.2837(%arg122) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1504 = call @aten.matmul.2841(%1503, %1468) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1505 = call @aten.view.2816(%1504) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1506 = call @aten.permute.2820(%1505) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1507 = call @aten.view.2824(%1506) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1508 = call @aten.view.2552(%1507) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1509 = call @aten.sum.2688(%1508) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1510 = call @aten.view.2695(%1509) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1511 = call @aten.view.2824(%1506) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1512 = call @aten.view.2552(%1511) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1513 = call @aten.permute.2700(%1512) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1514 = call @aten.mm.2704(%1513, %arg574) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1515 = call @aten.permute.2709(%1514) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1516 = call @aten.permute.2713(%1515) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1517 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1518 = "mhlo.broadcast_in_dim"(%1517) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1519 = call @aten.view.2552(%1518) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1520 = call @aten.permute.2858(%arg523) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1521 = call @aten.mm.2865(%1519, %1520) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1522 = call @aten.view.2870(%1521) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1523 = call @aten.gelu_backward.2874(%1522, %arg512) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1524 = call @aten.view.2960(%1523) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1525 = call @aten.sum.2968(%1524) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1526 = call @aten.view.2975(%1525) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1527 = call @aten.view.2960(%1523) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1528 = call @aten.permute.2980(%1527) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1529 = call @aten.mm.2984(%1528, %arg513) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1530 = call @aten.permute.2858(%1529) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1531 = call @aten.permute.2990(%1530) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1532 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1533 = "mhlo.broadcast_in_dim"(%1532) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1534 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1535 = "mhlo.broadcast_in_dim"(%1534) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1536 = call @aten.sum.2688(%1519) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1537 = call @aten.view.2695(%1536) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1538 = call @aten.view.2552(%1518) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1539 = call @aten.permute.2700(%1538) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1540 = call @aten.mm.3002(%1539, %arg514) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1541 = call @aten.permute.3007(%1540) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1542 = call @aten.permute.3011(%1541) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1543 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1544 = "mhlo.broadcast_in_dim"(%1543) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1545 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1546 = "mhlo.broadcast_in_dim"(%1545) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1547 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1548 = "mhlo.broadcast_in_dim"(%1547) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1549 = call @aten.view.2552(%1548) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1550 = call @aten.sum.2688(%1549) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1551 = call @aten.view.2695(%1550) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1552 = call @aten.view.2552(%1548) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1553 = call @aten.permute.2700(%1552) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1554 = call @aten.mm.2704(%1553, %arg373) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1555 = call @aten.permute.2709(%1554) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1556 = call @aten.permute.2713(%1555) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1557 = call @aten.permute.2718(%arg708) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1558 = call @aten.permute.2709(%arg134) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1559 = call @aten.mm.2723(%1549, %1558) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1560 = call @aten.view.2728(%1559) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1561 = call @aten.view.2732(%1560) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1562 = call @aten.permute.2736(%1561) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1563 = call @aten.view.2740(%1562) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1564 = call @aten.permute.2718(%arg124) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1565 = call @aten.matmul.2744(%1563, %1564) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1566 = call @aten.view.2749(%1565) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1567 = call @aten._softmax_backward_data.2757(%1566, %arg253) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1568 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1569 = call @aten.div.2767(%1567, %1568) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1570 = call @aten.view.2773(%1569) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1571 = call @aten.matmul.2778(%1557, %1570) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1572 = call @aten.view.2783(%1571) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1573 = call @aten.permute.2787(%1572) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1574 = call @aten.permute.2791(%1573) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1575 = call @aten.view.2795(%1574) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1576 = call @aten.view.2552(%1575) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1577 = call @aten.sum.2688(%1576) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1578 = call @aten.view.2695(%1577) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1579 = call @aten.view.2552(%1575) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1580 = call @aten.permute.2700(%1579) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1581 = call @aten.mm.2704(%1580, %arg541) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1582 = call @aten.permute.2709(%1581) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1583 = call @aten.permute.2713(%1582) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1584 = call @aten.permute.2807(%arg149) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1585 = call @aten.matmul.2811(%1570, %1584) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1586 = call @aten.view.2816(%1585) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1587 = call @aten.permute.2820(%1586) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1588 = call @aten.view.2824(%1587) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1589 = call @aten.view.2552(%1588) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1590 = call @aten.sum.2688(%1589) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1591 = call @aten.view.2695(%1590) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1592 = call @aten.view.2824(%1587) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1593 = call @aten.view.2552(%1592) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1594 = call @aten.permute.2700(%1593) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1595 = call @aten.mm.2704(%1594, %arg536) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1596 = call @aten.permute.2709(%1595) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1597 = call @aten.permute.2713(%1596) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1598 = call @aten.permute.2837(%arg361) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1599 = call @aten.matmul.2841(%1598, %1563) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1600 = call @aten.view.2816(%1599) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1601 = call @aten.permute.2820(%1600) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1602 = call @aten.view.2824(%1601) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1603 = call @aten.view.2552(%1602) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1604 = call @aten.sum.2688(%1603) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1605 = call @aten.view.2695(%1604) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1606 = call @aten.view.2824(%1601) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1607 = call @aten.view.2552(%1606) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1608 = call @aten.permute.2700(%1607) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1609 = call @aten.mm.2704(%1608, %arg140) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1610 = call @aten.permute.2709(%1609) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1611 = call @aten.permute.2713(%1610) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1612 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1613 = "mhlo.broadcast_in_dim"(%1612) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1614 = call @aten.view.2552(%1613) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1615 = call @aten.permute.2858(%arg401) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1616 = call @aten.mm.2865(%1614, %1615) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1617 = call @aten.view.2870(%1616) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1618 = call @aten.gelu_backward.2874(%1617, %arg389) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1619 = call @aten.view.2960(%1618) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1620 = call @aten.sum.2968(%1619) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1621 = call @aten.view.2975(%1620) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1622 = call @aten.view.2960(%1618) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1623 = call @aten.permute.2980(%1622) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1624 = call @aten.mm.2984(%1623, %arg390) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1625 = call @aten.permute.2858(%1624) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1626 = call @aten.permute.2990(%1625) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1627 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1628 = "mhlo.broadcast_in_dim"(%1627) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1629 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1630 = "mhlo.broadcast_in_dim"(%1629) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1631 = call @aten.sum.2688(%1614) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1632 = call @aten.view.2695(%1631) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1633 = call @aten.view.2552(%1613) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1634 = call @aten.permute.2700(%1633) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1635 = call @aten.mm.3002(%1634, %arg404) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1636 = call @aten.permute.3007(%1635) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1637 = call @aten.permute.3011(%1636) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1638 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1639 = "mhlo.broadcast_in_dim"(%1638) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1640 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1641 = "mhlo.broadcast_in_dim"(%1640) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1642 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1643 = "mhlo.broadcast_in_dim"(%1642) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1644 = call @aten.view.2552(%1643) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1645 = call @aten.sum.2688(%1644) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1646 = call @aten.view.2695(%1645) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1647 = call @aten.view.2552(%1643) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1648 = call @aten.permute.2700(%1647) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1649 = call @aten.mm.2704(%1648, %arg374) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1650 = call @aten.permute.2709(%1649) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1651 = call @aten.permute.2713(%1650) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1652 = call @aten.permute.2718(%arg129) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1653 = call @aten.permute.2709(%arg106) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1654 = call @aten.mm.2723(%1644, %1653) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1655 = call @aten.view.2728(%1654) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1656 = call @aten.view.2732(%1655) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1657 = call @aten.permute.2736(%1656) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1658 = call @aten.view.2740(%1657) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1659 = call @aten.permute.2718(%arg130) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1660 = call @aten.matmul.2744(%1658, %1659) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1661 = call @aten.view.2749(%1660) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1662 = call @aten._softmax_backward_data.2757(%1661, %arg578) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1663 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1664 = call @aten.div.2767(%1662, %1663) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1665 = call @aten.view.2773(%1664) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1666 = call @aten.matmul.2778(%1652, %1665) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1667 = call @aten.view.2783(%1666) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1668 = call @aten.permute.2787(%1667) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1669 = call @aten.permute.2791(%1668) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1670 = call @aten.view.2795(%1669) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1671 = call @aten.view.2552(%1670) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1672 = call @aten.sum.2688(%1671) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1673 = call @aten.view.2695(%1672) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1674 = call @aten.view.2552(%1670) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1675 = call @aten.permute.2700(%1674) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1676 = call @aten.mm.2704(%1675, %arg103) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1677 = call @aten.permute.2709(%1676) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1678 = call @aten.permute.2713(%1677) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1679 = call @aten.permute.2807(%arg128) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1680 = call @aten.matmul.2811(%1665, %1679) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1681 = call @aten.view.2816(%1680) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1682 = call @aten.permute.2820(%1681) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1683 = call @aten.view.2824(%1682) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1684 = call @aten.view.2552(%1683) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1685 = call @aten.sum.2688(%1684) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1686 = call @aten.view.2695(%1685) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1687 = call @aten.view.2824(%1682) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1688 = call @aten.view.2552(%1687) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1689 = call @aten.permute.2700(%1688) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1690 = call @aten.mm.2704(%1689, %arg100) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1691 = call @aten.permute.2709(%1690) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1692 = call @aten.permute.2713(%1691) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1693 = call @aten.permute.2837(%arg380) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1694 = call @aten.matmul.2841(%1693, %1658) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1695 = call @aten.view.2816(%1694) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1696 = call @aten.permute.2820(%1695) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1697 = call @aten.view.2824(%1696) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1698 = call @aten.view.2552(%1697) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1699 = call @aten.sum.2688(%1698) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1700 = call @aten.view.2695(%1699) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1701 = call @aten.view.2824(%1696) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1702 = call @aten.view.2552(%1701) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1703 = call @aten.permute.2700(%1702) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1704 = call @aten.mm.2704(%1703, %arg117) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1705 = call @aten.permute.2709(%1704) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1706 = call @aten.permute.2713(%1705) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1707 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1708 = "mhlo.broadcast_in_dim"(%1707) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1709 = call @aten.view.2552(%1708) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1710 = call @aten.permute.2858(%arg101) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1711 = call @aten.mm.2865(%1709, %1710) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1712 = call @aten.view.2870(%1711) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1713 = call @aten.gelu_backward.2874(%1712, %arg93) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1714 = call @aten.view.2960(%1713) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1715 = call @aten.sum.2968(%1714) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1716 = call @aten.view.2975(%1715) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1717 = call @aten.view.2960(%1713) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1718 = call @aten.permute.2980(%1717) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1719 = call @aten.mm.2984(%1718, %arg411) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1720 = call @aten.permute.2858(%1719) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1721 = call @aten.permute.2990(%1720) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1722 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1723 = "mhlo.broadcast_in_dim"(%1722) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1724 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1725 = "mhlo.broadcast_in_dim"(%1724) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1726 = call @aten.sum.2688(%1709) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1727 = call @aten.view.2695(%1726) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1728 = call @aten.view.2552(%1708) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1729 = call @aten.permute.2700(%1728) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1730 = call @aten.mm.3002(%1729, %arg105) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1731 = call @aten.permute.3007(%1730) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1732 = call @aten.permute.3011(%1731) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1733 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %1734 = "mhlo.broadcast_in_dim"(%1733) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1735 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %1736 = "mhlo.broadcast_in_dim"(%1735) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1737 = mhlo.constant dense<0xFFC22638> : tensor<f32>
    %1738 = "mhlo.broadcast_in_dim"(%1737) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1739 = call @aten.view.2552(%1738) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1740 = call @aten.sum.2688(%1739) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1741 = call @aten.view.2695(%1740) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1742 = call @aten.view.2552(%1738) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1743 = call @aten.permute.2700(%1742) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1744 = call @aten.mm.2704(%1743, %arg487) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1745 = call @aten.permute.2709(%1744) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1746 = call @aten.permute.2713(%1745) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1747 = call @aten.permute.2718(%arg412) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1748 = call @aten.permute.2709(%arg492) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1749 = call @aten.mm.2723(%1739, %1748) : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    %1750 = call @aten.view.2728(%1749) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    %1751 = call @aten.view.2732(%1750) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    %1752 = call @aten.permute.2736(%1751) {xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    %1753 = call @aten.view.2740(%1752) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    %1754 = call @aten.permute.2718(%arg496) {xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    %1755 = call @aten.matmul.2744(%1753, %1754) : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    %1756 = call @aten.view.2749(%1755) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1757 = call @aten._softmax_backward_data.2757(%1756, %arg99) : (tensor<2x12x240x240xf32>, tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32>
    %1758 = mhlo.constant dense<8.000000e+00> : tensor<f32>
    %1759 = call @aten.div.2767(%1757, %1758) : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240x240xf32>
    %1760 = call @aten.view.2773(%1759) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    %1761 = call @aten.matmul.2778(%1747, %1760) : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    %1762 = call @aten.view.2783(%1761) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    %1763 = call @aten.permute.2787(%1762) {xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    %1764 = call @aten.permute.2791(%1763) {xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1765 = call @aten.view.2795(%1764) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1766 = call @aten.view.2552(%1765) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1767 = call @aten.sum.2688(%1766) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1768 = call @aten.view.2695(%1767) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1769 = call @aten.view.2552(%1765) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1770 = call @aten.permute.2700(%1769) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1771 = call @aten.mm.2704(%1770, %arg114) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1772 = call @aten.permute.2709(%1771) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1773 = call @aten.permute.2713(%1772) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1774 = call @aten.permute.2807(%arg481) {xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    %1775 = call @aten.matmul.2811(%1760, %1774) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1776 = call @aten.view.2816(%1775) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1777 = call @aten.permute.2820(%1776) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1778 = call @aten.view.2824(%1777) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1779 = call @aten.view.2552(%1778) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1780 = call @aten.sum.2688(%1779) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1781 = call @aten.view.2695(%1780) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1782 = call @aten.view.2824(%1777) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1783 = call @aten.view.2552(%1782) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1784 = call @aten.permute.2700(%1783) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1785 = call @aten.mm.2704(%1784, %arg113) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1786 = call @aten.permute.2709(%1785) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1787 = call @aten.permute.2713(%1786) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1788 = call @aten.permute.2837(%arg489) {xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    %1789 = call @aten.matmul.2841(%1788, %1753) : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    %1790 = call @aten.view.2816(%1789) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    %1791 = call @aten.permute.2820(%1790) {xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    %1792 = call @aten.view.2824(%1791) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1793 = call @aten.view.2552(%1792) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1794 = call @aten.sum.2688(%1793) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1795 = call @aten.view.2695(%1794) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1796 = call @aten.view.2824(%1791) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    %1797 = call @aten.view.2552(%1796) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1798 = call @aten.permute.2700(%1797) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1799 = call @aten.mm.2704(%1798, %arg123) : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    %1800 = call @aten.permute.2709(%1799) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1801 = call @aten.permute.2713(%1800) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1802 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %1803 = "mhlo.broadcast_in_dim"(%1802) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x768xf32>
    %1804 = call @aten.view.2552(%1803) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1805 = call @aten.permute.2858(%arg50) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1806 = call @aten.mm.2865(%1804, %1805) : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    %1807 = call @aten.view.2870(%1806) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    %1808 = call @aten.gelu_backward.2874(%1807, %arg47) : (tensor<2x240x3072xf32>, tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %1809 = call @aten.view.2960(%1808) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1810 = call @aten.sum.2968(%1809) : (tensor<480x3072xf32>) -> tensor<1x3072xf32>
    %1811 = call @aten.view.2975(%1810) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %1812 = call @aten.view.2960(%1808) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    %1813 = call @aten.permute.2980(%1812) {xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    %1814 = call @aten.mm.2984(%1813, %arg48) : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    %1815 = call @aten.permute.2858(%1814) {xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1816 = call @aten.permute.2990(%1815) : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1817 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %1818 = "mhlo.broadcast_in_dim"(%1817) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1819 = mhlo.constant dense<0xFFFFFFFF> : tensor<f32>
    %1820 = "mhlo.broadcast_in_dim"(%1819) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<768xf32>
    %1821 = call @aten.sum.2688(%1804) : (tensor<480x768xf32>) -> tensor<1x768xf32>
    %1822 = call @aten.view.2695(%1821) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1823 = call @aten.view.2552(%1803) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    %1824 = call @aten.permute.2700(%1823) {xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    %1825 = call @aten.mm.3002(%1824, %arg54) : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    %1826 = call @aten.permute.3007(%1825) {xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1827 = call @aten.permute.3011(%1826) : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1828 = call @aten.permute.4085(%arg468) {xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %1829 = call @aten.mm.4089(%455, %1828) : (tensor<2x256xf32>, tensor<256x768xf32>) -> tensor<2x768xf32>
    %1830 = "mhlo.slice"(%18) {limit_indices = dense<[2, 2816]> : tensor<2xi64>, start_indices = dense<[0, 2048]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2816xf32>) -> tensor<2x768xf32>
    %1831 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1832 = call @aten.expand.2616(%1831) : (tensor<f32>) -> tensor<2x768xf32>
    %1833 = call @aten.mul.4069(%1830, %1832) : (tensor<2x768xf32>, tensor<2x768xf32>) -> tensor<2x768xf32>
    %1834 = call @aten.add.4094(%1829, %1833) : (tensor<2x768xf32>, tensor<2x768xf32>) -> tensor<2x768xf32>
    %1835 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1836 = call @aten.expand.2616(%1835) : (tensor<f32>) -> tensor<2x768xf32>
    %1837 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1838 = call @aten.expand.2616(%1837) : (tensor<f32>) -> tensor<2x768xf32>
    %1839 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %1840 = call @aten.expand.2616(%1839) : (tensor<f32>) -> tensor<2x768xf32>
    %1841 = call @aten.pow.4062(%arg458, %1840) : (tensor<2x768xf32>, tensor<2x768xf32>) -> tensor<2x768xf32>
    %1842 = call @aten.mul.4069(%1838, %1841) : (tensor<2x768xf32>, tensor<2x768xf32>) -> tensor<2x768xf32>
    %1843 = call @aten.sub.4076(%1836, %1842) : (tensor<2x768xf32>, tensor<2x768xf32>) -> tensor<2x768xf32>
    %1844 = call @aten.mul.4069(%1834, %1843) : (tensor<2x768xf32>, tensor<2x768xf32>) -> tensor<2x768xf32>
    %1845 = call @aten.sum.4104(%1844) : (tensor<2x768xf32>) -> tensor<1x768xf32>
    %1846 = call @aten.view.2695(%1845) : (tensor<1x768xf32>) -> tensor<768xf32>
    %1847 = call @aten.permute.4112(%1844) {xla_shape = "f32[768,2]{0,1}"} : (tensor<2x768xf32>) -> tensor<768x2xf32>
    %1848 = call @aten.mm.4116(%1847, %arg442) : (tensor<768x2xf32>, tensor<2x768xf32>) -> tensor<768x768xf32>
    %1849 = call @aten.permute.2709(%1848) {xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1850 = call @aten.permute.2713(%1849) : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1851 = "mhlo.tuple"(%412, %416, %418, %422, %424, %428, %430, %434, %436, %440, %442, %446, %448, %452, %457, %461, %462, %463, %464, %465, %466, %467, %468, %469, %470, %471, %472, %473, %474, %475, %476, %477, %478, %479, %480, %481, %482, %483, %484, %485, %486, %487, %488, %489, %490, %491, %492, %493, %494, %495, %496, %497, %498, %499, %500, %501, %502, %503, %504, %505, %506, %507, %508, %509, %510, %511, %512, %513, %514, %515, %516, %517, %518, %519, %520, %521, %522, %523, %524, %525, %526, %527, %528, %529, %530, %531, %532, %533, %534, %535, %536, %537, %538, %539, %540, %541, %542, %543, %544, %545, %546, %547, %548, %549, %550, %551, %552, %553, %554, %555, %556, %557, %558, %559, %560, %561, %562, %563, %564, %565, %566, %567, %568, %569, %570, %571, %572, %573, %574, %575, %576, %577, %578, %579, %580, %581, %582, %583, %584, %585, %586, %587, %588, %589, %590, %591, %592, %593, %594, %595, %596, %597, %598, %599, %600, %601, %602, %603, %604, %605, %606, %607, %608, %609, %610, %611, %612, %613, %614, %615, %616, %617, %618, %619, %620, %622, %624, %647, %667, %687, %689, %691, %696, %701, %723, %728, %736, %742, %750, %756, %766, %771, %773, %775, %777, %782, %784, %786, %791, %796, %818, %823, %831, %837, %845, %851, %861, %866, %868, %870, %872, %877, %879, %881, %886, %891, %913, %918, %926, %932, %940, %946, %956, %961, %963, %965, %967, %972, %974, %976, %981, %986, %1008, %1013, %1021, %1027, %1035, %1041, %1051, %1056, %1058, %1060, %1062, %1067, %1069, %1071, %1076, %1081, %1103, %1108, %1116, %1122, %1130, %1136, %1146, %1151, %1153, %1155, %1157, %1162, %1164, %1166, %1171, %1176, %1198, %1203, %1211, %1217, %1225, %1231, %1241, %1246, %1248, %1250, %1252, %1257, %1259, %1261, %1266, %1271, %1293, %1298, %1306, %1312, %1320, %1326, %1336, %1341, %1343, %1345, %1347, %1352, %1354, %1356, %1361, %1366, %1388, %1393, %1401, %1407, %1415, %1421, %1431, %1436, %1438, %1440, %1442, %1447, %1449, %1451, %1456, %1461, %1483, %1488, %1496, %1502, %1510, %1516, %1526, %1531, %1533, %1535, %1537, %1542, %1544, %1546, %1551, %1556, %1578, %1583, %1591, %1597, %1605, %1611, %1621, %1626, %1628, %1630, %1632, %1637, %1639, %1641, %1646, %1651, %1673, %1678, %1686, %1692, %1700, %1706, %1716, %1721, %1723, %1725, %1727, %1732, %1734, %1736, %1741, %1746, %1768, %1773, %1781, %1787, %1795, %1801, %1811, %1816, %1818, %1820, %1822, %1827, %1846, %1850) {xla_shape = "(f32[19]{0}, f32[19,256]{1,0}, f32[19]{0}, f32[19,128]{1,0}, f32[2]{0}, /*index=5*/f32[2,256]{1,0}, f32[2]{0}, f32[2,256]{1,0}, f32[256]{0}, f32[256,2816]{1,0}, /*index=10*/f32[128]{0}, f32[128,2816]{1,0}, f32[256]{0}, f32[256,2048]{1,0}, f32[256]{0}, /*index=15*/f32[256,768]{1,0}, f32[64]{0}, f32[64]{0}, f32[64,3,7,7]{0,1,3,2}, f32[64]{0}, /*index=20*/f32[64]{0}, f32[64,64,1,1]{0,1,3,2}, f32[64]{0}, f32[64]{0}, f32[64,64,3,3]{0,1,3,2}, /*index=25*/f32[256]{0}, f32[256]{0}, f32[256,64,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, /*index=30*/f32[256,64,1,1]{0,1,3,2}, f32[64]{0}, f32[64]{0}, f32[64,256,1,1]{0,1,3,2}, f32[64]{0}, /*index=35*/f32[64]{0}, f32[64,64,3,3]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,64,1,1]{0,1,3,2}, /*index=40*/f32[64]{0}, f32[64]{0}, f32[64,256,1,1]{0,1,3,2}, f32[64]{0}, f32[64]{0}, /*index=45*/f32[64,64,3,3]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,64,1,1]{0,1,3,2}, f32[128]{0}, /*index=50*/f32[128]{0}, f32[128,256,1,1]{0,1,3,2}, f32[128]{0}, f32[128]{0}, f32[128,128,3,3]{0,1,3,2}, /*index=55*/f32[512]{0}, f32[512]{0}, f32[512,128,1,1]{0,1,3,2}, f32[512]{0}, f32[512]{0}, /*index=60*/f32[512,256,1,1]{0,1,3,2}, f32[128]{0}, f32[128]{0}, f32[128,512,1,1]{0,1,3,2}, f32[128]{0}, /*index=65*/f32[128]{0}, f32[128,128,3,3]{0,1,3,2}, f32[512]{0}, f32[512]{0}, f32[512,128,1,1]{0,1,3,2}, /*index=70*/f32[128]{0}, f32[128]{0}, f32[128,512,1,1]{0,1,3,2}, f32[128]{0}, f32[128]{0}, /*index=75*/f32[128,128,3,3]{0,1,3,2}, f32[512]{0}, f32[512]{0}, f32[512,128,1,1]{0,1,3,2}, f32[128]{0}, /*index=80*/f32[128]{0}, f32[128,512,1,1]{0,1,3,2}, f32[128]{0}, f32[128]{0}, f32[128,128,3,3]{0,1,3,2}, /*index=85*/f32[512]{0}, f32[512]{0}, f32[512,128,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, /*index=90*/f32[256,512,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,256,3,3]{0,1,3,2}, f32[1024]{0}, /*index=95*/f32[1024]{0}, f32[1024,256,1,1]{0,1,3,2}, f32[1024]{0}, f32[1024]{0}, f32[1024,512,1,1]{0,1,3,2}, /*index=100*/f32[256]{0}, f32[256]{0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, /*index=105*/f32[256,256,3,3]{0,1,3,2}, f32[1024]{0}, f32[1024]{0}, f32[1024,256,1,1]{0,1,3,2}, f32[256]{0}, /*index=110*/f32[256]{0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,256,3,3]{0,1,3,2}, /*index=115*/f32[1024]{0}, f32[1024]{0}, f32[1024,256,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, /*index=120*/f32[256,1024,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,256,3,3]{0,1,3,2}, f32[1024]{0}, /*index=125*/f32[1024]{0}, f32[1024,256,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,1024,1,1]{0,1,3,2}, /*index=130*/f32[256]{0}, f32[256]{0}, f32[256,256,3,3]{0,1,3,2}, f32[1024]{0}, f32[1024]{0}, /*index=135*/f32[1024,256,1,1]{0,1,3,2}, f32[256]{0}, f32[256]{0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0}, /*index=140*/f32[256]{0}, f32[256,256,3,3]{0,1,3,2}, f32[1024]{0}, f32[1024]{0}, f32[1024,256,1,1]{0,1,3,2}, /*index=145*/f32[512]{0}, f32[512]{0}, f32[512,1024,1,1]{0,1,3,2}, f32[512]{0}, f32[512]{0}, /*index=150*/f32[512,512,3,3]{0,1,3,2}, f32[2048]{0}, f32[2048]{0}, f32[2048,512,1,1]{0,1,3,2}, f32[2048]{0}, /*index=155*/f32[2048]{0}, f32[2048,1024,1,1]{0,1,3,2}, f32[512]{0}, f32[512]{0}, f32[512,2048,1,1]{0,1,3,2}, /*index=160*/f32[512]{0}, f32[512]{0}, f32[512,512,3,3]{0,1,3,2}, f32[2048]{0}, f32[2048]{0}, /*index=165*/f32[2048,512,1,1]{0,1,3,2}, f32[512]{0}, f32[512]{0}, f32[512,2048,1,1]{0,1,3,2}, f32[512]{0}, /*index=170*/f32[512]{0}, f32[512,512,3,3]{0,1,3,2}, f32[2048]{0}, f32[2048]{0}, f32[2048,512,1,1]{0,1,3,2}, /*index=175*/f32[768]{0}, f32[768]{0}, f32[512,768]{1,0}, f32[2,768]{1,0}, f32[21128,768]{1,0}, /*index=180*/f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, /*index=185*/f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, /*index=190*/f32[3072]{0}, f32[3072,768]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, /*index=195*/f32[768,3072]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,768]{1,0}, /*index=200*/f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, /*index=205*/f32[768,768]{1,0}, f32[3072]{0}, f32[3072,768]{1,0}, f32[768]{0}, f32[768]{0}, /*index=210*/f32[768]{0}, f32[768,3072]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, /*index=215*/f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, /*index=220*/f32[768]{0}, f32[768,768]{1,0}, f32[3072]{0}, f32[3072,768]{1,0}, f32[768]{0}, /*index=225*/f32[768]{0}, f32[768]{0}, f32[768,3072]{1,0}, f32[768]{0}, f32[768]{0}, /*index=230*/f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, /*index=235*/f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[3072]{0}, f32[3072,768]{1,0}, /*index=240*/f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,3072]{1,0}, f32[768]{0}, /*index=245*/f32[768]{0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, /*index=250*/f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[3072]{0}, /*index=255*/f32[3072,768]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,3072]{1,0}, /*index=260*/f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, /*index=265*/f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, /*index=270*/f32[3072]{0}, f32[3072,768]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, /*index=275*/f32[768,3072]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,768]{1,0}, /*index=280*/f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, /*index=285*/f32[768,768]{1,0}, f32[3072]{0}, f32[3072,768]{1,0}, f32[768]{0}, f32[768]{0}, /*index=290*/f32[768]{0}, f32[768,3072]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, /*index=295*/f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, /*index=300*/f32[768]{0}, f32[768,768]{1,0}, f32[3072]{0}, f32[3072,768]{1,0}, f32[768]{0}, /*index=305*/f32[768]{0}, f32[768]{0}, f32[768,3072]{1,0}, f32[768]{0}, f32[768]{0}, /*index=310*/f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, /*index=315*/f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[3072]{0}, f32[3072,768]{1,0}, /*index=320*/f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,3072]{1,0}, f32[768]{0}, /*index=325*/f32[768]{0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, /*index=330*/f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[3072]{0}, /*index=335*/f32[3072,768]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,3072]{1,0}, /*index=340*/f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, /*index=345*/f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, /*index=350*/f32[3072]{0}, f32[3072,768]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, /*index=355*/f32[768,3072]{1,0}, f32[768]{0}, f32[768]{0}, f32[768]{0}, f32[768,768]{1,0}, /*index=360*/f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, f32[768,768]{1,0}, f32[768]{0}, /*index=365*/f32[768,768]{1,0}, f32[3072]{0}, f32[3072,768]{1,0}, f32[768]{0}, f32[768]{0}, /*index=370*/f32[768]{0}, f32[768,3072]{1,0}, f32[768]{0}, f32[768,768]{1,0})"} : (tensor<19xf32>, tensor<19x256xf32>, tensor<19xf32>, tensor<19x128xf32>, tensor<2xf32>, tensor<2x256xf32>, tensor<2xf32>, tensor<2x256xf32>, tensor<256xf32>, tensor<256x2816xf32>, tensor<128xf32>, tensor<128x2816xf32>, tensor<256xf32>, tensor<256x2048xf32>, tensor<256xf32>, tensor<256x768xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x512x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<512x768xf32>, tensor<2x768xf32>, tensor<21128x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768xf32>, tensor<768x768xf32>) -> !tuple
    return %1851 : !tuple
  }
  func.func private @aten.permute.978(%arg0: tensor<256x2xf32>) -> tensor<2x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[2,256]{0,1}"} : (tensor<256x2xf32>) -> tensor<2x256xf32>
    return %0 : tensor<2x256xf32>
  }
  func.func private @aten.mm.982(%arg0: tensor<2x2xf32>, %arg1: tensor<2x256xf32>) -> tensor<2x256xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x2xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    return %0 : tensor<2x256xf32>
  }
  func.func private @aten.threshold_backward.888(%arg0: tensor<2x256xf32>, %arg1: tensor<2x256xf32>) -> tensor<2x256xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x256xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<2x256xf32>, tensor<2x256xf32>) -> tensor<2x256xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x256xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<2x256xi1>, tensor<2x256xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    return %5 : tensor<2x256xf32>
  }
  func.func private @aten.permute.1136(%arg0: tensor<2048x256xf32>) -> tensor<256x2048xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[256,2048]{0,1}"} : (tensor<2048x256xf32>) -> tensor<256x2048xf32>
    return %0 : tensor<256x2048xf32>
  }
  func.func private @aten.mm.1140(%arg0: tensor<2x256xf32>, %arg1: tensor<256x2048xf32>) -> tensor<2x2048xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x256xf32>, tensor<256x2048xf32>) -> tensor<2x2048xf32>
    return %0 : tensor<2x2048xf32>
  }
  func.func private @aten.permute.879(%arg0: tensor<256x19xf32>) -> tensor<19x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[19,256]{0,1}"} : (tensor<256x19xf32>) -> tensor<19x256xf32>
    return %0 : tensor<19x256xf32>
  }
  func.func private @aten.mm.883(%arg0: tensor<2x19xf32>, %arg1: tensor<19x256xf32>) -> tensor<2x256xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x19xf32>, tensor<19x256xf32>) -> tensor<2x256xf32>
    return %0 : tensor<2x256xf32>
  }
  func.func private @aten.permute.1116(%arg0: tensor<2816x256xf32>) -> tensor<256x2816xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[256,2816]{0,1}"} : (tensor<2816x256xf32>) -> tensor<256x2816xf32>
    return %0 : tensor<256x2816xf32>
  }
  func.func private @aten.mm.1120(%arg0: tensor<2x256xf32>, %arg1: tensor<256x2816xf32>) -> tensor<2x2816xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x256xf32>, tensor<256x2816xf32>) -> tensor<2x2816xf32>
    return %0 : tensor<2x2816xf32>
  }
  func.func private @aten.permute.927(%arg0: tensor<128x19xf32>) -> tensor<19x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[19,128]{0,1}"} : (tensor<128x19xf32>) -> tensor<19x128xf32>
    return %0 : tensor<19x128xf32>
  }
  func.func private @aten.mm.931(%arg0: tensor<2x19xf32>, %arg1: tensor<19x128xf32>) -> tensor<2x128xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x19xf32>, tensor<19x128xf32>) -> tensor<2x128xf32>
    return %0 : tensor<2x128xf32>
  }
  func.func private @aten.threshold_backward.936(%arg0: tensor<2x128xf32>, %arg1: tensor<2x128xf32>) -> tensor<2x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<2x128xf32>, tensor<2x128xf32>) -> tensor<2x128xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<2x128xi1>, tensor<2x128xf32>, tensor<2x128xf32>) -> tensor<2x128xf32>
    return %5 : tensor<2x128xf32>
  }
  func.func private @aten.permute.1102(%arg0: tensor<2816x128xf32>) -> tensor<128x2816xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[128,2816]{0,1}"} : (tensor<2816x128xf32>) -> tensor<128x2816xf32>
    return %0 : tensor<128x2816xf32>
  }
  func.func private @aten.mm.1106(%arg0: tensor<2x128xf32>, %arg1: tensor<128x2816xf32>) -> tensor<2x2816xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x128xf32>, tensor<128x2816xf32>) -> tensor<2x2816xf32>
    return %0 : tensor<2x2816xf32>
  }
  func.func private @aten.expand.1095(%arg0: tensor<f32>) -> tensor<2x2816xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x2816xf32>
    return %3 : tensor<2x2816xf32>
  }
  func.func private @aten.mul.1111(%arg0: tensor<2x2816xf32>, %arg1: tensor<2x2816xf32>) -> tensor<2x2816xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x2816xf32>
    return %0 : tensor<2x2816xf32>
  }
  func.func private @aten.add.1125(%arg0: tensor<2x2816xf32>, %arg1: tensor<2x2816xf32>) -> tensor<2x2816xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x2816xf32>
    return %0 : tensor<2x2816xf32>
  }
  func.func private @aten.expand.1087(%arg0: tensor<f32>) -> tensor<2x2048xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x2048xf32>
    return %3 : tensor<2x2048xf32>
  }
  func.func private @aten.mul.1131(%arg0: tensor<2x2048xf32>, %arg1: tensor<2x2048xf32>) -> tensor<2x2048xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x2048xf32>
    return %0 : tensor<2x2048xf32>
  }
  func.func private @aten.add.1145(%arg0: tensor<2x2048xf32>, %arg1: tensor<2x2048xf32>) -> tensor<2x2048xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x2048xf32>
    return %0 : tensor<2x2048xf32>
  }
  func.func private @aten.view.1150(%arg0: tensor<2x2048xf32>) -> tensor<2x2048x1xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x2048xf32>) -> tensor<2x2048x1xf32>
    return %0 : tensor<2x2048x1xf32>
  }
  func.func private @aten.view.1154(%arg0: tensor<2x2048x1xf32>) -> tensor<2x2048x1x1xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x2048x1xf32>) -> tensor<2x2048x1x1xf32>
    return %0 : tensor<2x2048x1x1xf32>
  }
  func.func private @aten.view.1158(%arg0: tensor<2x2048x1x1xf32>) -> tensor<2x2048x1x1x1xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x2048x1x1xf32>) -> tensor<2x2048x1x1x1xf32>
    return %0 : tensor<2x2048x1x1x1xf32>
  }
  func.func private @aten.expand.1162(%arg0: tensor<2x2048x1x1x1xf32>) -> tensor<2x2048x8x8x8xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3, 4]> : tensor<5xi64>} : (tensor<2x2048x1x1x1xf32>) -> tensor<2x2048x1x1x1xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<2x2048x1x1x1xf32>) -> tensor<2x2048xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x2048xf32>) -> tensor<2x2048x8x8x8xf32>
    return %2 : tensor<2x2048x8x8x8xf32>
  }
  func.func private @aten.div.1168(%arg0: tensor<2x2048x8x8x8xf32>, %arg1: tensor<f32>) -> tensor<2x2048x8x8x8xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x2048x8x8x8xf32>
    %1 = mhlo.divide %arg0, %0 : tensor<2x2048x8x8x8xf32>
    return %1 : tensor<2x2048x8x8x8xf32>
  }
  func.func private @aten.permute.1174(%arg0: tensor<2x2048x8x8x8xf32>) -> tensor<2x8x2048x8x8xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3, 4]> : tensor<5xi64>, xla_shape = "f32[2,8,2048,8,8]{4,3,1,2,0}"} : (tensor<2x2048x8x8x8xf32>) -> tensor<2x8x2048x8x8xf32>
    return %0 : tensor<2x8x2048x8x8xf32>
  }
  func.func private @aten.view.1178(%arg0: tensor<2x8x2048x8x8xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x8x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    return %0 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.threshold_backward.1182(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x2048x8x8xi1>, tensor<16x2048x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32>
    return %5 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.native_batch_norm_backward.1192(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<16x2048x8x8xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<2048xf32>
    %3 = mhlo.multiply %2, %2 : tensor<2048xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2048xf32>
    %6 = mhlo.subtract %3, %5 : tensor<2048xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<16x2048x8x8xf32>) -> (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[2048]{0}, f32[2048]{0})"} : (tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>
    return %7 : tuple<tensor<16x2048x8x8xf32>, tensor<2048xf32>, tensor<2048xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1218(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<16x512x8x8xf32>, %arg2: tensor<2048x512x1x1xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,512,2048]{1,0,2,3}"} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,512,2048]{1,0,2,3}"} : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x2048x8x8xf32>, tensor<1x1x512x2048xf32>) -> tensor<16x512x8x8xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x8x8xf32>, tensor<16x2048x8x8xf32>) -> tensor<1x1x512x2048xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[2048,512,1,1]{0,1,3,2}"} : (tensor<1x1x512x2048xf32>) -> tensor<2048x512x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x2048x8x8xf32>, tensor<f32>) -> tensor<2048xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[2048,512,1,1]{0,1,3,2}, f32[2048]{0})"} : (tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>
    return %7 : tuple<tensor<16x512x8x8xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>>
  }
  func.func private @aten.threshold_backward.1234(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x8x8xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x8x8xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x512x8x8xi1>, tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<16x512x8x8xf32>
    return %5 : tensor<16x512x8x8xf32>
  }
  func.func private @aten.native_batch_norm_backward.1244(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<16x512x8x8xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<512xf32>
    %3 = mhlo.multiply %2, %2 : tensor<512xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %6 = mhlo.subtract %3, %5 : tensor<512xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512x8x8xf32>) -> (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1270(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<16x512x8x8xf32>, %arg2: tensor<512x512x3x3xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x8x8xf32>, tensor<3x3x512x512xf32>) -> tensor<16x512x8x8xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<3x3x512x512xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf32>) -> tensor<512x512x3x3xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x512x8x8xf32>, tensor<f32>) -> tensor<512xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,512,8,8]{3,2,1,0}, f32[512,512,3,3]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x512x8x8xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1295(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<16x2048x8x8xf32>, %arg2: tensor<512x2048x1x1xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,2048,512]{1,0,2,3}"} : (tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,2048,512]{1,0,2,3}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x8x8xf32>, tensor<1x1x2048x512xf32>) -> tensor<16x2048x8x8xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x2048x8x8xf32>, tensor<16x512x8x8xf32>) -> tensor<1x1x2048x512xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[512,2048,1,1]{0,1,3,2}"} : (tensor<1x1x2048x512xf32>) -> tensor<512x2048x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x512x8x8xf32>, tensor<f32>) -> tensor<512xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,2048,8,8]{3,2,1,0}, f32[512,2048,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>) -> tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x2048x8x8xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>>
  }
  func.func private @aten.expand.1076(%arg0: tensor<f32>) -> tensor<16x2048x8x8xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x2048x8x8xf32>
    return %3 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.mul.1311(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x2048x8x8xf32>
    return %0 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.add.1316(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<16x2048x8x8xf32>) -> tensor<16x2048x8x8xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x2048x8x8xf32>
    return %0 : tensor<16x2048x8x8xf32>
  }
  func.func private @aten.convolution_backward_overrideable.1359(%arg0: tensor<16x2048x8x8xf32>, %arg1: tensor<16x1024x16x16xf32>, %arg2: tensor<2048x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,1024,2048]{1,0,2,3}"} : (tensor<2048x1024x1x1xf32>) -> tensor<1x1x1024x2048xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,1024,2048]{1,0,2,3}"} : (tensor<1x1x1024x2048xf32>) -> tensor<1x1x1024x2048xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x2048x8x8xf32>, tensor<1x1x1024x2048xf32>) -> tensor<16x1024x16x16xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x1024x16x16xf32>, tensor<16x2048x8x8xf32>) -> tensor<1x1x1024x2048xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[2048,1024,1,1]{0,1,3,2}"} : (tensor<1x1x1024x2048xf32>) -> tensor<2048x1024x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x2048x8x8xf32>, tensor<f32>) -> tensor<2048xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[2048,1024,1,1]{0,1,3,2}, f32[2048]{0})"} : (tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>>
    return %7 : tuple<tensor<16x1024x16x16xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1397(%arg0: tensor<16x512x8x8xf32>, %arg1: tensor<16x512x16x16xf32>, %arg2: tensor<512x512x3x3xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x8x8xf32>, tensor<3x3x512x512xf32>) -> tensor<16x512x16x16xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x16x16xf32>, tensor<16x512x8x8xf32>) -> tensor<3x3x512x512xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf32>) -> tensor<512x512x3x3xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x512x8x8xf32>, tensor<f32>) -> tensor<512xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,512,16,16]{3,2,1,0}, f32[512,512,3,3]{0,1,3,2}, f32[512]{0})"} : (tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x512x16x16xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>>
  }
  func.func private @aten.threshold_backward.1413(%arg0: tensor<16x512x16x16xf32>, %arg1: tensor<16x512x16x16xf32>) -> tensor<16x512x16x16xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x16x16xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x512x16x16xf32>, tensor<16x512x16x16xf32>) -> tensor<16x512x16x16xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x16x16xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x512x16x16xi1>, tensor<16x512x16x16xf32>, tensor<16x512x16x16xf32>) -> tensor<16x512x16x16xf32>
    return %5 : tensor<16x512x16x16xf32>
  }
  func.func private @aten.native_batch_norm_backward.1423(%arg0: tensor<16x512x16x16xf32>, %arg1: tensor<16x512x16x16xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<512xf32>
    %3 = mhlo.multiply %2, %2 : tensor<512xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %6 = mhlo.subtract %3, %5 : tensor<512xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512x16x16xf32>) -> (tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,512,16,16]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1449(%arg0: tensor<16x512x16x16xf32>, %arg1: tensor<16x1024x16x16xf32>, %arg2: tensor<512x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,1024,512]{1,0,2,3}"} : (tensor<512x1024x1x1xf32>) -> tensor<1x1x1024x512xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,1024,512]{1,0,2,3}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x16x16xf32>, tensor<1x1x1024x512xf32>) -> tensor<16x1024x16x16xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x1024x16x16xf32>, tensor<16x512x16x16xf32>) -> tensor<1x1x1024x512xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[512,1024,1,1]{0,1,3,2}"} : (tensor<1x1x1024x512xf32>) -> tensor<512x1024x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x512x16x16xf32>, tensor<f32>) -> tensor<512xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[512,1024,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x1024x16x16xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>>
  }
  func.func private @aten.expand.1058(%arg0: tensor<f32>) -> tensor<16x1024x16x16xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    return %3 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.mul.1375(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x1024x16x16xf32>
    return %0 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.add.1465(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x1024x16x16xf32>
    return %0 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.threshold_backward.1470(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x1024x16x16xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x1024x16x16xi1>, tensor<16x1024x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<16x1024x16x16xf32>
    return %5 : tensor<16x1024x16x16xf32>
  }
  func.func private @aten.native_batch_norm_backward.1480(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<16x1024x16x16xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<1024xf32>
    %3 = mhlo.multiply %2, %2 : tensor<1024xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %6 = mhlo.subtract %3, %5 : tensor<1024xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<16x1024x16x16xf32>) -> (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[1024]{0}, f32[1024]{0})"} : (tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
    return %7 : tuple<tensor<16x1024x16x16xf32>, tensor<1024xf32>, tensor<1024xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1506(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<16x256x16x16xf32>, %arg2: tensor<1024x256x1x1xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x1024x16x16xf32>, tensor<1x1x256x1024xf32>) -> tensor<16x256x16x16xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x16x16xf32>, tensor<16x1024x16x16xf32>) -> tensor<1x1x256x1024xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[1024,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x1024xf32>) -> tensor<1024x256x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x1024x16x16xf32>, tensor<f32>) -> tensor<1024xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[1024,256,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>
    return %7 : tuple<tensor<16x256x16x16xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>>
  }
  func.func private @aten.threshold_backward.1522(%arg0: tensor<16x256x16x16xf32>, %arg1: tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x16x16xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x16x16xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x256x16x16xi1>, tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x256x16x16xf32>
    return %5 : tensor<16x256x16x16xf32>
  }
  func.func private @aten.native_batch_norm_backward.1532(%arg0: tensor<16x256x16x16xf32>, %arg1: tensor<16x256x16x16xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<256xf32>
    %3 = mhlo.multiply %2, %2 : tensor<256xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %6 = mhlo.subtract %3, %5 : tensor<256xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x16x16xf32>) -> (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
    return %7 : tuple<tensor<16x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1558(%arg0: tensor<16x256x16x16xf32>, %arg1: tensor<16x256x16x16xf32>, %arg2: tensor<256x256x3x3xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x16x16xf32>, tensor<3x3x256x256xf32>) -> tensor<16x256x16x16xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<3x3x256x256xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf32>) -> tensor<256x256x3x3xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x256x16x16xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,256,16,16]{3,2,1,0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
    return %7 : tuple<tensor<16x256x16x16xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1583(%arg0: tensor<16x256x16x16xf32>, %arg1: tensor<16x1024x16x16xf32>, %arg2: tensor<256x1024x1x1xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x16x16xf32>, tensor<1x1x1024x256xf32>) -> tensor<16x1024x16x16xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x1024x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<1x1x1024x256xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[256,1024,1,1]{0,1,3,2}"} : (tensor<1x1x1024x256xf32>) -> tensor<256x1024x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x256x16x16xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,1024,16,16]{3,2,1,0}, f32[256,1024,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>
    return %7 : tuple<tensor<16x1024x16x16xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1726(%arg0: tensor<16x1024x16x16xf32>, %arg1: tensor<16x512x32x32xf32>, %arg2: tensor<1024x512x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,512,1024]{1,0,2,3}"} : (tensor<1024x512x1x1xf32>) -> tensor<1x1x512x1024xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,512,1024]{1,0,2,3}"} : (tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x1024x16x16xf32>, tensor<1x1x512x1024xf32>) -> tensor<16x512x32x32xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x32x32xf32>, tensor<16x1024x16x16xf32>) -> tensor<1x1x512x1024xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[1024,512,1,1]{0,1,3,2}"} : (tensor<1x1x512x1024xf32>) -> tensor<1024x512x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x1024x16x16xf32>, tensor<f32>) -> tensor<1024xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[1024,512,1,1]{0,1,3,2}, f32[1024]{0})"} : (tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>>
    return %7 : tuple<tensor<16x512x32x32xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1764(%arg0: tensor<16x256x16x16xf32>, %arg1: tensor<16x256x32x32xf32>, %arg2: tensor<256x256x3x3xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x16x16xf32>, tensor<3x3x256x256xf32>) -> tensor<16x256x32x32xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x32x32xf32>, tensor<16x256x16x16xf32>) -> tensor<3x3x256x256xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf32>) -> tensor<256x256x3x3xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x256x16x16xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,256,32,32]{3,2,1,0}, f32[256,256,3,3]{0,1,3,2}, f32[256]{0})"} : (tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
    return %7 : tuple<tensor<16x256x32x32xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>>
  }
  func.func private @aten.threshold_backward.1780(%arg0: tensor<16x256x32x32xf32>, %arg1: tensor<16x256x32x32xf32>) -> tensor<16x256x32x32xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x32x32xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>) -> tensor<16x256x32x32xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x32x32xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x256x32x32xi1>, tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>) -> tensor<16x256x32x32xf32>
    return %5 : tensor<16x256x32x32xf32>
  }
  func.func private @aten.native_batch_norm_backward.1790(%arg0: tensor<16x256x32x32xf32>, %arg1: tensor<16x256x32x32xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<256xf32>
    %3 = mhlo.multiply %2, %2 : tensor<256xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %6 = mhlo.subtract %3, %5 : tensor<256xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x32x32xf32>) -> (tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,256,32,32]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>>
    return %7 : tuple<tensor<16x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1816(%arg0: tensor<16x256x32x32xf32>, %arg1: tensor<16x512x32x32xf32>, %arg2: tensor<256x512x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<1x1x512x256xf32>) -> tensor<1x1x512x256xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x32x32xf32>, tensor<1x1x512x256xf32>) -> tensor<16x512x32x32xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x32x32xf32>, tensor<16x256x32x32xf32>) -> tensor<1x1x512x256xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[256,512,1,1]{0,1,3,2}"} : (tensor<1x1x512x256xf32>) -> tensor<256x512x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x256x32x32xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[256,512,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>>
    return %7 : tuple<tensor<16x512x32x32xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>>
  }
  func.func private @aten.expand.1044(%arg0: tensor<f32>) -> tensor<16x512x32x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x32x32xf32>
    return %3 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.mul.1742(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x512x32x32xf32>
    return %0 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.add.1832(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x512x32x32xf32>
    return %0 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.threshold_backward.1837(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x512x32x32xi1>, tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    return %5 : tensor<16x512x32x32xf32>
  }
  func.func private @aten.native_batch_norm_backward.1847(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<16x512x32x32xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<512xf32>
    %3 = mhlo.multiply %2, %2 : tensor<512xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %6 = mhlo.subtract %3, %5 : tensor<512xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512x32x32xf32>) -> (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[512]{0}, f32[512]{0})"} : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1873(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<16x128x32x32xf32>, %arg2: tensor<512x128x1x1xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x32x32xf32>, tensor<1x1x128x512xf32>) -> tensor<16x128x32x32xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<1x1x128x512xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[512,128,1,1]{0,1,3,2}"} : (tensor<1x1x128x512xf32>) -> tensor<512x128x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x512x32x32xf32>, tensor<f32>) -> tensor<512xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[512,128,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x128x32x32xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>>
  }
  func.func private @aten.threshold_backward.1889(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x128x32x32xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x128x32x32xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x128x32x32xi1>, tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xf32>
    return %5 : tensor<16x128x32x32xf32>
  }
  func.func private @aten.native_batch_norm_backward.1899(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<16x128x32x32xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<128xf32>
    %3 = mhlo.multiply %2, %2 : tensor<128xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %6 = mhlo.subtract %3, %5 : tensor<128xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x32x32xf32>) -> (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
    return %7 : tuple<tensor<16x128x32x32xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1925(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<16x128x32x32xf32>, %arg2: tensor<128x128x3x3xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x32x32xf32>, tensor<3x3x128x128xf32>) -> tensor<16x128x32x32xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x3x128x128xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf32>) -> tensor<128x128x3x3xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x128x32x32xf32>, tensor<f32>) -> tensor<128xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,128,32,32]{3,2,1,0}, f32[128,128,3,3]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>
    return %7 : tuple<tensor<16x128x32x32xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.1950(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<16x512x32x32xf32>, %arg2: tensor<128x512x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,512,128]{1,0,2,3}"} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,512,128]{1,0,2,3}"} : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x32x32xf32>, tensor<1x1x512x128xf32>) -> tensor<16x512x32x32xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<1x1x512x128xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[128,512,1,1]{0,1,3,2}"} : (tensor<1x1x512x128xf32>) -> tensor<128x512x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x128x32x32xf32>, tensor<f32>) -> tensor<128xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,512,32,32]{3,2,1,0}, f32[128,512,1,1]{0,1,3,2}, f32[128]{0})"} : (tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>
    return %7 : tuple<tensor<16x512x32x32xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2035(%arg0: tensor<16x512x32x32xf32>, %arg1: tensor<16x256x64x64xf32>, %arg2: tensor<512x256x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x512x32x32xf32>, tensor<1x1x256x512xf32>) -> tensor<16x256x64x64xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x64x64xf32>, tensor<16x512x32x32xf32>) -> tensor<1x1x256x512xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[512,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x512xf32>) -> tensor<512x256x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x512x32x32xf32>, tensor<f32>) -> tensor<512xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[512,256,1,1]{0,1,3,2}, f32[512]{0})"} : (tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>>
    return %7 : tuple<tensor<16x256x64x64xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2073(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<16x128x64x64xf32>, %arg2: tensor<128x128x3x3xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x32x32xf32>, tensor<3x3x128x128xf32>) -> tensor<16x128x64x64xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x64x64xf32>, tensor<16x128x32x32xf32>) -> tensor<3x3x128x128xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf32>) -> tensor<128x128x3x3xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x128x32x32xf32>, tensor<f32>) -> tensor<128xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,128,64,64]{3,2,1,0}, f32[128,128,3,3]{0,1,3,2}, f32[128]{0})"} : (tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>
    return %7 : tuple<tensor<16x128x64x64xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>>
  }
  func.func private @aten.threshold_backward.2089(%arg0: tensor<16x128x64x64xf32>, %arg1: tensor<16x128x64x64xf32>) -> tensor<16x128x64x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x128x64x64xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x128x64x64xf32>, tensor<16x128x64x64xf32>) -> tensor<16x128x64x64xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x128x64x64xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x128x64x64xi1>, tensor<16x128x64x64xf32>, tensor<16x128x64x64xf32>) -> tensor<16x128x64x64xf32>
    return %5 : tensor<16x128x64x64xf32>
  }
  func.func private @aten.native_batch_norm_backward.2099(%arg0: tensor<16x128x64x64xf32>, %arg1: tensor<16x128x64x64xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<128xf32>
    %3 = mhlo.multiply %2, %2 : tensor<128xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %6 = mhlo.subtract %3, %5 : tensor<128xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x64x64xf32>) -> (tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,128,64,64]{3,2,1,0}, f32[128]{0}, f32[128]{0})"} : (tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>>
    return %7 : tuple<tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2125(%arg0: tensor<16x128x64x64xf32>, %arg1: tensor<16x256x64x64xf32>, %arg2: tensor<128x256x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x128xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x128x64x64xf32>, tensor<1x1x256x128xf32>) -> tensor<16x256x64x64xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x64x64xf32>, tensor<16x128x64x64xf32>) -> tensor<1x1x256x128xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[128,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x128xf32>) -> tensor<128x256x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x128x64x64xf32>, tensor<f32>) -> tensor<128xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[128,256,1,1]{0,1,3,2}, f32[128]{0})"} : (tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>>
    return %7 : tuple<tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>>
  }
  func.func private @aten.expand.1032(%arg0: tensor<f32>) -> tensor<16x256x64x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x64x64xf32>
    return %3 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.mul.2051(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x256x64x64xf32>
    return %0 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.add.2141(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x256x64x64xf32>
    return %0 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.threshold_backward.2146(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x256x64x64xi1>, tensor<16x256x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<16x256x64x64xf32>
    return %5 : tensor<16x256x64x64xf32>
  }
  func.func private @aten.native_batch_norm_backward.2156(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<16x256x64x64xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<256xf32>
    %3 = mhlo.multiply %2, %2 : tensor<256xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %6 = mhlo.subtract %3, %5 : tensor<256xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x64x64xf32>) -> (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[256]{0}, f32[256]{0})"} : (tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>
    return %7 : tuple<tensor<16x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2182(%arg0: tensor<16x256x64x64xf32>, %arg1: tensor<16x64x64x64xf32>, %arg2: tensor<256x64x1x1xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x64x64xf32>, tensor<1x1x64x256xf32>) -> tensor<16x64x64x64xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<16x256x64x64xf32>) -> tensor<1x1x64x256xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[256,64,1,1]{0,1,3,2}"} : (tensor<1x1x64x256xf32>) -> tensor<256x64x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x256x64x64xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[256,64,1,1]{0,1,3,2}, f32[256]{0})"} : (tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>
    return %7 : tuple<tensor<16x64x64x64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>>
  }
  func.func private @aten.threshold_backward.2198(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x64x64x64xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x64x64x64xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x64x64x64xi1>, tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
    return %5 : tensor<16x64x64x64xf32>
  }
  func.func private @aten.native_batch_norm_backward.2208(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<16x64x64x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<64xf32>
    %3 = mhlo.multiply %2, %2 : tensor<64xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %6 = mhlo.subtract %3, %5 : tensor<64xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x64x64xf32>) -> (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %7 : tuple<tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2234(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<16x64x64x64xf32>, %arg2: tensor<64x64x3x3xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<16x64x64x64xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf32>) -> tensor<64x64x3x3xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x64x64x64xf32>, tensor<f32>) -> tensor<64xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64,64,3,3]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>
    return %7 : tuple<tensor<16x64x64x64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2259(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<16x256x64x64xf32>, %arg2: tensor<64x256x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,256,64]{1,0,2,3}"} : (tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,256,64]{1,0,2,3}"} : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<1x1x256x64xf32>) -> tensor<16x256x64x64xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x256x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<1x1x256x64xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[64,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x64xf32>) -> tensor<64x256x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x64x64x64xf32>, tensor<f32>) -> tensor<64xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,256,64,64]{3,2,1,0}, f32[64,256,1,1]{0,1,3,2}, f32[64]{0})"} : (tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>
    return %7 : tuple<tensor<16x256x64x64xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2346(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<16x64x64x64xf32>, %arg2: tensor<64x64x1x1xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<1x1x64x64xf32>) -> tensor<16x64x64x64xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<16x64x64x64xf32>) -> tensor<1x1x64x64xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[64,64,1,1]{0,1,3,2}"} : (tensor<1x1x64x64xf32>) -> tensor<64x64x1x1xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x64x64x64xf32>, tensor<f32>) -> tensor<64xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,64,64,64]{3,2,1,0}, f32[64,64,1,1]{0,1,3,2}, f32[64]{0})"} : (tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>) -> tuple<tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>>
    return %7 : tuple<tensor<16x64x64x64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>>
  }
  func.func private @aten.expand.1024(%arg0: tensor<f32>) -> tensor<16x64x64x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x64x64x64xf32>
    return %3 : tensor<16x64x64x64xf32>
  }
  func.func private @aten.mul.2315(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<16x64x64x64xf32>
    return %0 : tensor<16x64x64x64xf32>
  }
  func.func private @aten.add.2362(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<16x64x64x64xf32>
    return %0 : tensor<16x64x64x64xf32>
  }
  func.func private @aten.max_pool2d_with_indices_backward.2375(%arg0: tensor<16x64x64x64xf32>, %arg1: tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.select_and_scatter"(%arg1, %arg0, %0) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = mhlo.add %arg2, %arg3 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<16x64x128x128xf32>, tensor<16x64x64x64xf32>, tensor<f32>) -> tensor<16x64x128x128xf32>
    return %1 : tensor<16x64x128x128xf32>
  }
  func.func private @aten.threshold_backward.2381(%arg0: tensor<16x64x128x128xf32>, %arg1: tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x64x128x128xf32>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<16x64x128x128xf32>, tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<16x64x128x128xf32>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<16x64x128x128xi1>, tensor<16x64x128x128xf32>, tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32>
    return %5 : tensor<16x64x128x128xf32>
  }
  func.func private @aten.native_batch_norm_backward.2391(%arg0: tensor<16x64x128x128xf32>, %arg1: tensor<16x64x128x128xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %2 = mhlo.divide %1, %arg4 : tensor<64xf32>
    %3 = mhlo.multiply %2, %2 : tensor<64xf32>
    %4 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %6 = mhlo.subtract %3, %5 : tensor<64xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %6, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x128x128xf32>) -> (tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>)
    %7 = "mhlo.tuple"(%grad_operand, %grad_scale, %grad_offset) {xla_shape = "(f32[16,64,128,128]{3,2,1,0}, f32[64]{0}, f32[64]{0})"} : (tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>>
    return %7 : tuple<tensor<16x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func.func private @aten.convolution_backward_overrideable.2417(%arg0: tensor<16x64x128x128xf32>, %arg1: tensor<16x3x256x256xf32>, %arg2: tensor<64x3x7x7xf32>) -> tuple<tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>> {
    %0 = "mhlo.transpose"(%arg2) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f32[7,7,3,64]{1,0,2,3}"} : (tensor<64x3x7x7xf32>) -> tensor<7x7x3x64xf32>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f32[7,7,3,64]{1,0,2,3}"} : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[3, 4], [3, 4]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x128x128xf32>, tensor<7x7x3x64xf32>) -> tensor<16x3x256x256xf32>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x3x256x256xf32>, tensor<16x64x128x128xf32>) -> tensor<7x7x3x64xf32>
    %4 = "mhlo.transpose"(%3) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f32[64,3,7,7]{0,1,3,2}"} : (tensor<7x7x3x64xf32>) -> tensor<64x3x7x7xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%arg0 init: %5) across dimensions = [0, 2, 3] : (tensor<16x64x128x128xf32>, tensor<f32>) -> tensor<64xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %8 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.tuple"(%2, %4, %6) {xla_shape = "(f32[16,3,256,256]{3,2,1,0}, f32[64,3,7,7]{0,1,3,2}, f32[64]{0})"} : (tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tuple<tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>>
    return %7 : tuple<tensor<16x3x256x256xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>>
  }
  func.func private @aten.sum.797(%arg0: tensor<2x19xf32>) -> tensor<1x19xf32> {
    %0 = mhlo.constant dense<2> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0] : (tensor<2x19xf32>, tensor<f32>) -> tensor<19xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.reshape"(%2) : (tensor<19xf32>) -> tensor<1x19xf32>
    return %3 : tensor<1x19xf32>
  }
  func.func private @aten.view.804(%arg0: tensor<1x19xf32>) -> tensor<19xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x19xf32>) -> tensor<19xf32>
    return %0 : tensor<19xf32>
  }
  func.func private @aten.permute.808(%arg0: tensor<2x19xf32>) -> tensor<19x2xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[19,2]{0,1}"} : (tensor<2x19xf32>) -> tensor<19x2xf32>
    return %0 : tensor<19x2xf32>
  }
  func.func private @aten.mm.812(%arg0: tensor<19x2xf32>, %arg1: tensor<2x256xf32>) -> tensor<19x256xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<19x2xf32>, tensor<2x256xf32>) -> tensor<19x256xf32>
    return %0 : tensor<19x256xf32>
  }
  func.func private @aten.permute.817(%arg0: tensor<19x256xf32>) -> tensor<256x19xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[256,19]{0,1}"} : (tensor<19x256xf32>) -> tensor<256x19xf32>
    return %0 : tensor<256x19xf32>
  }
  func.func private @aten.permute.821(%arg0: tensor<256x19xf32>) -> tensor<19x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x19xf32>) -> tensor<19x256xf32>
    return %0 : tensor<19x256xf32>
  }
  func.func private @aten.mm.828(%arg0: tensor<19x2xf32>, %arg1: tensor<2x128xf32>) -> tensor<19x128xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<19x2xf32>, tensor<2x128xf32>) -> tensor<19x128xf32>
    return %0 : tensor<19x128xf32>
  }
  func.func private @aten.permute.833(%arg0: tensor<19x128xf32>) -> tensor<128x19xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[128,19]{0,1}"} : (tensor<19x128xf32>) -> tensor<128x19xf32>
    return %0 : tensor<128x19xf32>
  }
  func.func private @aten.permute.837(%arg0: tensor<128x19xf32>) -> tensor<19x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x19xf32>) -> tensor<19x128xf32>
    return %0 : tensor<19x128xf32>
  }
  func.func private @aten.sum.845(%arg0: tensor<2x2xf32>) -> tensor<1x2xf32> {
    %0 = mhlo.constant dense<2> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.reshape"(%2) : (tensor<2xf32>) -> tensor<1x2xf32>
    return %3 : tensor<1x2xf32>
  }
  func.func private @aten.view.852(%arg0: tensor<1x2xf32>) -> tensor<2xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
  func.func private @aten.permute.856(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[2,2]{0,1}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @aten.mm.860(%arg0: tensor<2x2xf32>, %arg1: tensor<2x256xf32>) -> tensor<2x256xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x2xf32>, tensor<2x256xf32>) -> tensor<2x256xf32>
    return %0 : tensor<2x256xf32>
  }
  func.func private @aten.permute.865(%arg0: tensor<2x256xf32>) -> tensor<256x2xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[256,2]{0,1}"} : (tensor<2x256xf32>) -> tensor<256x2xf32>
    return %0 : tensor<256x2xf32>
  }
  func.func private @aten.permute.869(%arg0: tensor<256x2xf32>) -> tensor<2x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x2xf32>) -> tensor<2x256xf32>
    return %0 : tensor<2x256xf32>
  }
  func.func private @aten.sum.902(%arg0: tensor<2x256xf32>) -> tensor<1x256xf32> {
    %0 = mhlo.constant dense<2> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0] : (tensor<2x256xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.reshape"(%2) : (tensor<256xf32>) -> tensor<1x256xf32>
    return %3 : tensor<1x256xf32>
  }
  func.func private @aten.view.909(%arg0: tensor<1x256xf32>) -> tensor<256xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  func.func private @aten.mm.914(%arg0: tensor<256x2xf32>, %arg1: tensor<2x2816xf32>) -> tensor<256x2816xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x2xf32>, tensor<2x2816xf32>) -> tensor<256x2816xf32>
    return %0 : tensor<256x2816xf32>
  }
  func.func private @aten.permute.919(%arg0: tensor<256x2816xf32>) -> tensor<2816x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[2816,256]{0,1}"} : (tensor<256x2816xf32>) -> tensor<2816x256xf32>
    return %0 : tensor<2816x256xf32>
  }
  func.func private @aten.permute.923(%arg0: tensor<2816x256xf32>) -> tensor<256x2816xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2816x256xf32>) -> tensor<256x2816xf32>
    return %0 : tensor<256x2816xf32>
  }
  func.func private @aten.sum.950(%arg0: tensor<2x128xf32>) -> tensor<1x128xf32> {
    %0 = mhlo.constant dense<2> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0] : (tensor<2x128xf32>, tensor<f32>) -> tensor<128xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.reshape"(%2) : (tensor<128xf32>) -> tensor<1x128xf32>
    return %3 : tensor<1x128xf32>
  }
  func.func private @aten.view.957(%arg0: tensor<1x128xf32>) -> tensor<128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  func.func private @aten.permute.961(%arg0: tensor<2x128xf32>) -> tensor<128x2xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[128,2]{0,1}"} : (tensor<2x128xf32>) -> tensor<128x2xf32>
    return %0 : tensor<128x2xf32>
  }
  func.func private @aten.mm.965(%arg0: tensor<128x2xf32>, %arg1: tensor<2x2816xf32>) -> tensor<128x2816xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<128x2xf32>, tensor<2x2816xf32>) -> tensor<128x2816xf32>
    return %0 : tensor<128x2816xf32>
  }
  func.func private @aten.permute.970(%arg0: tensor<128x2816xf32>) -> tensor<2816x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[2816,128]{0,1}"} : (tensor<128x2816xf32>) -> tensor<2816x128xf32>
    return %0 : tensor<2816x128xf32>
  }
  func.func private @aten.permute.974(%arg0: tensor<2816x128xf32>) -> tensor<128x2816xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2816x128xf32>) -> tensor<128x2816xf32>
    return %0 : tensor<128x2816xf32>
  }
  func.func private @aten.mm.991(%arg0: tensor<256x2xf32>, %arg1: tensor<2x2048xf32>) -> tensor<256x2048xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x2xf32>, tensor<2x2048xf32>) -> tensor<256x2048xf32>
    return %0 : tensor<256x2048xf32>
  }
  func.func private @aten.permute.996(%arg0: tensor<256x2048xf32>) -> tensor<2048x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[2048,256]{0,1}"} : (tensor<256x2048xf32>) -> tensor<2048x256xf32>
    return %0 : tensor<2048x256xf32>
  }
  func.func private @aten.permute.1000(%arg0: tensor<2048x256xf32>) -> tensor<256x2048xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2048x256xf32>) -> tensor<256x2048xf32>
    return %0 : tensor<256x2048xf32>
  }
  func.func private @aten.mm.1010(%arg0: tensor<256x2xf32>, %arg1: tensor<2x768xf32>) -> tensor<256x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x2xf32>, tensor<2x768xf32>) -> tensor<256x768xf32>
    return %0 : tensor<256x768xf32>
  }
  func.func private @aten.permute.1015(%arg0: tensor<256x768xf32>) -> tensor<768x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[768,256]{0,1}"} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    return %0 : tensor<768x256xf32>
  }
  func.func private @aten.permute.1019(%arg0: tensor<768x256xf32>) -> tensor<256x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<768x256xf32>) -> tensor<256x768xf32>
    return %0 : tensor<256x768xf32>
  }
  func.func private @aten.expand.2522(%arg0: tensor<f32>) -> tensor<512x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512x768xf32>
    return %3 : tensor<512x768xf32>
  }
  func.func private @aten.view.2463(%arg0: tensor<1x240xi64>) -> tensor<240xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x240xi64>) -> tensor<240xi64>
    return %0 : tensor<240xi64>
  }
  func.func private @aten.lt.2504(%arg0: tensor<240xi64>, %arg1: tensor<i64>) -> tensor<240xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<240xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<240xi64>, tensor<240xi64>) -> tensor<240xi1>
    return %1 : tensor<240xi1>
  }
  func.func private @aten.expand.2491(%arg0: tensor<i64>) -> tensor<240xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<i64>) -> tensor<1xi64>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi64>) -> tensor<i64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<240xi64>
    return %3 : tensor<240xi64>
  }
  func.func private @aten.add.2498(%arg0: tensor<240xi64>, %arg1: tensor<240xi64>) -> tensor<240xi64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<240xi64>
    return %0 : tensor<240xi64>
  }
  func.func private @aten.where.2510(%arg0: tensor<240xi1>, %arg1: tensor<240xi64>, %arg2: tensor<240xi64>) -> tensor<240xi64> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<240xi1>, tensor<240xi64>, tensor<240xi64>) -> tensor<240xi64>
    return %0 : tensor<240xi64>
  }
  func.func private @aten.stack.2516(%arg0: tensor<240xi64>) -> tensor<240x1xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<240xi64>) -> tensor<240x1xi64>
    %1 = "mhlo.concatenate"(%0) {dimension = 1 : i64} : (tensor<240x1xi64>) -> tensor<240x1xi64>
    return %1 : tensor<240x1xi64>
  }
  func.func private @aten.ne.2467(%arg0: tensor<240xi64>, %arg1: tensor<f64>) -> tensor<240xi1> {
    %0 = "mhlo.convert"(%arg0) : (tensor<240xi64>) -> tensor<240xf64>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<240xf64>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<240xf64>, tensor<240xf64>) -> tensor<240xi1>
    return %2 : tensor<240xi1>
  }
  func.func private @aten.view.2474(%arg0: tensor<240xi1>) -> tensor<240x1xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<240xi1>) -> tensor<240x1xi1>
    return %0 : tensor<240x1xi1>
  }
  func.func private @aten.expand.2478(%arg0: tensor<240x1xi1>) -> tensor<240x768xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<240x1xi1>) -> tensor<240x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<240x1xi1>) -> tensor<240xi1>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<240xi1>) -> tensor<240x768xi1>
    return %2 : tensor<240x768xi1>
  }
  func.func private @aten.sum.2451(%arg0: tensor<2x240x768xf32>) -> tensor<1x240x768xf32> {
    %0 = mhlo.constant dense<2> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0] : (tensor<2x240x768xf32>, tensor<f32>) -> tensor<240x768xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.reshape"(%2) : (tensor<240x768xf32>) -> tensor<1x240x768xf32>
    return %3 : tensor<1x240x768xf32>
  }
  func.func private @aten.view.2458(%arg0: tensor<1x240x768xf32>) -> tensor<240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x240x768xf32>) -> tensor<240x768xf32>
    return %0 : tensor<240x768xf32>
  }
  func.func private @aten.expand.2438(%arg0: tensor<f32>) -> tensor<240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<240x768xf32>
    return %3 : tensor<240x768xf32>
  }
  func.func private @aten.where.2484(%arg0: tensor<240x768xi1>, %arg1: tensor<240x768xf32>, %arg2: tensor<240x768xf32>) -> tensor<240x768xf32> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<240x768xi1>, tensor<240x768xf32>, tensor<240x768xf32>) -> tensor<240x768xf32>
    return %0 : tensor<240x768xf32>
  }
  func.func private @aten.index_put.2533(%arg0: tensor<512x768xf32>, %arg1: tensor<240x1xi64>, %arg2: tensor<240x768xf32>) -> tensor<512x768xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<240x768xf32>) -> tensor<240x768xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x768xf32>, tensor<240x1xi64>, tensor<240x768xf32>) -> tensor<512x768xf32>
    return %1 : tensor<512x768xf32>
  }
  func.func private @aten.permute.2540(%arg0: tensor<512x768xf32>) -> tensor<512x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<512x768xf32>) -> tensor<512x768xf32>
    return %0 : tensor<512x768xf32>
  }
  func.func private @aten.expand.2616(%arg0: tensor<f32>) -> tensor<2x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x768xf32>
    return %3 : tensor<2x768xf32>
  }
  func.func private @aten.view.2557(%arg0: tensor<2x240xi64>) -> tensor<480xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240xi64>) -> tensor<480xi64>
    return %0 : tensor<480xi64>
  }
  func.func private @aten.lt.2598(%arg0: tensor<480xi64>, %arg1: tensor<i64>) -> tensor<480xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<480xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<480xi64>, tensor<480xi64>) -> tensor<480xi1>
    return %1 : tensor<480xi1>
  }
  func.func private @aten.expand.2585(%arg0: tensor<i64>) -> tensor<480xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<i64>) -> tensor<1xi64>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi64>) -> tensor<i64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<480xi64>
    return %3 : tensor<480xi64>
  }
  func.func private @aten.add.2592(%arg0: tensor<480xi64>, %arg1: tensor<480xi64>) -> tensor<480xi64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<480xi64>
    return %0 : tensor<480xi64>
  }
  func.func private @aten.where.2604(%arg0: tensor<480xi1>, %arg1: tensor<480xi64>, %arg2: tensor<480xi64>) -> tensor<480xi64> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<480xi1>, tensor<480xi64>, tensor<480xi64>) -> tensor<480xi64>
    return %0 : tensor<480xi64>
  }
  func.func private @aten.stack.2610(%arg0: tensor<480xi64>) -> tensor<480x1xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<480xi64>) -> tensor<480x1xi64>
    %1 = "mhlo.concatenate"(%0) {dimension = 1 : i64} : (tensor<480x1xi64>) -> tensor<480x1xi64>
    return %1 : tensor<480x1xi64>
  }
  func.func private @aten.ne.2561(%arg0: tensor<480xi64>, %arg1: tensor<f64>) -> tensor<480xi1> {
    %0 = "mhlo.convert"(%arg0) : (tensor<480xi64>) -> tensor<480xf64>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<480xf64>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<480xf64>, tensor<480xf64>) -> tensor<480xi1>
    return %2 : tensor<480xi1>
  }
  func.func private @aten.view.2568(%arg0: tensor<480xi1>) -> tensor<480x1xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<480xi1>) -> tensor<480x1xi1>
    return %0 : tensor<480x1xi1>
  }
  func.func private @aten.expand.2572(%arg0: tensor<480x1xi1>) -> tensor<480x768xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<480x1xi1>) -> tensor<480x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<480x1xi1>) -> tensor<480xi1>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<480xi1>) -> tensor<480x768xi1>
    return %2 : tensor<480x768xi1>
  }
  func.func private @aten.view.2552(%arg0: tensor<2x240x768xf32>) -> tensor<480x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x768xf32>) -> tensor<480x768xf32>
    return %0 : tensor<480x768xf32>
  }
  func.func private @aten.expand.2545(%arg0: tensor<f32>) -> tensor<480x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<480x768xf32>
    return %3 : tensor<480x768xf32>
  }
  func.func private @aten.where.2578(%arg0: tensor<480x768xi1>, %arg1: tensor<480x768xf32>, %arg2: tensor<480x768xf32>) -> tensor<480x768xf32> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<480x768xi1>, tensor<480x768xf32>, tensor<480x768xf32>) -> tensor<480x768xf32>
    return %0 : tensor<480x768xf32>
  }
  func.func private @aten.index_put.2627(%arg0: tensor<2x768xf32>, %arg1: tensor<480x1xi64>, %arg2: tensor<480x768xf32>) -> tensor<2x768xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<480x768xf32>) -> tensor<480x768xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x768xf32>, tensor<480x1xi64>, tensor<480x768xf32>) -> tensor<2x768xf32>
    return %1 : tensor<2x768xf32>
  }
  func.func private @aten.permute.2634(%arg0: tensor<2x768xf32>) -> tensor<2x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x768xf32>) -> tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
  func.func private @aten.expand.2655(%arg0: tensor<f32>) -> tensor<21128x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<21128x768xf32>
    return %3 : tensor<21128x768xf32>
  }
  func.func private @aten.index_put.2666(%arg0: tensor<21128x768xf32>, %arg1: tensor<480x1xi64>, %arg2: tensor<480x768xf32>) -> tensor<21128x768xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<480x768xf32>) -> tensor<480x768xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<21128x768xf32>, tensor<480x1xi64>, tensor<480x768xf32>) -> tensor<21128x768xf32>
    return %1 : tensor<21128x768xf32>
  }
  func.func private @aten.permute.2673(%arg0: tensor<21128x768xf32>) -> tensor<21128x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<21128x768xf32>) -> tensor<21128x768xf32>
    return %0 : tensor<21128x768xf32>
  }
  func.func private @aten.sum.2688(%arg0: tensor<480x768xf32>) -> tensor<1x768xf32> {
    %0 = mhlo.constant dense<480> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0] : (tensor<480x768xf32>, tensor<f32>) -> tensor<768xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.reshape"(%2) : (tensor<768xf32>) -> tensor<1x768xf32>
    return %3 : tensor<1x768xf32>
  }
  func.func private @aten.view.2695(%arg0: tensor<1x768xf32>) -> tensor<768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x768xf32>) -> tensor<768xf32>
    return %0 : tensor<768xf32>
  }
  func.func private @aten.permute.2700(%arg0: tensor<480x768xf32>) -> tensor<768x480xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[768,480]{0,1}"} : (tensor<480x768xf32>) -> tensor<768x480xf32>
    return %0 : tensor<768x480xf32>
  }
  func.func private @aten.mm.2704(%arg0: tensor<768x480xf32>, %arg1: tensor<480x768xf32>) -> tensor<768x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<768x480xf32>, tensor<480x768xf32>) -> tensor<768x768xf32>
    return %0 : tensor<768x768xf32>
  }
  func.func private @aten.permute.2709(%arg0: tensor<768x768xf32>) -> tensor<768x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    return %0 : tensor<768x768xf32>
  }
  func.func private @aten.permute.2713(%arg0: tensor<768x768xf32>) -> tensor<768x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    return %0 : tensor<768x768xf32>
  }
  func.func private @aten.permute.2718(%arg0: tensor<24x240x64xf32>) -> tensor<24x64x240xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, xla_shape = "f32[24,64,240]{1,2,0}"} : (tensor<24x240x64xf32>) -> tensor<24x64x240xf32>
    return %0 : tensor<24x64x240xf32>
  }
  func.func private @aten.mm.2723(%arg0: tensor<480x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<480x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<480x768xf32>, tensor<768x768xf32>) -> tensor<480x768xf32>
    return %0 : tensor<480x768xf32>
  }
  func.func private @aten.view.2728(%arg0: tensor<480x768xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<480x768xf32>) -> tensor<2x240x768xf32>
    return %0 : tensor<2x240x768xf32>
  }
  func.func private @aten.view.2732(%arg0: tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x768xf32>) -> tensor<2x240x12x64xf32>
    return %0 : tensor<2x240x12x64xf32>
  }
  func.func private @aten.permute.2736(%arg0: tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>, xla_shape = "f32[2,12,240,64]{3,1,2,0}"} : (tensor<2x240x12x64xf32>) -> tensor<2x12x240x64xf32>
    return %0 : tensor<2x12x240x64xf32>
  }
  func.func private @aten.view.2740(%arg0: tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x12x240x64xf32>) -> tensor<24x240x64xf32>
    return %0 : tensor<24x240x64xf32>
  }
  func.func private @aten.matmul.2744(%arg0: tensor<24x240x64xf32>, %arg1: tensor<24x64x240xf32>) -> tensor<24x240x240xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<24x240x64xf32>, tensor<24x64x240xf32>) -> tensor<24x240x240xf32>
    return %0 : tensor<24x240x240xf32>
  }
  func.func private @aten.view.2749(%arg0: tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<24x240x240xf32>) -> tensor<2x12x240x240xf32>
    return %0 : tensor<2x12x240x240xf32>
  }
  func.func private @aten._softmax_backward_data.2757(%arg0: tensor<2x12x240x240xf32>, %arg1: tensor<2x12x240x240xf32>) -> tensor<2x12x240x240xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x12x240x240xf32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%0 init: %1) across dimensions = [3] : (tensor<2x12x240x240xf32>, tensor<f32>) -> tensor<2x12x240xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %6 = mhlo.add %arg2, %arg3 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x12x240xf32>) -> tensor<2x12x240x240xf32>
    %4 = mhlo.subtract %arg0, %3 : tensor<2x12x240x240xf32>
    %5 = mhlo.multiply %arg1, %4 : tensor<2x12x240x240xf32>
    return %5 : tensor<2x12x240x240xf32>
  }
  func.func private @aten.div.2767(%arg0: tensor<2x12x240x240xf32>, %arg1: tensor<f32>) -> tensor<2x12x240x240xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x12x240x240xf32>
    %1 = mhlo.divide %arg0, %0 : tensor<2x12x240x240xf32>
    return %1 : tensor<2x12x240x240xf32>
  }
  func.func private @aten.view.2773(%arg0: tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x12x240x240xf32>) -> tensor<24x240x240xf32>
    return %0 : tensor<24x240x240xf32>
  }
  func.func private @aten.matmul.2778(%arg0: tensor<24x64x240xf32>, %arg1: tensor<24x240x240xf32>) -> tensor<24x64x240xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<24x64x240xf32>, tensor<24x240x240xf32>) -> tensor<24x64x240xf32>
    return %0 : tensor<24x64x240xf32>
  }
  func.func private @aten.view.2783(%arg0: tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<24x64x240xf32>) -> tensor<2x12x64x240xf32>
    return %0 : tensor<2x12x64x240xf32>
  }
  func.func private @aten.permute.2787(%arg0: tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>, xla_shape = "f32[2,12,240,64]{2,3,1,0}"} : (tensor<2x12x64x240xf32>) -> tensor<2x12x240x64xf32>
    return %0 : tensor<2x12x240x64xf32>
  }
  func.func private @aten.permute.2791(%arg0: tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>, xla_shape = "f32[2,240,12,64]{1,3,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    return %0 : tensor<2x240x12x64xf32>
  }
  func.func private @aten.view.2795(%arg0: tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    return %0 : tensor<2x240x768xf32>
  }
  func.func private @aten.permute.2807(%arg0: tensor<24x64x240xf32>) -> tensor<24x240x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, xla_shape = "f32[24,240,64]{1,2,0}"} : (tensor<24x64x240xf32>) -> tensor<24x240x64xf32>
    return %0 : tensor<24x240x64xf32>
  }
  func.func private @aten.matmul.2811(%arg0: tensor<24x240x240xf32>, %arg1: tensor<24x240x64xf32>) -> tensor<24x240x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    return %0 : tensor<24x240x64xf32>
  }
  func.func private @aten.view.2816(%arg0: tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<24x240x64xf32>) -> tensor<2x12x240x64xf32>
    return %0 : tensor<2x12x240x64xf32>
  }
  func.func private @aten.permute.2820(%arg0: tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>, xla_shape = "f32[2,240,12,64]{3,1,2,0}"} : (tensor<2x12x240x64xf32>) -> tensor<2x240x12x64xf32>
    return %0 : tensor<2x240x12x64xf32>
  }
  func.func private @aten.view.2824(%arg0: tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x12x64xf32>) -> tensor<2x240x768xf32>
    return %0 : tensor<2x240x768xf32>
  }
  func.func private @aten.permute.2837(%arg0: tensor<24x240x240xf32>) -> tensor<24x240x240xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, xla_shape = "f32[24,240,240]{1,2,0}"} : (tensor<24x240x240xf32>) -> tensor<24x240x240xf32>
    return %0 : tensor<24x240x240xf32>
  }
  func.func private @aten.matmul.2841(%arg0: tensor<24x240x240xf32>, %arg1: tensor<24x240x64xf32>) -> tensor<24x240x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<24x240x240xf32>, tensor<24x240x64xf32>) -> tensor<24x240x64xf32>
    return %0 : tensor<24x240x64xf32>
  }
  func.func private @aten.permute.2858(%arg0: tensor<3072x768xf32>) -> tensor<768x3072xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[768,3072]{0,1}"} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    return %0 : tensor<768x3072xf32>
  }
  func.func private @aten.mm.2865(%arg0: tensor<480x768xf32>, %arg1: tensor<768x3072xf32>) -> tensor<480x3072xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<480x768xf32>, tensor<768x3072xf32>) -> tensor<480x3072xf32>
    return %0 : tensor<480x3072xf32>
  }
  func.func private @aten.view.2870(%arg0: tensor<480x3072xf32>) -> tensor<2x240x3072xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<480x3072xf32>) -> tensor<2x240x3072xf32>
    return %0 : tensor<2x240x3072xf32>
  }
  func.func private @aten.gelu_backward.2874(%arg0: tensor<2x240x3072xf32>, %arg1: tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32> {
    %0 = mhlo.constant dense<5.000000e-01> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %2 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %4 = mhlo.constant dense<-4.000000e+00> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %6 = mhlo.constant dense<0.707106769> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %8 = mhlo.multiply %arg1, %7 : tensor<2x240x3072xf32>
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
    %67 = mhlo.add %3, %66 : tensor<2x240x3072xf32>
    %68 = mhlo.multiply %1, %67 : tensor<2x240x3072xf32>
    %69 = mhlo.multiply %arg1, %arg1 : tensor<2x240x3072xf32>
    %70 = "mhlo.negate"(%0) : (tensor<f32>) -> tensor<f32>
    %71 = "mhlo.broadcast_in_dim"(%70) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %72 = mhlo.multiply %69, %71 : tensor<2x240x3072xf32>
    %73 = "mhlo.exponential"(%72) : (tensor<2x240x3072xf32>) -> tensor<2x240x3072xf32>
    %74 = mhlo.multiply %arg1, %73 : tensor<2x240x3072xf32>
    %75 = mhlo.constant dense<1.12837923> : tensor<f32>
    %76 = mhlo.multiply %75, %6 : tensor<f32>
    %77 = mhlo.multiply %76, %0 : tensor<f32>
    %78 = "mhlo.broadcast_in_dim"(%77) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x240x3072xf32>
    %79 = mhlo.multiply %74, %78 : tensor<2x240x3072xf32>
    %80 = mhlo.add %68, %79 : tensor<2x240x3072xf32>
    %81 = mhlo.multiply %arg0, %80 : tensor<2x240x3072xf32>
    return %81 : tensor<2x240x3072xf32>
  }
  func.func private @aten.view.2960(%arg0: tensor<2x240x3072xf32>) -> tensor<480x3072xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x240x3072xf32>) -> tensor<480x3072xf32>
    return %0 : tensor<480x3072xf32>
  }
  func.func private @aten.sum.2968(%arg0: tensor<480x3072xf32>) -> tensor<1x3072xf32> {
    %0 = mhlo.constant dense<480> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0] : (tensor<480x3072xf32>, tensor<f32>) -> tensor<3072xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.reshape"(%2) : (tensor<3072xf32>) -> tensor<1x3072xf32>
    return %3 : tensor<1x3072xf32>
  }
  func.func private @aten.view.2975(%arg0: tensor<1x3072xf32>) -> tensor<3072xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x3072xf32>) -> tensor<3072xf32>
    return %0 : tensor<3072xf32>
  }
  func.func private @aten.permute.2980(%arg0: tensor<480x3072xf32>) -> tensor<3072x480xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[3072,480]{0,1}"} : (tensor<480x3072xf32>) -> tensor<3072x480xf32>
    return %0 : tensor<3072x480xf32>
  }
  func.func private @aten.mm.2984(%arg0: tensor<3072x480xf32>, %arg1: tensor<480x768xf32>) -> tensor<3072x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3072x480xf32>, tensor<480x768xf32>) -> tensor<3072x768xf32>
    return %0 : tensor<3072x768xf32>
  }
  func.func private @aten.permute.2990(%arg0: tensor<768x3072xf32>) -> tensor<3072x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    return %0 : tensor<3072x768xf32>
  }
  func.func private @aten.mm.3002(%arg0: tensor<768x480xf32>, %arg1: tensor<480x3072xf32>) -> tensor<768x3072xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<768x480xf32>, tensor<480x3072xf32>) -> tensor<768x3072xf32>
    return %0 : tensor<768x3072xf32>
  }
  func.func private @aten.permute.3007(%arg0: tensor<768x3072xf32>) -> tensor<3072x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[3072,768]{0,1}"} : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    return %0 : tensor<3072x768xf32>
  }
  func.func private @aten.permute.3011(%arg0: tensor<3072x768xf32>) -> tensor<768x3072xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    return %0 : tensor<768x3072xf32>
  }
  func.func private @aten.permute.4085(%arg0: tensor<768x256xf32>) -> tensor<256x768xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
    return %0 : tensor<256x768xf32>
  }
  func.func private @aten.mm.4089(%arg0: tensor<2x256xf32>, %arg1: tensor<256x768xf32>) -> tensor<2x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<2x256xf32>, tensor<256x768xf32>) -> tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
  func.func private @aten.mul.4069(%arg0: tensor<2x768xf32>, %arg1: tensor<2x768xf32>) -> tensor<2x768xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
  func.func private @aten.add.4094(%arg0: tensor<2x768xf32>, %arg1: tensor<2x768xf32>) -> tensor<2x768xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
  func.func private @aten.pow.4062(%arg0: tensor<2x768xf32>, %arg1: tensor<2x768xf32>) -> tensor<2x768xf32> {
    %0 = mhlo.power %arg0, %arg1 : tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
  func.func private @aten.sub.4076(%arg0: tensor<2x768xf32>, %arg1: tensor<2x768xf32>) -> tensor<2x768xf32> {
    %0 = mhlo.subtract %arg0, %arg1 : tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
  func.func private @aten.sum.4104(%arg0: tensor<2x768xf32>) -> tensor<1x768xf32> {
    %0 = mhlo.constant dense<2> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0] : (tensor<2x768xf32>, tensor<f32>) -> tensor<768xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }
    %3 = "mhlo.reshape"(%2) : (tensor<768xf32>) -> tensor<1x768xf32>
    return %3 : tensor<1x768xf32>
  }
  func.func private @aten.permute.4112(%arg0: tensor<2x768xf32>) -> tensor<768x2xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[768,2]{0,1}"} : (tensor<2x768xf32>) -> tensor<768x2xf32>
    return %0 : tensor<768x2xf32>
  }
  func.func private @aten.mm.4116(%arg0: tensor<768x2xf32>, %arg1: tensor<2x768xf32>) -> tensor<768x768xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<768x2xf32>, tensor<2x768xf32>) -> tensor<768x768xf32>
    return %0 : tensor<768x768xf32>
  }
}

