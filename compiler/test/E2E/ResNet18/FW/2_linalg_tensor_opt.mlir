// RUN: byteir-opt %s -linalg-tensor-opt | FileCheck %s

// CHECK-LABEL: func.func @main

module {
  func.func private @Unknown0(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16>
    return %0 : tensor<1x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    return %0 : tensor<64x3x7x7xf16>
  }
  func.func private @BatchNormTrainingOp2(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf16>
    return %1, %batch_mean, %batch_var : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @Unknown3(%arg0: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x64x112x112xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<1x64x112x112xf16>
    return %1 : tensor<1x64x112x112xf16>
  }
  func.func private @Unknown4(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    return %0 : tensor<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp5(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %1, %batch_mean, %batch_var : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @Unknown6(%arg0: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x64x56x56xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown9(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x64x56x56xf16>
    %2 = mhlo.maximum %1, %0 : tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown16(%arg0: tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    return %0 : tensor<128x64x1x1xf16>
  }
  func.func private @BatchNormTrainingOp17(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %1, %batch_mean, %batch_var : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @Unknown18(%arg0: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    return %0 : tensor<128x64x3x3xf16>
  }
  func.func private @Unknown20(%arg0: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x128x28x28xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown21(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    return %0 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown23(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x128x28x28xf16>
    %2 = mhlo.maximum %1, %0 : tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown30(%arg0: tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    return %0 : tensor<256x128x1x1xf16>
  }
  func.func private @BatchNormTrainingOp31(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %1, %batch_mean, %batch_var : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @Unknown32(%arg0: tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    return %0 : tensor<256x128x3x3xf16>
  }
  func.func private @Unknown34(%arg0: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x256x14x14xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown35(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    return %0 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown37(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x256x14x14xf16>
    %2 = mhlo.maximum %1, %0 : tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown44(%arg0: tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    return %0 : tensor<512x256x1x1xf16>
  }
  func.func private @BatchNormTrainingOp45(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %1, %batch_mean, %batch_var : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @Unknown46(%arg0: tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    return %0 : tensor<512x256x3x3xf16>
  }
  func.func private @Unknown48(%arg0: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x512x7x7xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown49(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    return %0 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown51(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x512x7x7xf16>
    %2 = mhlo.maximum %1, %0 : tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown58(%arg0: tensor<1x512x7x7xf16>) -> tensor<1x512xf16> attributes {__byteir_reduction_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [3, 2] : (tensor<1x512x7x7xf16>, tensor<f16>) -> tensor<1x512xf16>
     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
      %2 = mhlo.add %arg1, %arg2 : tensor<f16>
      mhlo.return %2 : tensor<f16>
    }
    return %1 : tensor<1x512xf16>
  }
  func.func private @Unknown59(%arg0: tensor<1x512xf16>) -> tensor<1x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<2.040100e-02> : tensor<1x512xf16>
    %1 = mhlo.multiply %arg0, %0 : tensor<1x512xf16>
    return %1 : tensor<1x512xf16>
  }
  func.func private @Unknown60(%arg0: tensor<1000x512xf32>) -> tensor<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func.func private @Unknown61(%arg0: tensor<1000xf32>, %arg1: tensor<1x1000xf16>) -> tensor<1x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<1000xf32>) -> tensor<1000xf16>
    %1 = mhlo.reshape %0 : (tensor<1000xf16>) -> tensor<1x1000xf16>
    %2 = mhlo.add %arg1, %1 : tensor<1x1000xf16>
    return %2 : tensor<1x1000xf16>
  }
  func.func private @Unknown62(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @Unknown72(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func.func private @Unknown82(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func.func private @Unknown92(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<1000xf32>, %arg4: tensor<1000x512xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64x64x3x3xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64x64x3x3xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x64x3x3xf32>, %arg22: tensor<128x128x3x3xf32>, %arg23: tensor<128x64x1x1xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128x128x3x3xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x128x3x3xf32>, %arg37: tensor<256x256x3x3xf32>, %arg38: tensor<256x128x1x1xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<256xf32>, %arg44: tensor<256xf32>, %arg45: tensor<256x256x3x3xf32>, %arg46: tensor<256x256x3x3xf32>, %arg47: tensor<512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<512x256x3x3xf32>, %arg52: tensor<512x512x3x3xf32>, %arg53: tensor<512x256x1x1xf32>, %arg54: tensor<512xf32>, %arg55: tensor<512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<512xf32>, %arg59: tensor<512xf32>, %arg60: tensor<512x512x3x3xf32>, %arg61: tensor<512x512x3x3xf32>, %arg62: tensor<i64>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<i64>, %arg66: tensor<64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<i64>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<i64>, %arg72: tensor<64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<i64>, %arg75: tensor<64xf32>, %arg76: tensor<64xf32>, %arg77: tensor<i64>, %arg78: tensor<128xf32>, %arg79: tensor<128xf32>, %arg80: tensor<i64>, %arg81: tensor<128xf32>, %arg82: tensor<128xf32>, %arg83: tensor<i64>, %arg84: tensor<128xf32>, %arg85: tensor<128xf32>, %arg86: tensor<i64>, %arg87: tensor<128xf32>, %arg88: tensor<128xf32>, %arg89: tensor<i64>, %arg90: tensor<128xf32>, %arg91: tensor<128xf32>, %arg92: tensor<i64>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<i64>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<i64>, %arg99: tensor<256xf32>, %arg100: tensor<256xf32>, %arg101: tensor<i64>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<i64>, %arg105: tensor<256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<i64>, %arg108: tensor<512xf32>, %arg109: tensor<512xf32>, %arg110: tensor<i64>, %arg111: tensor<512xf32>, %arg112: tensor<512xf32>, %arg113: tensor<i64>, %arg114: tensor<512xf32>, %arg115: tensor<512xf32>, %arg116: tensor<i64>, %arg117: tensor<512xf32>, %arg118: tensor<512xf32>, %arg119: tensor<i64>, %arg120: tensor<512xf32>, %arg121: tensor<512xf32>, %arg122: tensor<1x3x224x224xf32>) -> (tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>) {
    %0 = mhlo.constant dense<0xFC00> : tensor<f16>
    %1 = call @Unknown0(%arg122) : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16>
    %2 = call @Unknown1(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %3 = mhlo.convolution(%1, %2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<1x64x112x112xf16>
    %4:3 = call @BatchNormTrainingOp2(%3, %arg1, %arg0) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %5 = call @Unknown3(%4#0) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %6 = "mhlo.reduce_window"(%5, %0) ({
    ^bb0(%arg123: tensor<f16>, %arg124: tensor<f16>):
      %126 = mhlo.maximum %arg123, %arg124 : tensor<f16>
      mhlo.return %126 : tensor<f16>
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x64x112x112xf16>, tensor<f16>) -> tensor<1x64x56x56xf16>
    %7 = call @Unknown4(%arg9) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %8 = mhlo.convolution(%6, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %9:3 = call @BatchNormTrainingOp5(%8, %arg6, %arg5) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %10 = call @Unknown6(%9#0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %11 = call @Unknown4(%arg10) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %12 = mhlo.convolution(%10, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %13:3 = call @BatchNormTrainingOp5(%12, %arg8, %arg7) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %14 = call @Unknown9(%13#0, %6) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %15 = call @Unknown4(%arg15) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %16 = mhlo.convolution(%14, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %17:3 = call @BatchNormTrainingOp5(%16, %arg12, %arg11) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %18 = call @Unknown6(%17#0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %19 = call @Unknown4(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %20 = mhlo.convolution(%18, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %21:3 = call @BatchNormTrainingOp5(%20, %arg14, %arg13) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %22 = call @Unknown9(%21#0, %14) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %23 = call @Unknown16(%arg23) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %24 = mhlo.convolution(%22, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<1x128x28x28xf16>
    %25:3 = call @BatchNormTrainingOp17(%24, %arg25, %arg24) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %26 = call @Unknown18(%arg21) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %27 = mhlo.convolution(%22, %26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<1x128x28x28xf16>
    %28:3 = call @BatchNormTrainingOp17(%27, %arg18, %arg17) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %29 = call @Unknown20(%28#0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %30 = call @Unknown21(%arg22) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %31 = mhlo.convolution(%29, %30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %32:3 = call @BatchNormTrainingOp17(%31, %arg20, %arg19) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %33 = call @Unknown23(%32#0, %25#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %34 = call @Unknown21(%arg30) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %35 = mhlo.convolution(%33, %34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %36:3 = call @BatchNormTrainingOp17(%35, %arg27, %arg26) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %37 = call @Unknown20(%36#0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %38 = call @Unknown21(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %39 = mhlo.convolution(%37, %38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %40:3 = call @BatchNormTrainingOp17(%39, %arg29, %arg28) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %41 = call @Unknown23(%40#0, %33) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %42 = call @Unknown30(%arg38) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %43 = mhlo.convolution(%41, %42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<1x256x14x14xf16>
    %44:3 = call @BatchNormTrainingOp31(%43, %arg40, %arg39) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %45 = call @Unknown32(%arg36) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %46 = mhlo.convolution(%41, %45) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<1x256x14x14xf16>
    %47:3 = call @BatchNormTrainingOp31(%46, %arg33, %arg32) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %48 = call @Unknown34(%47#0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %49 = call @Unknown35(%arg37) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %50 = mhlo.convolution(%48, %49) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %51:3 = call @BatchNormTrainingOp31(%50, %arg35, %arg34) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %52 = call @Unknown37(%51#0, %44#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %53 = call @Unknown35(%arg45) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %54 = mhlo.convolution(%52, %53) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %55:3 = call @BatchNormTrainingOp31(%54, %arg42, %arg41) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %56 = call @Unknown34(%55#0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %57 = call @Unknown35(%arg46) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %58 = mhlo.convolution(%56, %57) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %59:3 = call @BatchNormTrainingOp31(%58, %arg44, %arg43) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %60 = call @Unknown37(%59#0, %52) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %61 = call @Unknown44(%arg53) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %62 = mhlo.convolution(%60, %61) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<1x512x7x7xf16>
    %63:3 = call @BatchNormTrainingOp45(%62, %arg55, %arg54) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %64 = call @Unknown46(%arg51) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %65 = mhlo.convolution(%60, %64) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<1x512x7x7xf16>
    %66:3 = call @BatchNormTrainingOp45(%65, %arg48, %arg47) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %67 = call @Unknown48(%66#0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %68 = call @Unknown49(%arg52) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %69 = mhlo.convolution(%67, %68) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %70:3 = call @BatchNormTrainingOp45(%69, %arg50, %arg49) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %71 = call @Unknown51(%70#0, %63#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %72 = call @Unknown49(%arg60) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %73 = mhlo.convolution(%71, %72) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %74:3 = call @BatchNormTrainingOp45(%73, %arg57, %arg56) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %75 = call @Unknown48(%74#0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %76 = call @Unknown49(%arg61) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %77 = mhlo.convolution(%75, %76) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %78:3 = call @BatchNormTrainingOp45(%77, %arg59, %arg58) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %79 = call @Unknown51(%78#0, %71) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %80 = call @Unknown58(%79) : (tensor<1x512x7x7xf16>) -> tensor<1x512xf16>
    %81 = call @Unknown59(%80) : (tensor<1x512xf16>) -> tensor<1x512xf16>
    %82 = call @Unknown60(%arg4) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %83 = "mhlo.transpose"(%82) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    %84 = "mhlo.dot_general"(%81, %82) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512xf16>, tensor<1000x512xf16>) -> tensor<1x1000xf16>
    %85 = call @Unknown61(%arg3, %84) : (tensor<1000xf32>, tensor<1x1000xf16>) -> tensor<1x1000xf16>
    %86 = call @Unknown62(%4#1, %arg63) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %87 = call @Unknown62(%4#2, %arg64) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %88 = call @Unknown62(%9#1, %arg66) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %89 = call @Unknown62(%9#2, %arg67) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %90 = call @Unknown62(%13#1, %arg69) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %91 = call @Unknown62(%13#2, %arg70) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %92 = call @Unknown62(%17#1, %arg72) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %93 = call @Unknown62(%17#2, %arg73) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %94 = call @Unknown62(%21#1, %arg75) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %95 = call @Unknown62(%21#2, %arg76) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %96 = call @Unknown72(%28#1, %arg78) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %97 = call @Unknown72(%28#2, %arg79) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %98 = call @Unknown72(%32#1, %arg81) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %99 = call @Unknown72(%32#2, %arg82) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %100 = call @Unknown72(%25#1, %arg84) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %101 = call @Unknown72(%25#2, %arg85) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %102 = call @Unknown72(%36#1, %arg87) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %103 = call @Unknown72(%36#2, %arg88) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %104 = call @Unknown72(%40#1, %arg90) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %105 = call @Unknown72(%40#2, %arg91) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %106 = call @Unknown82(%47#1, %arg93) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %107 = call @Unknown82(%47#2, %arg94) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %108 = call @Unknown82(%51#1, %arg96) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %109 = call @Unknown82(%51#2, %arg97) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %110 = call @Unknown82(%44#1, %arg99) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %111 = call @Unknown82(%44#2, %arg100) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %112 = call @Unknown82(%55#1, %arg102) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %113 = call @Unknown82(%55#2, %arg103) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %114 = call @Unknown82(%59#1, %arg105) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %115 = call @Unknown82(%59#2, %arg106) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %116 = call @Unknown92(%66#1, %arg108) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %117 = call @Unknown92(%66#2, %arg109) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %118 = call @Unknown92(%70#1, %arg111) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %119 = call @Unknown92(%70#2, %arg112) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %120 = call @Unknown92(%63#1, %arg114) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %121 = call @Unknown92(%63#2, %arg115) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %122 = call @Unknown92(%74#1, %arg117) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %123 = call @Unknown92(%74#2, %arg118) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %124 = call @Unknown92(%78#1, %arg120) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %125 = call @Unknown92(%78#2, %arg121) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    return %85, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %2, %1, %3, %5, %6, %7, %8, %10, %11, %12, %14, %15, %16, %18, %19, %20, %22, %26, %27, %29, %30, %31, %23, %24, %33, %34, %35, %37, %38, %39, %41, %45, %46, %48, %49, %50, %42, %43, %52, %53, %54, %56, %57, %58, %60, %64, %65, %67, %68, %69, %61, %62, %71, %72, %73, %75, %76, %77, %79, %81, %83 : tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>
  }
}