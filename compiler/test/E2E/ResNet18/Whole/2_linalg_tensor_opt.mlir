// RUN: byteir-opt %s -linalg-tensor-opt | FileCheck %s

// CHECK-LABEL: func.func @main

module @IrToMhlo.2452 {
  func.func private @Unknown0(%arg0: tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16>
    return %0 : tensor<4x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    return %0 : tensor<64x3x7x7xf16>
  }
  func.func private @BatchNormTrainingOp2(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x112x112xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    return %1 : tensor<4x64x112x112xf16>
  }
  func.func private @Unknown3(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    return %0 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown7(%arg0: tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    return %0 : tensor<128x64x1x1xf16>
  }
  func.func private @Unknown8(%arg0: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    return %0 : tensor<128x64x3x3xf16>
  }
  func.func private @Unknown9(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    return %0 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown12(%arg0: tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    return %0 : tensor<256x128x1x1xf16>
  }
  func.func private @Unknown13(%arg0: tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    return %0 : tensor<256x128x3x3xf16>
  }
  func.func private @Unknown14(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    return %0 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown17(%arg0: tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    return %0 : tensor<512x256x1x1xf16>
  }
  func.func private @Unknown18(%arg0: tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    return %0 : tensor<512x256x3x3xf16>
  }
  func.func private @Unknown19(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    return %0 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown22(%arg0: tensor<4x1000xf32>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<-2.500000e-01> : tensor<4x1000xf32>
    %1 = mhlo.multiply %arg0, %0 : tensor<4x1000xf32>
    %2 = mhlo.convert %1 : (tensor<4x1000xf32>) -> tensor<4x1000xf16>
    return %2 : tensor<4x1000xf16>
  }
  func.func private @Unknown23(%arg0: tensor<1000x512xf32>) -> tensor<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func.func private @Unknown24(%arg0: tensor<1000xf32>) -> tensor<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<1000xf32>) -> tensor<1000xf16>
    return %0 : tensor<1000xf16>
  }
  func.func private @Unknown25(%arg0: tensor<4x1000xf16>) -> tensor<4xf16> attributes {__byteir_reduction_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
      %2 = mhlo.add %arg1, %arg2 : tensor<f16>
      mhlo.return %2 : tensor<f16>
    }
    return %1 : tensor<4xf16>
  }
  func.func private @Unknown26(%arg0: tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x112x112xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x64x112x112xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x64x112x112xf16>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xi1>
    return %1, %2 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>
  }
  func.func private @BatchNormTrainingOp27(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown28(%arg0: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x64x56x56xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    return %1, %2 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @Unknown30(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x64x56x56xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    return %2, %3 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp35(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown37(%arg0: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x128x28x28xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    return %1, %2 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @Unknown39(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x128x28x28xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    return %2, %3 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp44(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown46(%arg0: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x256x14x14xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    return %1, %2 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @Unknown48(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x256x14x14xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    return %2, %3 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp53(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown55(%arg0: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x512x7x7xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    return %1, %2 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @Unknown57(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x512x7x7xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x512x7x7xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    return %2, %3 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @Unknown62(%arg0: tensor<4x512x7x7xf16>) -> tensor<4x512xf16> attributes {__byteir_reduction_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [3, 2] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<4x512xf16>
     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
      %2 = mhlo.add %arg1, %arg2 : tensor<f16>
      mhlo.return %2 : tensor<f16>
    }
    return %1 : tensor<4x512xf16>
  }
  func.func private @Unknown63(%arg0: tensor<4x512xf16>) -> tensor<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<2.040100e-02> : tensor<4x512xf16>
    %1 = mhlo.multiply %arg0, %0 : tensor<4x512xf16>
    return %1 : tensor<4x512xf16>
  }
  func.func private @Unknown64(%arg0: tensor<1000xf16>, %arg1: tensor<4x1000xf16>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf16>) -> tensor<4x1000xf16>
    %1 = mhlo.add %arg1, %0 : tensor<4x1000xf16>
    return %1 : tensor<4x1000xf16>
  }
  func.func private @Unknown65(%arg0: tensor<4x1000xf16>) -> tensor<4xf16> attributes {__byteir_reduction_fusion__} {
    %0 = mhlo.constant dense<0xFC00> : tensor<f16>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
      %2 = mhlo.maximum %arg1, %arg2 : tensor<f16>
      mhlo.return %2 : tensor<f16>
    }
    return %1 : tensor<4xf16>
  }
  func.func private @Unknown66(%arg0: tensor<4xf16>, %arg1: tensor<4x1000xf16>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %1 = mhlo.subtract %arg1, %0 : tensor<4x1000xf16>
    return %1 : tensor<4x1000xf16>
  }
  func.func private @Unknown67(%arg0: tensor<4x1000xf16>) -> tensor<4xf16> attributes {__byteir_reduction_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.exponential %arg0 : tensor<4x1000xf16>
    %2 = mhlo.reduce(%1 init: %0) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
      %3 = mhlo.add %arg1, %arg2 : tensor<f16>
      mhlo.return %3 : tensor<f16>
    }
    return %2 : tensor<4xf16>
  }
  func.func private @Unknown68(%arg0: tensor<4xf16>) -> tensor<4xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.log %arg0 : tensor<4xf16>
    return %0 : tensor<4xf16>
  }
  func.func private @Unknown69(%arg0: tensor<4xf16>, %arg1: tensor<4x1000xf16>, %arg2: tensor<4xf16>, %arg3: tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %1 = mhlo.subtract %arg1, %0 : tensor<4x1000xf16>
    %2 = mhlo.exponential %1 : tensor<4x1000xf16>
    %3 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %4 = mhlo.multiply %2, %3 : tensor<4x1000xf16>
    %5 = mhlo.subtract %arg3, %4 : tensor<4x1000xf16>
    return %1, %5 : tensor<4x1000xf16>, tensor<4x1000xf16>
  }
  func.func private @Unknown70(%arg0: tensor<4x512xf16>, %arg1: tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<4.900000e+01> : tensor<4x512x7x7xf16>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %2 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x512xf16>) -> tensor<4x512x7x7xf16>
    %3 = mhlo.divide %2, %0 : tensor<4x512x7x7xf16>
    %4 = mhlo.select %arg1, %3, %1 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %4 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp71(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp72(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp73(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown74(%arg0: tensor<4x512x7x7xi1>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown78(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x512x7x7xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @ConvBackwardDataOp84(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp85(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func.func private @ConvBackwardDataOp87(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp88(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func.func private @Unknown89(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp90(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp91(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp92(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown93(%arg0: tensor<4x256x14x14xi1>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardDataOp103(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp104(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func.func private @ConvBackwardDataOp106(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp107(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func.func private @Unknown108(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp109(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp110(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp111(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown112(%arg0: tensor<4x128x28x28xi1>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardDataOp122(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp123(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func.func private @ConvBackwardDataOp125(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp126(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func.func private @Unknown127(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp128(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp129(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp130(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown131(%arg0: tensor<4x64x56x56xi1>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown143(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown144(%arg0: tensor<4x64x112x112xi1>, %arg1: tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x112x112xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>
    return %1 : tensor<4x64x112x112xf16>
  }
  func.func private @BatchNormGradOp145(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x112x112xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardFilterOp146(%arg0: tensor<4x3x224x224xf16>, %arg1: tensor<4x64x112x112xf16>) -> tensor<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
  }
  func.func private @Unknown147(%arg0: tensor<4x1000xf16>, %arg1: tensor<4x1000xf32>) -> tensor<f32> attributes {__byteir_reduction_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.convert %arg0 : (tensor<4x1000xf16>) -> tensor<4x1000xf32>
    %2 = mhlo.multiply %1, %arg1 : tensor<4x1000xf32>
    %3 = mhlo.reduce(%2 init: %0) across dimensions = [0, 1] : (tensor<4x1000xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %4 = mhlo.add %arg2, %arg3 : tensor<f32>
      mhlo.return %4 : tensor<f32>
    }
    return %3 : tensor<f32>
  }
  func.func private @Unknown148(%arg0: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<4.000000e+00> : tensor<f32>
    %1 = mhlo.negate %arg0 : tensor<f32>
    %2 = mhlo.divide %1, %0 : tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @Unknown149(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[64,3,7,7]{0,1,3,2}"} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    return %0 : tensor<64x3x7x7xf32>
  }
  func.func private @Unknown150(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown154(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[128,64,3,3]{0,1,3,2}"} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    return %0 : tensor<128x64x3x3xf32>
  }
  func.func private @Unknown155(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown156(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[128,64,1,1]{0,1,3,2}"} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    return %0 : tensor<128x64x1x1xf32>
  }
  func.func private @Unknown159(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[256,128,3,3]{0,1,3,2}"} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    return %0 : tensor<256x128x3x3xf32>
  }
  func.func private @Unknown160(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown161(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[256,128,1,1]{0,1,3,2}"} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    return %0 : tensor<256x128x1x1xf32>
  }
  func.func private @Unknown164(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[512,256,3,3]{0,1,3,2}"} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    return %0 : tensor<512x256x3x3xf32>
  }
  func.func private @Unknown165(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func.func private @Unknown166(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[512,256,1,1]{0,1,3,2}"} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    return %0 : tensor<512x256x1x1xf32>
  }
  func.func private @MatmulOp169(%arg0: tensor<4x512xf16>, %arg1: tensor<4x1000xf16>) -> tensor<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<4x1000xf16>) -> tensor<512x1000xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %1 : tensor<1000x512xf16>
  }
  func.func private @Unknown170(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[1000,512]{0,1}"} : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    return %0 : tensor<1000x512xf32>
  }
  func.func private @Unknown171(%arg0: tensor<4x1000xf16>) -> tensor<1000xf32> attributes {__byteir_reduction_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.convert %arg0 : (tensor<4x1000xf16>) -> tensor<4x1000xf32>
    %2 = mhlo.reduce(%1 init: %0) across dimensions = [0] : (tensor<4x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %3 = mhlo.add %arg1, %arg2 : tensor<f32>
      mhlo.return %3 : tensor<f32>
    }
    return %2 : tensor<1000xf32>
  }
  func.func private @Unknown172(%arg0: tensor<1000xf32>) -> tensor<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<1000xf32>) -> tensor<1000xf16>
    %1 = mhlo.convert %0 : (tensor<1000xf16>) -> tensor<1000xf32>
    return %1 : tensor<1000xf32>
  }
  func.func @main(%arg0: tensor<4x3x224x224xf32>, %arg1: tensor<4x1000xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64x64x3x3xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64x64x3x3xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64x64x3x3xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64xf32>, %arg22: tensor<64x64x3x3xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<128x64x3x3xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128xf32>, %arg32: tensor<128x128x3x3xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128x64x1x1xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128x3x3xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128x128x3x3xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<256x128x3x3xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256xf32>, %arg57: tensor<256x256x3x3xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256x128x1x1xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256x256x3x3xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<256x256x3x3xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<256xf32>, %arg77: tensor<512x256x3x3xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512xf32>, %arg82: tensor<512x512x3x3xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512x256x1x1xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512xf32>, %arg91: tensor<512xf32>, %arg92: tensor<512x512x3x3xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512xf32>, %arg97: tensor<512x512x3x3xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<512xf32>, %arg102: tensor<1000x512xf32>, %arg103: tensor<1000xf32>) -> (tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.constant dense<0xFC00> : tensor<f16>
    %2 = call @Unknown0(%arg0) : (tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16>
    %3 = call @Unknown1(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %4 = mhlo.convolution(%2, %3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<4x64x112x112xf16>
    %5 = call @BatchNormTrainingOp2(%4, %arg3, %arg4) : (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x112x112xf16>
    %6 = call @Unknown3(%arg7) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %7 = call @Unknown3(%arg12) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %8 = call @Unknown3(%arg17) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %9 = call @Unknown3(%arg22) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %10 = call @Unknown7(%arg37) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %11 = call @Unknown8(%arg27) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %12 = call @Unknown9(%arg32) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %13 = call @Unknown9(%arg42) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %14 = call @Unknown9(%arg47) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %15 = call @Unknown12(%arg62) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %16 = call @Unknown13(%arg52) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %17 = call @Unknown14(%arg57) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %18 = call @Unknown14(%arg67) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %19 = call @Unknown14(%arg72) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %20 = call @Unknown17(%arg87) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %21 = call @Unknown18(%arg77) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %22 = call @Unknown19(%arg82) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %23 = call @Unknown19(%arg92) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %24 = call @Unknown19(%arg97) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %25 = call @Unknown22(%arg1) : (tensor<4x1000xf32>) -> tensor<4x1000xf16>
    %26 = call @Unknown23(%arg102) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %27 = call @Unknown24(%arg103) : (tensor<1000xf32>) -> tensor<1000xf16>
    %28 = call @Unknown25(%25) : (tensor<4x1000xf16>) -> tensor<4xf16>
    %29:2 = call @Unknown26(%5) : (tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>)
    %30 = "mhlo.reduce_window"(%29#0, %1) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<f16>) -> tensor<4x64x56x56xf16>
    %31 = mhlo.convolution(%30, %6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %32 = call @BatchNormTrainingOp27(%31, %arg8, %arg9) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %33:2 = call @Unknown28(%32) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %34 = mhlo.convolution(%33#0, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %35 = call @BatchNormTrainingOp27(%34, %arg13, %arg14) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %36:2 = call @Unknown30(%35, %30) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %37 = mhlo.convolution(%36#0, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %38 = call @BatchNormTrainingOp27(%37, %arg18, %arg19) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %39:2 = call @Unknown28(%38) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %40 = mhlo.convolution(%39#0, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %41 = call @BatchNormTrainingOp27(%40, %arg23, %arg24) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %42:2 = call @Unknown30(%41, %36#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %43 = mhlo.convolution(%42#0, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<4x128x28x28xf16>
    %44 = call @BatchNormTrainingOp35(%43, %arg38, %arg39) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %45 = mhlo.convolution(%42#0, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<4x128x28x28xf16>
    %46 = call @BatchNormTrainingOp35(%45, %arg28, %arg29) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %47:2 = call @Unknown37(%46) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %48 = mhlo.convolution(%47#0, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %49 = call @BatchNormTrainingOp35(%48, %arg33, %arg34) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %50:2 = call @Unknown39(%49, %44) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %51 = mhlo.convolution(%50#0, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %52 = call @BatchNormTrainingOp35(%51, %arg43, %arg44) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %53:2 = call @Unknown37(%52) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %54 = mhlo.convolution(%53#0, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %55 = call @BatchNormTrainingOp35(%54, %arg48, %arg49) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %56:2 = call @Unknown39(%55, %50#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %57 = mhlo.convolution(%56#0, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<4x256x14x14xf16>
    %58 = call @BatchNormTrainingOp44(%57, %arg63, %arg64) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %59 = mhlo.convolution(%56#0, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<4x256x14x14xf16>
    %60 = call @BatchNormTrainingOp44(%59, %arg53, %arg54) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %61:2 = call @Unknown46(%60) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %62 = mhlo.convolution(%61#0, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %63 = call @BatchNormTrainingOp44(%62, %arg58, %arg59) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %64:2 = call @Unknown48(%63, %58) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %65 = mhlo.convolution(%64#0, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %66 = call @BatchNormTrainingOp44(%65, %arg68, %arg69) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %67:2 = call @Unknown46(%66) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %68 = mhlo.convolution(%67#0, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %69 = call @BatchNormTrainingOp44(%68, %arg73, %arg74) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %70:2 = call @Unknown48(%69, %64#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %71 = mhlo.convolution(%70#0, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<4x512x7x7xf16>
    %72 = call @BatchNormTrainingOp53(%71, %arg88, %arg89) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %73 = mhlo.convolution(%70#0, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<4x512x7x7xf16>
    %74 = call @BatchNormTrainingOp53(%73, %arg78, %arg79) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %75:2 = call @Unknown55(%74) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %76 = mhlo.convolution(%75#0, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %77 = call @BatchNormTrainingOp53(%76, %arg83, %arg84) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %78:2 = call @Unknown57(%77, %72) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %79 = mhlo.convolution(%78#0, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %80 = call @BatchNormTrainingOp53(%79, %arg93, %arg94) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %81:2 = call @Unknown55(%80) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %82 = mhlo.convolution(%81#0, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %83 = call @BatchNormTrainingOp53(%82, %arg98, %arg99) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %84:2 = call @Unknown57(%83, %78#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %85 = call @Unknown62(%84#0) : (tensor<4x512x7x7xf16>) -> tensor<4x512xf16>
    %86 = call @Unknown63(%85) : (tensor<4x512xf16>) -> tensor<4x512xf16>
    %87 = "mhlo.dot_general"(%86, %26) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<1000x512xf16>) -> tensor<4x1000xf16>
    %88 = call @Unknown64(%27, %87) : (tensor<1000xf16>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %89 = call @Unknown65(%88) : (tensor<4x1000xf16>) -> tensor<4xf16>
    %90 = call @Unknown66(%89, %88) : (tensor<4xf16>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %91 = call @Unknown67(%90) : (tensor<4x1000xf16>) -> tensor<4xf16>
    %92 = call @Unknown68(%91) : (tensor<4xf16>) -> tensor<4xf16>
    %93:2 = call @Unknown69(%92, %90, %28, %25) : (tensor<4xf16>, tensor<4x1000xf16>, tensor<4xf16>, tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>)
    %94 = "mhlo.dot"(%93#1, %26) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x1000xf16>, tensor<1000x512xf16>) -> tensor<4x512xf16>
    %95 = call @Unknown70(%94, %84#1) : (tensor<4x512xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %96:3 = call @BatchNormGradOp71(%82, %arg98, %95) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %97 = call @ConvBackwardDataOp72(%96#0, %24) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %98 = call @ConvBackwardFilterOp73(%81#0, %96#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %99 = call @Unknown74(%81#1, %97) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %100:3 = call @BatchNormGradOp71(%79, %arg93, %99) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %101 = call @ConvBackwardDataOp72(%100#0, %23) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %102 = call @ConvBackwardFilterOp73(%78#0, %100#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %103 = call @Unknown78(%95, %101, %78#1) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %104:3 = call @BatchNormGradOp71(%76, %arg83, %103) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %105 = call @ConvBackwardDataOp72(%104#0, %22) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %106 = call @ConvBackwardFilterOp73(%75#0, %104#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %107 = call @Unknown74(%75#1, %105) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %108:3 = call @BatchNormGradOp71(%73, %arg78, %107) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %109 = call @ConvBackwardDataOp84(%108#0, %21) : (tensor<4x512x7x7xf16>, tensor<512x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %110 = call @ConvBackwardFilterOp85(%70#0, %108#0) : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<512x256x3x3xf16>
    %111:3 = call @BatchNormGradOp71(%71, %arg88, %103) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %112 = call @ConvBackwardDataOp87(%111#0, %20) : (tensor<4x512x7x7xf16>, tensor<512x256x1x1xf16>) -> tensor<4x256x14x14xf16>
    %113 = call @ConvBackwardFilterOp88(%70#0, %111#0) : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<512x256x1x1xf16>
    %114 = call @Unknown89(%112, %109, %70#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %115:3 = call @BatchNormGradOp90(%68, %arg73, %114) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %116 = call @ConvBackwardDataOp91(%115#0, %19) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %117 = call @ConvBackwardFilterOp92(%67#0, %115#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %118 = call @Unknown93(%67#1, %116) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %119:3 = call @BatchNormGradOp90(%65, %arg68, %118) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %120 = call @ConvBackwardDataOp91(%119#0, %18) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %121 = call @ConvBackwardFilterOp92(%64#0, %119#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %122 = call @Unknown89(%114, %120, %64#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %123:3 = call @BatchNormGradOp90(%62, %arg58, %122) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %124 = call @ConvBackwardDataOp91(%123#0, %17) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %125 = call @ConvBackwardFilterOp92(%61#0, %123#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %126 = call @Unknown93(%61#1, %124) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %127:3 = call @BatchNormGradOp90(%59, %arg53, %126) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %128 = call @ConvBackwardDataOp103(%127#0, %16) : (tensor<4x256x14x14xf16>, tensor<256x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %129 = call @ConvBackwardFilterOp104(%56#0, %127#0) : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<256x128x3x3xf16>
    %130:3 = call @BatchNormGradOp90(%57, %arg63, %122) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %131 = call @ConvBackwardDataOp106(%130#0, %15) : (tensor<4x256x14x14xf16>, tensor<256x128x1x1xf16>) -> tensor<4x128x28x28xf16>
    %132 = call @ConvBackwardFilterOp107(%56#0, %130#0) : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<256x128x1x1xf16>
    %133 = call @Unknown108(%131, %128, %56#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %134:3 = call @BatchNormGradOp109(%54, %arg48, %133) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %135 = call @ConvBackwardDataOp110(%134#0, %14) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %136 = call @ConvBackwardFilterOp111(%53#0, %134#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %137 = call @Unknown112(%53#1, %135) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %138:3 = call @BatchNormGradOp109(%51, %arg43, %137) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %139 = call @ConvBackwardDataOp110(%138#0, %13) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %140 = call @ConvBackwardFilterOp111(%50#0, %138#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %141 = call @Unknown108(%133, %139, %50#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %142:3 = call @BatchNormGradOp109(%48, %arg33, %141) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %143 = call @ConvBackwardDataOp110(%142#0, %12) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %144 = call @ConvBackwardFilterOp111(%47#0, %142#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %145 = call @Unknown112(%47#1, %143) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %146:3 = call @BatchNormGradOp109(%45, %arg28, %145) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %147 = call @ConvBackwardDataOp122(%146#0, %11) : (tensor<4x128x28x28xf16>, tensor<128x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %148 = call @ConvBackwardFilterOp123(%42#0, %146#0) : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<128x64x3x3xf16>
    %149:3 = call @BatchNormGradOp109(%43, %arg38, %141) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %150 = call @ConvBackwardDataOp125(%149#0, %10) : (tensor<4x128x28x28xf16>, tensor<128x64x1x1xf16>) -> tensor<4x64x56x56xf16>
    %151 = call @ConvBackwardFilterOp126(%42#0, %149#0) : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<128x64x1x1xf16>
    %152 = call @Unknown127(%150, %147, %42#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %153:3 = call @BatchNormGradOp128(%40, %arg23, %152) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %154 = call @ConvBackwardDataOp129(%153#0, %9) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %155 = call @ConvBackwardFilterOp130(%39#0, %153#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %156 = call @Unknown131(%39#1, %154) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %157:3 = call @BatchNormGradOp128(%37, %arg18, %156) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %158 = call @ConvBackwardDataOp129(%157#0, %8) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %159 = call @ConvBackwardFilterOp130(%36#0, %157#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %160 = call @Unknown127(%152, %158, %36#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %161:3 = call @BatchNormGradOp128(%34, %arg13, %160) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %162 = call @ConvBackwardDataOp129(%161#0, %7) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %163 = call @ConvBackwardFilterOp130(%33#0, %161#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %164 = call @Unknown131(%33#1, %162) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %165:3 = call @BatchNormGradOp128(%31, %arg8, %164) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %166 = call @ConvBackwardDataOp129(%165#0, %6) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %167 = call @ConvBackwardFilterOp130(%30, %165#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %168 = call @Unknown143(%160, %166) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %169 = "mhlo.select_and_scatter"(%29#0, %168, %0) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.compare  GE, %arg104, %arg105 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %199 : tensor<i1>
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<4x64x112x112xf16>
    %170 = call @Unknown144(%29#1, %169) : (tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %171:3 = call @BatchNormGradOp145(%4, %arg3, %170) : (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %172 = call @ConvBackwardFilterOp146(%2, %171#0) : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<64x3x7x7xf16>
    %173 = call @Unknown147(%93#0, %arg1) : (tensor<4x1000xf16>, tensor<4x1000xf32>) -> tensor<f32>
    %174 = call @Unknown148(%173) : (tensor<f32>) -> tensor<f32>
    %175 = call @Unknown149(%172) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %176 = call @Unknown150(%167) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %177 = call @Unknown150(%163) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %178 = call @Unknown150(%159) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %179 = call @Unknown150(%155) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %180 = call @Unknown154(%148) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %181 = call @Unknown155(%144) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %182 = call @Unknown156(%151) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %183 = call @Unknown155(%140) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %184 = call @Unknown155(%136) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %185 = call @Unknown159(%129) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %186 = call @Unknown160(%125) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %187 = call @Unknown161(%132) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %188 = call @Unknown160(%121) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %189 = call @Unknown160(%117) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %190 = call @Unknown164(%110) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %191 = call @Unknown165(%106) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %192 = call @Unknown166(%113) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %193 = call @Unknown165(%102) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %194 = call @Unknown165(%98) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %195 = call @MatmulOp169(%86, %93#1) : (tensor<4x512xf16>, tensor<4x1000xf16>) -> tensor<1000x512xf16>
    %196 = call @Unknown170(%195) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %197 = call @Unknown171(%93#1) : (tensor<4x1000xf16>) -> tensor<1000xf32>
    %198 = call @Unknown172(%197) : (tensor<1000xf32>) -> tensor<1000xf32>
    return %174, %175, %171#1, %171#2, %176, %165#1, %165#2, %177, %161#1, %161#2, %178, %157#1, %157#2, %179, %153#1, %153#2, %180, %146#1, %146#2, %181, %142#1, %142#2, %182, %149#1, %149#2, %183, %138#1, %138#2, %184, %134#1, %134#2, %185, %127#1, %127#2, %186, %123#1, %123#2, %187, %130#1, %130#2, %188, %119#1, %119#2, %189, %115#1, %115#2, %190, %108#1, %108#2, %191, %104#1, %104#2, %192, %111#1, %111#2, %193, %100#1, %100#2, %194, %96#1, %96#2, %196, %198 : tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>
  }
}