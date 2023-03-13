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
  func.func private @Unknown4(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    return %0 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown5(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    return %0 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown6(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown10(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    return %0 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown11(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown15(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    return %0 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown16(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown20(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    return %0 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown21(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown25(%arg0: tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x112x112xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x64x112x112xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x64x112x112xf16>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xi1>
    return %1, %2 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>
  }
  func.func private @BatchNormTrainingOp26(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown27(%arg0: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x64x56x56xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    return %1, %2 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp28(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown29(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x64x56x56xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    return %2, %3 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp30(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown31(%arg0: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x64x56x56xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    return %1, %2 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp32(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown33(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x64x56x56xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xi1>
    return %2, %3 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp34(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormTrainingOp35(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown36(%arg0: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x128x28x28xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    return %1, %2 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp37(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown38(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x128x28x28xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    return %2, %3 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp39(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown40(%arg0: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x128x28x28xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    return %1, %2 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp41(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown42(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x128x28x28xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xi1>
    return %2, %3 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp43(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormTrainingOp44(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown45(%arg0: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x256x14x14xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    return %1, %2 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp46(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown47(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x256x14x14xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    return %2, %3 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp48(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown49(%arg0: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x256x14x14xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    return %1, %2 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp50(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown51(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x256x14x14xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xi1>
    return %2, %3 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp52(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormTrainingOp53(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown54(%arg0: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x512x7x7xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    return %1, %2 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp55(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown56(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x512x7x7xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x512x7x7xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    return %2, %3 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp57(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown58(%arg0: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<4x512x7x7xf16>
    %2 = mhlo.compare  GT, %1, %0 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    return %1, %2 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp59(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown60(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x512x7x7xf16>
    %2 = mhlo.maximum %1, %0 : tensor<4x512x7x7xf16>
    %3 = mhlo.compare  GT, %2, %0 : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xi1>
    return %2, %3 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @Unknown61(%arg0: tensor<4x512xf16>) -> tensor<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<2.040100e-02> : tensor<4x512xf16>
    %1 = mhlo.multiply %arg0, %0 : tensor<4x512xf16>
    return %1 : tensor<4x512xf16>
  }
  func.func private @Unknown62(%arg0: tensor<1000xf16>, %arg1: tensor<4x1000xf16>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf16>) -> tensor<4x1000xf16>
    %1 = mhlo.add %arg1, %0 : tensor<4x1000xf16>
    return %1 : tensor<4x1000xf16>
  }
  func.func private @Unknown63(%arg0: tensor<4xf16>, %arg1: tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %1 = mhlo.subtract %arg1, %0 : tensor<4x1000xf16>
    %2 = mhlo.exponential %1 : tensor<4x1000xf16>
    return %1, %2 : tensor<4x1000xf16>, tensor<4x1000xf16>
  }
  func.func private @Unknown64(%arg0: tensor<4xf16>) -> tensor<4xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.log %arg0 : tensor<4xf16>
    return %0 : tensor<4xf16>
  }
  func.func private @Unknown65(%arg0: tensor<4xf16>, %arg1: tensor<4x1000xf16>, %arg2: tensor<4xf16>, %arg3: tensor<4x1000xf16>, %arg4: tensor<4x1000xf32>) -> (tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %1 = mhlo.subtract %arg1, %0 : tensor<4x1000xf16>
    %2 = mhlo.exponential %1 : tensor<4x1000xf16>
    %3 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x1000xf16>
    %4 = mhlo.multiply %2, %3 : tensor<4x1000xf16>
    %5 = mhlo.subtract %arg3, %4 : tensor<4x1000xf16>
    %6 = mhlo.convert %1 : (tensor<4x1000xf16>) -> tensor<4x1000xf32>
    %7 = mhlo.multiply %6, %arg4 : tensor<4x1000xf32>
    %8 = mhlo.convert %5 : (tensor<4x1000xf16>) -> tensor<4x1000xf32>
    return %5, %7, %8 : tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>
  }
  func.func private @Unknown66(%arg0: tensor<4x512xf16>, %arg1: tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<4.900000e+01> : tensor<4x512x7x7xf16>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %2 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x512xf16>) -> tensor<4x512x7x7xf16>
    %3 = mhlo.divide %2, %0 : tensor<4x512x7x7xf16>
    %4 = mhlo.select %arg1, %3, %1 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %4 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp67(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp68(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp69(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown70(%arg0: tensor<4x512x7x7xi1>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp71(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
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
  func.func private @Unknown74(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x512x7x7xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp75(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp76(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp77(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown78(%arg0: tensor<4x512x7x7xi1>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x512x7x7xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp79(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp80(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp81(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func.func private @BatchNormGradOp82(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp83(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp84(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func.func private @Unknown85(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp86(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp87(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp88(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown89(%arg0: tensor<4x256x14x14xi1>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp90(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
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
  func.func private @Unknown93(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x256x14x14xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp94(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp95(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp96(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown97(%arg0: tensor<4x256x14x14xi1>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x256x14x14xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp98(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp99(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp100(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func.func private @BatchNormGradOp101(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp102(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp103(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func.func private @Unknown104(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp105(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp106(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp107(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown108(%arg0: tensor<4x128x28x28xi1>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp109(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
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
  func.func private @Unknown112(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x128x28x28xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp113(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp114(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp115(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown116(%arg0: tensor<4x128x28x28xi1>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x128x28x28xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp117(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp118(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp119(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func.func private @BatchNormGradOp120(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp121(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp122(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func.func private @Unknown123(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp124(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp125(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp126(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown127(%arg0: tensor<4x64x56x56xi1>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp128(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
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
  func.func private @Unknown131(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    %2 = mhlo.select %arg2, %1, %0 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp132(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp133(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp134(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown135(%arg0: tensor<4x64x56x56xi1>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x56x56xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp136(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp137(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp138(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown139(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<4x64x56x56xf16>
    return %0 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown140(%arg0: tensor<4x64x112x112xi1>, %arg1: tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<4x64x112x112xf16>
    %1 = mhlo.select %arg0, %arg1, %0 : tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>
    return %1 : tensor<4x64x112x112xf16>
  }
  func.func private @BatchNormGradOp141(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %1 = mhlo.convert %arg2 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x112x112xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardFilterOp142(%arg0: tensor<4x3x224x224xf16>, %arg1: tensor<4x64x112x112xf16>) -> tensor<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
  }
  func.func private @Unknown143(%arg0: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<4.000000e+00> : tensor<f32>
    %1 = mhlo.negate %arg0 : tensor<f32>
    %2 = mhlo.divide %1, %0 : tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @Unknown144(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[64,3,7,7]{0,1,3,2}"} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    return %0 : tensor<64x3x7x7xf32>
  }
  func.func private @Unknown145(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown146(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown147(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown148(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown149(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[128,64,3,3]{0,1,3,2}"} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    return %0 : tensor<128x64x3x3xf32>
  }
  func.func private @Unknown150(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown151(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[128,64,1,1]{0,1,3,2}"} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    return %0 : tensor<128x64x1x1xf32>
  }
  func.func private @Unknown152(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown153(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown154(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[256,128,3,3]{0,1,3,2}"} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    return %0 : tensor<256x128x3x3xf32>
  }
  func.func private @Unknown155(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown156(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[256,128,1,1]{0,1,3,2}"} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    return %0 : tensor<256x128x1x1xf32>
  }
  func.func private @Unknown157(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown158(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown159(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[512,256,3,3]{0,1,3,2}"} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    return %0 : tensor<512x256x3x3xf32>
  }
  func.func private @Unknown160(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func.func private @Unknown161(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[512,256,1,1]{0,1,3,2}"} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    return %0 : tensor<512x256x1x1xf32>
  }
  func.func private @Unknown162(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func.func private @Unknown163(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func.func private @MatmulOp164(%arg0: tensor<4x512xf16>, %arg1: tensor<4x1000xf16>) -> tensor<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<4x1000xf16>) -> tensor<512x1000xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %1 : tensor<1000x512xf16>
  }
  func.func private @Unknown165(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.convert %arg0 {xla_shape = "f32[1000,512]{0,1}"} : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    return %0 : tensor<1000x512xf32>
  }
  func.func @main(%arg0: tensor<4x3x224x224xf32>, %arg1: tensor<4x1000xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64x64x3x3xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64x64x3x3xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64x64x3x3xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64xf32>, %arg22: tensor<64x64x3x3xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<128x64x3x3xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128xf32>, %arg32: tensor<128x128x3x3xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128x64x1x1xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128x3x3xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128x128x3x3xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<256x128x3x3xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256xf32>, %arg57: tensor<256x256x3x3xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256x128x1x1xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256x256x3x3xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<256x256x3x3xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<256xf32>, %arg77: tensor<512x256x3x3xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512xf32>, %arg82: tensor<512x512x3x3xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512x256x1x1xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512xf32>, %arg91: tensor<512xf32>, %arg92: tensor<512x512x3x3xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512xf32>, %arg97: tensor<512x512x3x3xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<512xf32>, %arg102: tensor<1000x512xf32>, %arg103: tensor<1000xf32>) -> (tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %2 = mhlo.constant dense<0xFC00> : tensor<f16>
    %3 = call @Unknown0(%arg0) : (tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16>
    %4 = call @Unknown1(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %5 = mhlo.convolution(%3, %4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<4x64x112x112xf16>
    %6 = call @BatchNormTrainingOp2(%5, %arg3, %arg4) : (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x112x112xf16>
    %7 = call @Unknown3(%arg7) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %8 = call @Unknown4(%arg12) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %9 = call @Unknown5(%arg17) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %10 = call @Unknown6(%arg22) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %11 = call @Unknown7(%arg37) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %12 = call @Unknown8(%arg27) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %13 = call @Unknown9(%arg32) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %14 = call @Unknown10(%arg42) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %15 = call @Unknown11(%arg47) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %16 = call @Unknown12(%arg62) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %17 = call @Unknown13(%arg52) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %18 = call @Unknown14(%arg57) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %19 = call @Unknown15(%arg67) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %20 = call @Unknown16(%arg72) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %21 = call @Unknown17(%arg87) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %22 = call @Unknown18(%arg77) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %23 = call @Unknown19(%arg82) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %24 = call @Unknown20(%arg92) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %25 = call @Unknown21(%arg97) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %26 = call @Unknown22(%arg1) : (tensor<4x1000xf32>) -> tensor<4x1000xf16>
    %27 = call @Unknown23(%arg102) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %28 = call @Unknown24(%arg103) : (tensor<1000xf32>) -> tensor<1000xf16>
    %29 = mhlo.reduce(%26 init: %1) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %199 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }
    %30:2 = call @Unknown25(%6) : (tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>)
    %31 = "mhlo.reduce_window"(%30#0, %2) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<f16>) -> tensor<4x64x56x56xf16>
    %32 = mhlo.convolution(%31, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %33 = call @BatchNormTrainingOp26(%32, %arg8, %arg9) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %34:2 = call @Unknown27(%33) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %35 = mhlo.convolution(%34#0, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %36 = call @BatchNormTrainingOp28(%35, %arg13, %arg14) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %37:2 = call @Unknown29(%36, %31) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %38 = mhlo.convolution(%37#0, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %39 = call @BatchNormTrainingOp30(%38, %arg18, %arg19) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %40:2 = call @Unknown31(%39) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %41 = mhlo.convolution(%40#0, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %42 = call @BatchNormTrainingOp32(%41, %arg23, %arg24) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %43:2 = call @Unknown33(%42, %37#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %44 = mhlo.convolution(%43#0, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<4x128x28x28xf16>
    %45 = call @BatchNormTrainingOp34(%44, %arg38, %arg39) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %46 = mhlo.convolution(%43#0, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<4x128x28x28xf16>
    %47 = call @BatchNormTrainingOp35(%46, %arg28, %arg29) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %48:2 = call @Unknown36(%47) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %49 = mhlo.convolution(%48#0, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %50 = call @BatchNormTrainingOp37(%49, %arg33, %arg34) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %51:2 = call @Unknown38(%50, %45) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %52 = mhlo.convolution(%51#0, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %53 = call @BatchNormTrainingOp39(%52, %arg43, %arg44) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %54:2 = call @Unknown40(%53) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %55 = mhlo.convolution(%54#0, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %56 = call @BatchNormTrainingOp41(%55, %arg48, %arg49) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %57:2 = call @Unknown42(%56, %51#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %58 = mhlo.convolution(%57#0, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<4x256x14x14xf16>
    %59 = call @BatchNormTrainingOp43(%58, %arg63, %arg64) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %60 = mhlo.convolution(%57#0, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<4x256x14x14xf16>
    %61 = call @BatchNormTrainingOp44(%60, %arg53, %arg54) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %62:2 = call @Unknown45(%61) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %63 = mhlo.convolution(%62#0, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %64 = call @BatchNormTrainingOp46(%63, %arg58, %arg59) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %65:2 = call @Unknown47(%64, %59) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %66 = mhlo.convolution(%65#0, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %67 = call @BatchNormTrainingOp48(%66, %arg68, %arg69) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %68:2 = call @Unknown49(%67) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %69 = mhlo.convolution(%68#0, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %70 = call @BatchNormTrainingOp50(%69, %arg73, %arg74) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %71:2 = call @Unknown51(%70, %65#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %72 = mhlo.convolution(%71#0, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<4x512x7x7xf16>
    %73 = call @BatchNormTrainingOp52(%72, %arg88, %arg89) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %74 = mhlo.convolution(%71#0, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<4x512x7x7xf16>
    %75 = call @BatchNormTrainingOp53(%74, %arg78, %arg79) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %76:2 = call @Unknown54(%75) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %77 = mhlo.convolution(%76#0, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %78 = call @BatchNormTrainingOp55(%77, %arg83, %arg84) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %79:2 = call @Unknown56(%78, %73) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %80 = mhlo.convolution(%79#0, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %81 = call @BatchNormTrainingOp57(%80, %arg93, %arg94) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %82:2 = call @Unknown58(%81) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %83 = mhlo.convolution(%82#0, %25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %84 = call @BatchNormTrainingOp59(%83, %arg98, %arg99) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %85:2 = call @Unknown60(%84, %79#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %86 = mhlo.reduce(%85#0 init: %1) across dimensions = [3, 2] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<4x512xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %199 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }
    %87 = call @Unknown61(%86) : (tensor<4x512xf16>) -> tensor<4x512xf16>
    %88 = "mhlo.dot_general"(%87, %27) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<1000x512xf16>) -> tensor<4x1000xf16>
    %89 = call @Unknown62(%28, %88) : (tensor<1000xf16>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %90 = mhlo.reduce(%89 init: %2) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %199 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }
    %91:2 = call @Unknown63(%90, %89) : (tensor<4xf16>, tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>)
    %92 = mhlo.reduce(%91#1 init: %1) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %199 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }
    %93 = call @Unknown64(%92) : (tensor<4xf16>) -> tensor<4xf16>
    %94:3 = call @Unknown65(%93, %91#0, %29, %26, %arg1) : (tensor<4xf16>, tensor<4x1000xf16>, tensor<4xf16>, tensor<4x1000xf16>, tensor<4x1000xf32>) -> (tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>)
    %95 = "mhlo.dot"(%94#0, %27) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x1000xf16>, tensor<1000x512xf16>) -> tensor<4x512xf16>
    %96 = call @Unknown66(%95, %85#1) : (tensor<4x512xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %97:3 = call @BatchNormGradOp67(%83, %arg98, %96) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %98 = call @ConvBackwardDataOp68(%97#0, %25) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %99 = call @ConvBackwardFilterOp69(%82#0, %97#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %100 = call @Unknown70(%82#1, %98) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %101:3 = call @BatchNormGradOp71(%80, %arg93, %100) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %102 = call @ConvBackwardDataOp72(%101#0, %24) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %103 = call @ConvBackwardFilterOp73(%79#0, %101#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %104 = call @Unknown74(%96, %102, %79#1) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %105:3 = call @BatchNormGradOp75(%77, %arg83, %104) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %106 = call @ConvBackwardDataOp76(%105#0, %23) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %107 = call @ConvBackwardFilterOp77(%76#0, %105#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %108 = call @Unknown78(%76#1, %106) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %109:3 = call @BatchNormGradOp79(%74, %arg78, %108) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %110 = call @ConvBackwardDataOp80(%109#0, %22) : (tensor<4x512x7x7xf16>, tensor<512x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %111 = call @ConvBackwardFilterOp81(%71#0, %109#0) : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<512x256x3x3xf16>
    %112:3 = call @BatchNormGradOp82(%72, %arg88, %104) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %113 = call @ConvBackwardDataOp83(%112#0, %21) : (tensor<4x512x7x7xf16>, tensor<512x256x1x1xf16>) -> tensor<4x256x14x14xf16>
    %114 = call @ConvBackwardFilterOp84(%71#0, %112#0) : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<512x256x1x1xf16>
    %115 = call @Unknown85(%113, %110, %71#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %116:3 = call @BatchNormGradOp86(%69, %arg73, %115) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %117 = call @ConvBackwardDataOp87(%116#0, %20) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %118 = call @ConvBackwardFilterOp88(%68#0, %116#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %119 = call @Unknown89(%68#1, %117) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %120:3 = call @BatchNormGradOp90(%66, %arg68, %119) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %121 = call @ConvBackwardDataOp91(%120#0, %19) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %122 = call @ConvBackwardFilterOp92(%65#0, %120#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %123 = call @Unknown93(%115, %121, %65#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %124:3 = call @BatchNormGradOp94(%63, %arg58, %123) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %125 = call @ConvBackwardDataOp95(%124#0, %18) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %126 = call @ConvBackwardFilterOp96(%62#0, %124#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %127 = call @Unknown97(%62#1, %125) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %128:3 = call @BatchNormGradOp98(%60, %arg53, %127) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %129 = call @ConvBackwardDataOp99(%128#0, %17) : (tensor<4x256x14x14xf16>, tensor<256x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %130 = call @ConvBackwardFilterOp100(%57#0, %128#0) : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<256x128x3x3xf16>
    %131:3 = call @BatchNormGradOp101(%58, %arg63, %123) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %132 = call @ConvBackwardDataOp102(%131#0, %16) : (tensor<4x256x14x14xf16>, tensor<256x128x1x1xf16>) -> tensor<4x128x28x28xf16>
    %133 = call @ConvBackwardFilterOp103(%57#0, %131#0) : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<256x128x1x1xf16>
    %134 = call @Unknown104(%132, %129, %57#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %135:3 = call @BatchNormGradOp105(%55, %arg48, %134) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %136 = call @ConvBackwardDataOp106(%135#0, %15) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %137 = call @ConvBackwardFilterOp107(%54#0, %135#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %138 = call @Unknown108(%54#1, %136) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %139:3 = call @BatchNormGradOp109(%52, %arg43, %138) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %140 = call @ConvBackwardDataOp110(%139#0, %14) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %141 = call @ConvBackwardFilterOp111(%51#0, %139#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %142 = call @Unknown112(%134, %140, %51#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %143:3 = call @BatchNormGradOp113(%49, %arg33, %142) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %144 = call @ConvBackwardDataOp114(%143#0, %13) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %145 = call @ConvBackwardFilterOp115(%48#0, %143#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %146 = call @Unknown116(%48#1, %144) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %147:3 = call @BatchNormGradOp117(%46, %arg28, %146) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %148 = call @ConvBackwardDataOp118(%147#0, %12) : (tensor<4x128x28x28xf16>, tensor<128x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %149 = call @ConvBackwardFilterOp119(%43#0, %147#0) : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<128x64x3x3xf16>
    %150:3 = call @BatchNormGradOp120(%44, %arg38, %142) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %151 = call @ConvBackwardDataOp121(%150#0, %11) : (tensor<4x128x28x28xf16>, tensor<128x64x1x1xf16>) -> tensor<4x64x56x56xf16>
    %152 = call @ConvBackwardFilterOp122(%43#0, %150#0) : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<128x64x1x1xf16>
    %153 = call @Unknown123(%151, %148, %43#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %154:3 = call @BatchNormGradOp124(%41, %arg23, %153) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %155 = call @ConvBackwardDataOp125(%154#0, %10) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %156 = call @ConvBackwardFilterOp126(%40#0, %154#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %157 = call @Unknown127(%40#1, %155) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %158:3 = call @BatchNormGradOp128(%38, %arg18, %157) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %159 = call @ConvBackwardDataOp129(%158#0, %9) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %160 = call @ConvBackwardFilterOp130(%37#0, %158#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %161 = call @Unknown131(%153, %159, %37#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %162:3 = call @BatchNormGradOp132(%35, %arg13, %161) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %163 = call @ConvBackwardDataOp133(%162#0, %8) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %164 = call @ConvBackwardFilterOp134(%34#0, %162#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %165 = call @Unknown135(%34#1, %163) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %166:3 = call @BatchNormGradOp136(%32, %arg8, %165) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %167 = call @ConvBackwardDataOp137(%166#0, %7) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %168 = call @ConvBackwardFilterOp138(%31, %166#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %169 = call @Unknown139(%161, %167) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %170 = "mhlo.select_and_scatter"(%30#0, %169, %1) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.compare  GE, %arg104, %arg105 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %199 : tensor<i1>
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<4x64x112x112xf16>
    %171 = call @Unknown140(%30#1, %170) : (tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %172:3 = call @BatchNormGradOp141(%5, %arg3, %171) : (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %173 = call @ConvBackwardFilterOp142(%3, %172#0) : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<64x3x7x7xf16>
    %174 = mhlo.reduce(%94#1 init: %0) across dimensions = [0, 1] : (tensor<4x1000xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg104: tensor<f32>, %arg105: tensor<f32>)  {
      %199 = mhlo.add %arg104, %arg105 : tensor<f32>
      mhlo.return %199 : tensor<f32>
    }
    %175 = call @Unknown143(%174) : (tensor<f32>) -> tensor<f32>
    %176 = call @Unknown144(%173) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %177 = call @Unknown145(%168) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %178 = call @Unknown146(%164) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %179 = call @Unknown147(%160) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %180 = call @Unknown148(%156) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %181 = call @Unknown149(%149) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %182 = call @Unknown150(%145) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %183 = call @Unknown151(%152) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %184 = call @Unknown152(%141) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %185 = call @Unknown153(%137) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %186 = call @Unknown154(%130) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %187 = call @Unknown155(%126) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %188 = call @Unknown156(%133) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %189 = call @Unknown157(%122) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %190 = call @Unknown158(%118) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %191 = call @Unknown159(%111) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %192 = call @Unknown160(%107) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %193 = call @Unknown161(%114) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %194 = call @Unknown162(%103) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %195 = call @Unknown163(%99) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %196 = call @MatmulOp164(%87, %94#0) : (tensor<4x512xf16>, tensor<4x1000xf16>) -> tensor<1000x512xf16>
    %197 = call @Unknown165(%196) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %198 = mhlo.reduce(%94#2 init: %0) across dimensions = [0] : (tensor<4x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg104: tensor<f32>, %arg105: tensor<f32>)  {
      %199 = mhlo.add %arg104, %arg105 : tensor<f32>
      mhlo.return %199 : tensor<f32>
    }
    return %175, %176, %172#1, %172#2, %177, %166#1, %166#2, %178, %162#1, %162#2, %179, %158#1, %158#2, %180, %154#1, %154#2, %181, %147#1, %147#2, %182, %143#1, %143#2, %183, %150#1, %150#2, %184, %139#1, %139#2, %185, %135#1, %135#2, %186, %128#1, %128#2, %187, %124#1, %124#2, %188, %131#1, %131#2, %189, %120#1, %120#2, %190, %116#1, %116#2, %191, %109#1, %109#2, %192, %105#1, %105#2, %193, %112#1, %112#2, %194, %101#1, %101#2, %195, %97#1, %97#2, %197, %198 : tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>
  }
}