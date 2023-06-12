// RUN: byteir-opt %s -byre-tensor-opt="append-arg-types entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func private @Unknown0(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1x3x224x224xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x3x224x224xf32>) outs(%0 : tensor<1x3x224x224xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<1x3x224x224xf16>
    return %1 : tensor<1x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x3x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x3x7x7xf32>) outs(%0 : tensor<64x3x7x7xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
  }
  func.func private @BatchNormTrainingOp2(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf16>
    return %1, %batch_mean, %batch_var : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @Unknown3(%arg0: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x112x112xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x64x112x112xf16>) outs(%0 : tensor<1x64x112x112xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x64x112x112xf16>
    return %1 : tensor<1x64x112x112xf16>
  }
  func.func private @Unknown4(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf32>) outs(%0 : tensor<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp5(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %1, %batch_mean, %batch_var : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @Unknown6(%arg0: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown7(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf32>) outs(%0 : tensor<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp8(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %1, %batch_mean, %batch_var : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @Unknown9(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown10(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf32>) outs(%0 : tensor<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp11(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %1, %batch_mean, %batch_var : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @Unknown12(%arg0: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown13(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf32>) outs(%0 : tensor<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp14(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %1, %batch_mean, %batch_var : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @Unknown15(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown16(%arg0: tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x1x1xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x1x1xf32>) outs(%0 : tensor<128x64x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func.func private @BatchNormTrainingOp17(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %1, %batch_mean, %batch_var : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @Unknown18(%arg0: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x3x3xf32>) outs(%0 : tensor<128x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp19(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %1, %batch_mean, %batch_var : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @Unknown20(%arg0: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x128x28x28xf16>) outs(%0 : tensor<1x128x28x28xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown21(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf32>) outs(%0 : tensor<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @BatchNormTrainingOp22(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %1, %batch_mean, %batch_var : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @Unknown23(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%0 : tensor<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown24(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf32>) outs(%0 : tensor<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @BatchNormTrainingOp25(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %1, %batch_mean, %batch_var : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @Unknown26(%arg0: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x128x28x28xf16>) outs(%0 : tensor<1x128x28x28xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown27(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf32>) outs(%0 : tensor<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @BatchNormTrainingOp28(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %1, %batch_mean, %batch_var : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @Unknown29(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%0 : tensor<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown30(%arg0: tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x1x1xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x1x1xf32>) outs(%0 : tensor<256x128x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func.func private @BatchNormTrainingOp31(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %1, %batch_mean, %batch_var : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @Unknown32(%arg0: tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x3x3xf32>) outs(%0 : tensor<256x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func.func private @BatchNormTrainingOp33(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %1, %batch_mean, %batch_var : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @Unknown34(%arg0: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x256x14x14xf16>) outs(%0 : tensor<1x256x14x14xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown35(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf32>) outs(%0 : tensor<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @BatchNormTrainingOp36(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %1, %batch_mean, %batch_var : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @Unknown37(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%0 : tensor<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown38(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf32>) outs(%0 : tensor<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @BatchNormTrainingOp39(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %1, %batch_mean, %batch_var : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @Unknown40(%arg0: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x256x14x14xf16>) outs(%0 : tensor<1x256x14x14xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown41(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf32>) outs(%0 : tensor<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @BatchNormTrainingOp42(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %1, %batch_mean, %batch_var : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @Unknown43(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%0 : tensor<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown44(%arg0: tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x1x1xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x1x1xf32>) outs(%0 : tensor<512x256x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func.func private @BatchNormTrainingOp45(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %1, %batch_mean, %batch_var : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @Unknown46(%arg0: tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x3x3xf32>) outs(%0 : tensor<512x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func.func private @BatchNormTrainingOp47(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %1, %batch_mean, %batch_var : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @Unknown48(%arg0: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x512x7x7xf16>) outs(%0 : tensor<1x512x7x7xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown49(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf32>) outs(%0 : tensor<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @BatchNormTrainingOp50(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %1, %batch_mean, %batch_var : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @Unknown51(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) outs(%0 : tensor<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown52(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf32>) outs(%0 : tensor<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @BatchNormTrainingOp53(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %1, %batch_mean, %batch_var : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @Unknown54(%arg0: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x512x7x7xf16>) outs(%0 : tensor<1x512x7x7xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maxf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown55(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf32>) outs(%0 : tensor<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @BatchNormTrainingOp56(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %1, %batch_mean, %batch_var : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @Unknown57(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) outs(%0 : tensor<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown58(%arg0: tensor<1x512xf16>) -> tensor<1x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %0 = tensor.empty() : tensor<1x512xf16>
    %1 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf16>) outs(%0 : tensor<1x512xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.mulf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<1x512xf16>
    return %1 : tensor<1x512xf16>
  }
  func.func private @Unknown59(%arg0: tensor<1000x512xf32>) -> tensor<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1000x512xf16>
    %1 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1000x512xf32>) outs(%0 : tensor<1000x512xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<1000x512xf16>
    return %1 : tensor<1000x512xf16>
  }
  func.func private @Unknown60(%arg0: tensor<1000xf32>, %arg1: tensor<1x1000xf16>) -> tensor<1x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %expanded = tensor.expand_shape %arg0 [[0, 1]] : tensor<1000xf32> into tensor<1x1000xf32>
    %0 = tensor.empty() : tensor<1x1000xf16>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %expanded : tensor<1x1000xf16>, tensor<1x1000xf32>) outs(%0 : tensor<1x1000xf16>) {
    ^bb0(%in: f16, %in_0: f32, %out: f16):
      %2 = arith.truncf %in_0 : f32 to f16
      %3 = arith.addf %in, %2 : f16
      linalg.yield %3 : f16
    } -> tensor<1x1000xf16>
    return %1 : tensor<1x1000xf16>
  }
  func.func private @Unknown61(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown62(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown63(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown64(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown65(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown66(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown67(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown68(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown69(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown70(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown71(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown72(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown73(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown74(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown75(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown76(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown77(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown78(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown79(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown80(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown81(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown82(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown83(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown84(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown85(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown86(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown87(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown88(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown89(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown90(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown91(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown92(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown93(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown94(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown95(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown96(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown97(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown98(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown99(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func private @Unknown100(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.mulf %in_1, %cst : f32
      %3 = arith.mulf %in, %cst_0 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<1000xf32>, %arg4: tensor<1000x512xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64x64x3x3xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64x64x3x3xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x64x3x3xf32>, %arg22: tensor<128x128x3x3xf32>, %arg23: tensor<128x64x1x1xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128x128x3x3xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x128x3x3xf32>, %arg37: tensor<256x256x3x3xf32>, %arg38: tensor<256x128x1x1xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<256xf32>, %arg44: tensor<256xf32>, %arg45: tensor<256x256x3x3xf32>, %arg46: tensor<256x256x3x3xf32>, %arg47: tensor<512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<512x256x3x3xf32>, %arg52: tensor<512x512x3x3xf32>, %arg53: tensor<512x256x1x1xf32>, %arg54: tensor<512xf32>, %arg55: tensor<512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<512xf32>, %arg59: tensor<512xf32>, %arg60: tensor<512x512x3x3xf32>, %arg61: tensor<512x512x3x3xf32>, %arg62: tensor<i64>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<i64>, %arg66: tensor<64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<i64>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<i64>, %arg72: tensor<64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<i64>, %arg75: tensor<64xf32>, %arg76: tensor<64xf32>, %arg77: tensor<i64>, %arg78: tensor<128xf32>, %arg79: tensor<128xf32>, %arg80: tensor<i64>, %arg81: tensor<128xf32>, %arg82: tensor<128xf32>, %arg83: tensor<i64>, %arg84: tensor<128xf32>, %arg85: tensor<128xf32>, %arg86: tensor<i64>, %arg87: tensor<128xf32>, %arg88: tensor<128xf32>, %arg89: tensor<i64>, %arg90: tensor<128xf32>, %arg91: tensor<128xf32>, %arg92: tensor<i64>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<i64>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<i64>, %arg99: tensor<256xf32>, %arg100: tensor<256xf32>, %arg101: tensor<i64>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<i64>, %arg105: tensor<256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<i64>, %arg108: tensor<512xf32>, %arg109: tensor<512xf32>, %arg110: tensor<i64>, %arg111: tensor<512xf32>, %arg112: tensor<512xf32>, %arg113: tensor<i64>, %arg114: tensor<512xf32>, %arg115: tensor<512xf32>, %arg116: tensor<i64>, %arg117: tensor<512xf32>, %arg118: tensor<512xf32>, %arg119: tensor<i64>, %arg120: tensor<512xf32>, %arg121: tensor<512xf32>, %arg122: tensor<1x3x224x224xf32>) -> (tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.constant dense<0xFC00> : tensor<f16>
    %2 = call @Unknown0(%arg122) : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16>
    %3 = call @Unknown1(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %4 = mhlo.convolution(%2, %3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<1x64x112x112xf16>
    %5:3 = call @BatchNormTrainingOp2(%4, %arg1, %arg0) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %6 = call @Unknown3(%5#0) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %7 = "mhlo.reduce_window"(%6, %1) ({
    ^bb0(%arg123: tensor<f16>, %arg124: tensor<f16>):
      %127 = mhlo.maximum %arg123, %arg124 : tensor<f16>
      mhlo.return %127 : tensor<f16>
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x64x112x112xf16>, tensor<f16>) -> tensor<1x64x56x56xf16>
    %8 = call @Unknown4(%arg9) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %9 = mhlo.convolution(%7, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %10:3 = call @BatchNormTrainingOp5(%9, %arg6, %arg5) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %11 = call @Unknown6(%10#0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %12 = call @Unknown7(%arg10) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %13 = mhlo.convolution(%11, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %14:3 = call @BatchNormTrainingOp8(%13, %arg8, %arg7) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %15 = call @Unknown9(%14#0, %7) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %16 = call @Unknown10(%arg15) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %17 = mhlo.convolution(%15, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %18:3 = call @BatchNormTrainingOp11(%17, %arg12, %arg11) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %19 = call @Unknown12(%18#0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %20 = call @Unknown13(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %21 = mhlo.convolution(%19, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %22:3 = call @BatchNormTrainingOp14(%21, %arg14, %arg13) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %23 = call @Unknown15(%22#0, %15) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %24 = call @Unknown16(%arg23) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %25 = mhlo.convolution(%23, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<1x128x28x28xf16>
    %26:3 = call @BatchNormTrainingOp17(%25, %arg25, %arg24) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %27 = call @Unknown18(%arg21) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %28 = mhlo.convolution(%23, %27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<1x128x28x28xf16>
    %29:3 = call @BatchNormTrainingOp19(%28, %arg18, %arg17) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %30 = call @Unknown20(%29#0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %31 = call @Unknown21(%arg22) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %32 = mhlo.convolution(%30, %31) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %33:3 = call @BatchNormTrainingOp22(%32, %arg20, %arg19) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %34 = call @Unknown23(%33#0, %26#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %35 = call @Unknown24(%arg30) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %36 = mhlo.convolution(%34, %35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %37:3 = call @BatchNormTrainingOp25(%36, %arg27, %arg26) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %38 = call @Unknown26(%37#0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %39 = call @Unknown27(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %40 = mhlo.convolution(%38, %39) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %41:3 = call @BatchNormTrainingOp28(%40, %arg29, %arg28) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %42 = call @Unknown29(%41#0, %34) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %43 = call @Unknown30(%arg38) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %44 = mhlo.convolution(%42, %43) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<1x256x14x14xf16>
    %45:3 = call @BatchNormTrainingOp31(%44, %arg40, %arg39) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %46 = call @Unknown32(%arg36) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %47 = mhlo.convolution(%42, %46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<1x256x14x14xf16>
    %48:3 = call @BatchNormTrainingOp33(%47, %arg33, %arg32) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %49 = call @Unknown34(%48#0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %50 = call @Unknown35(%arg37) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %51 = mhlo.convolution(%49, %50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %52:3 = call @BatchNormTrainingOp36(%51, %arg35, %arg34) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %53 = call @Unknown37(%52#0, %45#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %54 = call @Unknown38(%arg45) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %55 = mhlo.convolution(%53, %54) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %56:3 = call @BatchNormTrainingOp39(%55, %arg42, %arg41) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %57 = call @Unknown40(%56#0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %58 = call @Unknown41(%arg46) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %59 = mhlo.convolution(%57, %58) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %60:3 = call @BatchNormTrainingOp42(%59, %arg44, %arg43) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %61 = call @Unknown43(%60#0, %53) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %62 = call @Unknown44(%arg53) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %63 = mhlo.convolution(%61, %62) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<1x512x7x7xf16>
    %64:3 = call @BatchNormTrainingOp45(%63, %arg55, %arg54) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %65 = call @Unknown46(%arg51) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %66 = mhlo.convolution(%61, %65) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<1x512x7x7xf16>
    %67:3 = call @BatchNormTrainingOp47(%66, %arg48, %arg47) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %68 = call @Unknown48(%67#0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %69 = call @Unknown49(%arg52) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %70 = mhlo.convolution(%68, %69) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %71:3 = call @BatchNormTrainingOp50(%70, %arg50, %arg49) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %72 = call @Unknown51(%71#0, %64#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %73 = call @Unknown52(%arg60) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %74 = mhlo.convolution(%72, %73) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %75:3 = call @BatchNormTrainingOp53(%74, %arg57, %arg56) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %76 = call @Unknown54(%75#0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %77 = call @Unknown55(%arg61) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %78 = mhlo.convolution(%76, %77) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %79:3 = call @BatchNormTrainingOp56(%78, %arg59, %arg58) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %80 = call @Unknown57(%79#0, %72) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %81 = mhlo.reduce(%80 init: %0) across dimensions = [3, 2] : (tensor<1x512x7x7xf16>, tensor<f16>) -> tensor<1x512xf16>
     reducer(%arg123: tensor<f16>, %arg124: tensor<f16>)  {
      %127 = mhlo.add %arg123, %arg124 : tensor<f16>
      mhlo.return %127 : tensor<f16>
    }
    %82 = call @Unknown58(%81) : (tensor<1x512xf16>) -> tensor<1x512xf16>
    %83 = call @Unknown59(%arg4) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %84 = "mhlo.transpose"(%83) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    %85 = "mhlo.dot_general"(%82, %83) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512xf16>, tensor<1000x512xf16>) -> tensor<1x1000xf16>
    %86 = call @Unknown60(%arg3, %85) : (tensor<1000xf32>, tensor<1x1000xf16>) -> tensor<1x1000xf16>
    %87 = call @Unknown61(%5#1, %arg63) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %88 = call @Unknown62(%5#2, %arg64) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %89 = call @Unknown63(%10#1, %arg66) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %90 = call @Unknown64(%10#2, %arg67) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %91 = call @Unknown65(%14#1, %arg69) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %92 = call @Unknown66(%14#2, %arg70) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %93 = call @Unknown67(%18#1, %arg72) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %94 = call @Unknown68(%18#2, %arg73) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %95 = call @Unknown69(%22#1, %arg75) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %96 = call @Unknown70(%22#2, %arg76) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %97 = call @Unknown71(%29#1, %arg78) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %98 = call @Unknown72(%29#2, %arg79) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %99 = call @Unknown73(%33#1, %arg81) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %100 = call @Unknown74(%33#2, %arg82) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %101 = call @Unknown75(%26#1, %arg84) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %102 = call @Unknown76(%26#2, %arg85) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %103 = call @Unknown77(%37#1, %arg87) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %104 = call @Unknown78(%37#2, %arg88) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %105 = call @Unknown79(%41#1, %arg90) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %106 = call @Unknown80(%41#2, %arg91) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %107 = call @Unknown81(%48#1, %arg93) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %108 = call @Unknown82(%48#2, %arg94) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %109 = call @Unknown83(%52#1, %arg96) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %110 = call @Unknown84(%52#2, %arg97) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %111 = call @Unknown85(%45#1, %arg99) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %112 = call @Unknown86(%45#2, %arg100) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %113 = call @Unknown87(%56#1, %arg102) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %114 = call @Unknown88(%56#2, %arg103) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %115 = call @Unknown89(%60#1, %arg105) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %116 = call @Unknown90(%60#2, %arg106) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %117 = call @Unknown91(%67#1, %arg108) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %118 = call @Unknown92(%67#2, %arg109) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %119 = call @Unknown93(%71#1, %arg111) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %120 = call @Unknown94(%71#2, %arg112) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %121 = call @Unknown95(%64#1, %arg114) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %122 = call @Unknown96(%64#2, %arg115) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %123 = call @Unknown97(%75#1, %arg117) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %124 = call @Unknown98(%75#2, %arg118) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %125 = call @Unknown99(%79#1, %arg120) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %126 = call @Unknown100(%79#2, %arg121) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    return %86, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %3, %2, %4, %6, %7, %8, %9, %11, %12, %13, %15, %16, %17, %19, %20, %21, %23, %27, %28, %30, %31, %32, %24, %25, %34, %35, %36, %38, %39, %40, %42, %46, %47, %49, %50, %51, %43, %44, %53, %54, %55, %57, %58, %59, %61, %65, %66, %68, %69, %70, %62, %63, %72, %73, %74, %76, %77, %78, %80, %82, %84 : tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>
  }
}