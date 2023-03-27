// RUN: byteir-opt %s -byteir-bufferize-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map5 = affine_map<() -> ()>
#map6 = affine_map<(d0) -> (d0)>
module @IrToMhlo.2452 {
  func.func private @Unknown0(%arg0: tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<4x3x224x224xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x3x224x224xf32>) outs(%0 : tensor<4x3x224x224xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<4x3x224x224xf16>
    return %1 : tensor<4x3x224x224xf16>
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
  func.func private @BatchNormTrainingOp2(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x112x112xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    return %1 : tensor<4x64x112x112xf16>
  }
  func.func private @Unknown3(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf32>) outs(%0 : tensor<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
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
  func.func private @Unknown5(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf32>) outs(%0 : tensor<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown6(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf32>) outs(%0 : tensor<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown7(%arg0: tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x1x1xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x1x1xf32>) outs(%0 : tensor<128x64x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func.func private @Unknown8(%arg0: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x3x3xf32>) outs(%0 : tensor<128x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func.func private @Unknown9(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf32>) outs(%0 : tensor<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown10(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf32>) outs(%0 : tensor<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown11(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf32>) outs(%0 : tensor<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown12(%arg0: tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x1x1xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x1x1xf32>) outs(%0 : tensor<256x128x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func.func private @Unknown13(%arg0: tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x3x3xf32>) outs(%0 : tensor<256x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func.func private @Unknown14(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf32>) outs(%0 : tensor<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown15(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf32>) outs(%0 : tensor<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown16(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf32>) outs(%0 : tensor<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown17(%arg0: tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x1x1xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x1x1xf32>) outs(%0 : tensor<512x256x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func.func private @Unknown18(%arg0: tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x3x3xf32>) outs(%0 : tensor<512x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func.func private @Unknown19(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf32>) outs(%0 : tensor<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown20(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf32>) outs(%0 : tensor<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown21(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf32>) outs(%0 : tensor<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown22(%arg0: tensor<4x1000xf32>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -2.500000e-01 : f32
    %0 = tensor.empty() : tensor<4x1000xf16>
    %1 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x1000xf32>) outs(%0 : tensor<4x1000xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.mulf %in, %cst : f32
      %3 = arith.truncf %2 : f32 to f16
      linalg.yield %3 : f16
    } -> tensor<4x1000xf16>
    return %1 : tensor<4x1000xf16>
  }
  func.func private @Unknown23(%arg0: tensor<1000x512xf32>) -> tensor<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1000x512xf16>
    %1 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1000x512xf32>) outs(%0 : tensor<1000x512xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<1000x512xf16>
    return %1 : tensor<1000x512xf16>
  }
  func.func private @Unknown24(%arg0: tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x112x112xf16>
    %1 = tensor.empty() : tensor<4x64x112x112xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x64x112x112xf16>) outs(%0, %1 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>)
    return %2#0, %2#1 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>
  }
  func.func private @BatchNormTrainingOp25(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown26(%arg0: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = tensor.empty() : tensor<4x64x56x56xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x64x56x56xf16>) outs(%0, %1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    return %2#0, %2#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp27(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown28(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = tensor.empty() : tensor<4x64x56x56xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%0, %1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: i1):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.maxf %3, %cst : f16
      %5 = arith.cmpf ogt, %4, %cst : f16
      linalg.yield %4, %5 : f16, i1
    } -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    return %2#0, %2#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp29(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown30(%arg0: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = tensor.empty() : tensor<4x64x56x56xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x64x56x56xf16>) outs(%0, %1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    return %2#0, %2#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp31(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown32(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = tensor.empty() : tensor<4x64x56x56xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%0, %1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: i1):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.maxf %3, %cst : f16
      %5 = arith.cmpf ogt, %4, %cst : f16
      linalg.yield %4, %5 : f16, i1
    } -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    return %2#0, %2#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp33(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormTrainingOp34(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown35(%arg0: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = tensor.empty() : tensor<4x128x28x28xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x128x28x28xf16>) outs(%0, %1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    return %2#0, %2#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp36(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown37(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = tensor.empty() : tensor<4x128x28x28xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%0, %1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: i1):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.maxf %3, %cst : f16
      %5 = arith.cmpf ogt, %4, %cst : f16
      linalg.yield %4, %5 : f16, i1
    } -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    return %2#0, %2#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp38(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown39(%arg0: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = tensor.empty() : tensor<4x128x28x28xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x128x28x28xf16>) outs(%0, %1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    return %2#0, %2#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp40(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown41(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = tensor.empty() : tensor<4x128x28x28xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%0, %1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: i1):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.maxf %3, %cst : f16
      %5 = arith.cmpf ogt, %4, %cst : f16
      linalg.yield %4, %5 : f16, i1
    } -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    return %2#0, %2#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp42(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormTrainingOp43(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown44(%arg0: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = tensor.empty() : tensor<4x256x14x14xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x256x14x14xf16>) outs(%0, %1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    return %2#0, %2#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp45(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown46(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = tensor.empty() : tensor<4x256x14x14xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%0, %1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: i1):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.maxf %3, %cst : f16
      %5 = arith.cmpf ogt, %4, %cst : f16
      linalg.yield %4, %5 : f16, i1
    } -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    return %2#0, %2#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp47(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown48(%arg0: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = tensor.empty() : tensor<4x256x14x14xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x256x14x14xf16>) outs(%0, %1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    return %2#0, %2#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp49(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown50(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = tensor.empty() : tensor<4x256x14x14xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%0, %1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: i1):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.maxf %3, %cst : f16
      %5 = arith.cmpf ogt, %4, %cst : f16
      linalg.yield %4, %5 : f16, i1
    } -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    return %2#0, %2#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp51(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormTrainingOp52(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown53(%arg0: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = tensor.empty() : tensor<4x512x7x7xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x512x7x7xf16>) outs(%0, %1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    return %2#0, %2#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp54(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown55(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = tensor.empty() : tensor<4x512x7x7xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%0, %1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: i1):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.maxf %3, %cst : f16
      %5 = arith.cmpf ogt, %4, %cst : f16
      linalg.yield %4, %5 : f16, i1
    } -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    return %2#0, %2#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp56(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown57(%arg0: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = tensor.empty() : tensor<4x512x7x7xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x512x7x7xf16>) outs(%0, %1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
    ^bb0(%in: f16, %out: f16, %out_0: i1):
      %3 = arith.maxf %in, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %3, %4 : f16, i1
    } -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    return %2#0, %2#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp58(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown59(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = tensor.empty() : tensor<4x512x7x7xi1>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%0, %1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: i1):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.maxf %3, %cst : f16
      %5 = arith.cmpf ogt, %4, %cst : f16
      linalg.yield %4, %5 : f16, i1
    } -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    return %2#0, %2#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @Unknown60(%arg0: tensor<4x512xf16>) -> tensor<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %0 = tensor.empty() : tensor<4x512xf16>
    %1 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x512xf16>) outs(%0 : tensor<4x512xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.mulf %in, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x512xf16>
    return %1 : tensor<4x512xf16>
  }
  func.func private @Unknown61(%arg0: tensor<1000xf32>, %arg1: tensor<4x1000xf16>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<4x1000xf16>
    %1 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : tensor<4x1000xf16>, tensor<1000xf32>) outs(%0 : tensor<4x1000xf16>) {
    ^bb0(%in: f16, %in_0: f32, %out: f16):
      %2 = arith.truncf %in_0 : f32 to f16
      %3 = arith.addf %in, %2 : f16
      linalg.yield %3 : f16
    } -> tensor<4x1000xf16>
    return %1 : tensor<4x1000xf16>
  }
  func.func private @Unknown62(%arg0: tensor<4xf16>, %arg1: tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<4x1000xf16>
    %1:2 = linalg.generic {indexing_maps = [#map1, #map3, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : tensor<4x1000xf16>, tensor<4xf16>) outs(%0, %0 : tensor<4x1000xf16>, tensor<4x1000xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: f16):
      %2 = arith.subf %in, %in_0 : f16
      %3 = math.exp %2 : f16
      linalg.yield %2, %3 : f16, f16
    } -> (tensor<4x1000xf16>, tensor<4x1000xf16>)
    return %1#0, %1#1 : tensor<4x1000xf16>, tensor<4x1000xf16>
  }
  func.func private @Unknown63(%arg0: tensor<4xf16>, %arg1: tensor<4x1000xf16>, %arg2: tensor<4xf16>, %arg3: tensor<4x1000xf16>, %arg4: tensor<4x1000xf32>) -> (tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<4x1000xf16>
    %1 = tensor.empty() : tensor<4x1000xf32>
    %2:3 = linalg.generic {indexing_maps = [#map1, #map1, #map3, #map3, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg1, %arg0, %arg2, %arg4 : tensor<4x1000xf16>, tensor<4x1000xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4x1000xf32>) outs(%0, %1, %1 : tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %in_3: f32, %out: f16, %out_4: f32, %out_5: f32):
      %3 = math.log %in_1 : f16
      %4 = arith.subf %in_0, %3 : f16
      %5 = math.exp %4 : f16
      %6 = arith.mulf %5, %in_2 : f16
      %7 = arith.subf %in, %6 : f16
      %8 = arith.extf %4 : f16 to f32
      %9 = arith.mulf %8, %in_3 : f32
      %10 = arith.extf %7 : f16 to f32
      linalg.yield %7, %9, %10 : f16, f32, f32
    } -> (tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>)
    return %2#0, %2#1, %2#2 : tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>
  }
  func.func private @Unknown64(%arg0: tensor<4x512xf16>, %arg1: tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 4.900000e+01 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg0 : tensor<4x512x7x7xi1>, tensor<4x512xf16>) outs(%0 : tensor<4x512x7x7xf16>) {
    ^bb0(%in: i1, %in_1: f16, %out: f16):
      %2 = arith.divf %in_1, %cst_0 : f16
      %3 = arith.select %in, %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp65(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp66(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp67(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown68(%arg0: tensor<4x512x7x7xi1>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) outs(%0 : tensor<4x512x7x7xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp69(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp70(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp71(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown72(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%0 : tensor<4x512x7x7xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.select %in, %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp73(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp74(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp75(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown76(%arg0: tensor<4x512x7x7xi1>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) outs(%0 : tensor<4x512x7x7xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp77(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp78(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp79(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func.func private @BatchNormGradOp80(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp81(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp82(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func.func private @Unknown83(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%0 : tensor<4x256x14x14xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.select %in, %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp84(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp85(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp86(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown87(%arg0: tensor<4x256x14x14xi1>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) outs(%0 : tensor<4x256x14x14xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp88(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp89(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp90(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown91(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%0 : tensor<4x256x14x14xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.select %in, %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp92(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp93(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp94(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown95(%arg0: tensor<4x256x14x14xi1>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) outs(%0 : tensor<4x256x14x14xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp96(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp97(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp98(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func.func private @BatchNormGradOp99(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp100(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp101(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func.func private @Unknown102(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%0 : tensor<4x128x28x28xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.select %in, %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp103(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp104(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp105(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown106(%arg0: tensor<4x128x28x28xi1>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) outs(%0 : tensor<4x128x28x28xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp107(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp108(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp109(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown110(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%0 : tensor<4x128x28x28xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.select %in, %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp111(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp112(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp113(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown114(%arg0: tensor<4x128x28x28xi1>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) outs(%0 : tensor<4x128x28x28xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp115(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp116(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp117(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func.func private @BatchNormGradOp118(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp119(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp120(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func.func private @Unknown121(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%0 : tensor<4x64x56x56xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.select %in, %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp122(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp123(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp124(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown125(%arg0: tensor<4x64x56x56xi1>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) outs(%0 : tensor<4x64x56x56xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp126(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp127(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp128(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown129(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%0 : tensor<4x64x56x56xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.select %in, %2, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp130(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp131(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp132(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown133(%arg0: tensor<4x64x56x56xi1>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) outs(%0 : tensor<4x64x56x56xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp134(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp135(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp136(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown137(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%0 : tensor<4x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      linalg.yield %2 : f16
    } -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown138(%arg0: tensor<4x64x112x112xi1>, %arg1: tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x112x112xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>) outs(%0 : tensor<4x64x112x112xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %2 = arith.select %in, %in_0, %cst : f16
      linalg.yield %2 : f16
    } -> tensor<4x64x112x112xf16>
    return %1 : tensor<4x64x112x112xf16>
  }
  func.func private @BatchNormGradOp139(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x112x112xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardFilterOp140(%arg0: tensor<4x3x224x224xf16>, %arg1: tensor<4x64x112x112xf16>) -> tensor<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
  }
  func.func private @Unknown141(%arg0: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = []} ins(%arg0 : tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.negf %in : f32
      %3 = arith.divf %2, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @Unknown142(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x3x7x7xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x3x7x7xf16>) outs(%0 : tensor<64x3x7x7xf32>) attrs =  {xla_shape = "f32[64,3,7,7]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x3x7x7xf32>
    return %1 : tensor<64x3x7x7xf32>
  }
  func.func private @Unknown143(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf16>) outs(%0 : tensor<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x64x3x3xf32>
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown144(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf16>) outs(%0 : tensor<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x64x3x3xf32>
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown145(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf16>) outs(%0 : tensor<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x64x3x3xf32>
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown146(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf16>) outs(%0 : tensor<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x64x3x3xf32>
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown147(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x3x3xf16>) outs(%0 : tensor<128x64x3x3xf32>) attrs =  {xla_shape = "f32[128,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x64x3x3xf32>
    return %1 : tensor<128x64x3x3xf32>
  }
  func.func private @Unknown148(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf16>) outs(%0 : tensor<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x128x3x3xf32>
    return %1 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown149(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x1x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x1x1xf16>) outs(%0 : tensor<128x64x1x1xf32>) attrs =  {xla_shape = "f32[128,64,1,1]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x64x1x1xf32>
    return %1 : tensor<128x64x1x1xf32>
  }
  func.func private @Unknown150(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf16>) outs(%0 : tensor<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x128x3x3xf32>
    return %1 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown151(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf16>) outs(%0 : tensor<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x128x3x3xf32>
    return %1 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown152(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x3x3xf16>) outs(%0 : tensor<256x128x3x3xf32>) attrs =  {xla_shape = "f32[256,128,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x128x3x3xf32>
    return %1 : tensor<256x128x3x3xf32>
  }
  func.func private @Unknown153(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf16>) outs(%0 : tensor<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x256x3x3xf32>
    return %1 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown154(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x1x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x1x1xf16>) outs(%0 : tensor<256x128x1x1xf32>) attrs =  {xla_shape = "f32[256,128,1,1]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x128x1x1xf32>
    return %1 : tensor<256x128x1x1xf32>
  }
  func.func private @Unknown155(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf16>) outs(%0 : tensor<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x256x3x3xf32>
    return %1 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown156(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf16>) outs(%0 : tensor<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x256x3x3xf32>
    return %1 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown157(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x3x3xf16>) outs(%0 : tensor<512x256x3x3xf32>) attrs =  {xla_shape = "f32[512,256,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x256x3x3xf32>
    return %1 : tensor<512x256x3x3xf32>
  }
  func.func private @Unknown158(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf16>) outs(%0 : tensor<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x512x3x3xf32>
    return %1 : tensor<512x512x3x3xf32>
  }
  func.func private @Unknown159(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x1x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x1x1xf16>) outs(%0 : tensor<512x256x1x1xf32>) attrs =  {xla_shape = "f32[512,256,1,1]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x256x1x1xf32>
    return %1 : tensor<512x256x1x1xf32>
  }
  func.func private @Unknown160(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf16>) outs(%0 : tensor<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x512x3x3xf32>
    return %1 : tensor<512x512x3x3xf32>
  }
  func.func private @Unknown161(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf16>) outs(%0 : tensor<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x512x3x3xf32>
    return %1 : tensor<512x512x3x3xf32>
  }
  func.func private @MatmulOp162(%arg0: tensor<4x512xf16>, %arg1: tensor<4x1000xf16>) -> tensor<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<4x1000xf16>) -> tensor<512x1000xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %1 : tensor<1000x512xf16>
  }
  func.func private @Unknown163(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1000x512xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1000x512xf16>) outs(%0 : tensor<1000x512xf32>) attrs =  {xla_shape = "f32[1000,512]{0,1}"} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<1000x512xf32>
    return %1 : tensor<1000x512xf32>
  }
  func.func private @Unknown164(%arg0: tensor<1000xf32>) -> tensor<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1000xf32>
    %1 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel"]} ins(%arg0 : tensor<1000xf32>) outs(%0 : tensor<1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.truncf %in : f32 to f16
      %3 = arith.extf %2 : f16 to f32
      linalg.yield %3 : f32
    } -> tensor<1000xf32>
    return %1 : tensor<1000xf32>
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
    %28 = mhlo.reduce(%26 init: %1) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %198 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %198 : tensor<f16>
    }
    %29:2 = call @Unknown24(%6) : (tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>)
    %30 = "mhlo.reduce_window"(%29#0, %2) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %198 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      mhlo.return %198 : tensor<f16>
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<f16>) -> tensor<4x64x56x56xf16>
    %31 = mhlo.convolution(%30, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %32 = call @BatchNormTrainingOp25(%31, %arg8, %arg9) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %33:2 = call @Unknown26(%32) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %34 = mhlo.convolution(%33#0, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %35 = call @BatchNormTrainingOp27(%34, %arg13, %arg14) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %36:2 = call @Unknown28(%35, %30) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %37 = mhlo.convolution(%36#0, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %38 = call @BatchNormTrainingOp29(%37, %arg18, %arg19) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %39:2 = call @Unknown30(%38) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %40 = mhlo.convolution(%39#0, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %41 = call @BatchNormTrainingOp31(%40, %arg23, %arg24) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %42:2 = call @Unknown32(%41, %36#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %43 = mhlo.convolution(%42#0, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<4x128x28x28xf16>
    %44 = call @BatchNormTrainingOp33(%43, %arg38, %arg39) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %45 = mhlo.convolution(%42#0, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<4x128x28x28xf16>
    %46 = call @BatchNormTrainingOp34(%45, %arg28, %arg29) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %47:2 = call @Unknown35(%46) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %48 = mhlo.convolution(%47#0, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %49 = call @BatchNormTrainingOp36(%48, %arg33, %arg34) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %50:2 = call @Unknown37(%49, %44) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %51 = mhlo.convolution(%50#0, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %52 = call @BatchNormTrainingOp38(%51, %arg43, %arg44) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %53:2 = call @Unknown39(%52) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %54 = mhlo.convolution(%53#0, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %55 = call @BatchNormTrainingOp40(%54, %arg48, %arg49) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %56:2 = call @Unknown41(%55, %50#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %57 = mhlo.convolution(%56#0, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<4x256x14x14xf16>
    %58 = call @BatchNormTrainingOp42(%57, %arg63, %arg64) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %59 = mhlo.convolution(%56#0, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<4x256x14x14xf16>
    %60 = call @BatchNormTrainingOp43(%59, %arg53, %arg54) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %61:2 = call @Unknown44(%60) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %62 = mhlo.convolution(%61#0, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %63 = call @BatchNormTrainingOp45(%62, %arg58, %arg59) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %64:2 = call @Unknown46(%63, %58) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %65 = mhlo.convolution(%64#0, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %66 = call @BatchNormTrainingOp47(%65, %arg68, %arg69) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %67:2 = call @Unknown48(%66) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %68 = mhlo.convolution(%67#0, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %69 = call @BatchNormTrainingOp49(%68, %arg73, %arg74) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %70:2 = call @Unknown50(%69, %64#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %71 = mhlo.convolution(%70#0, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<4x512x7x7xf16>
    %72 = call @BatchNormTrainingOp51(%71, %arg88, %arg89) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %73 = mhlo.convolution(%70#0, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<4x512x7x7xf16>
    %74 = call @BatchNormTrainingOp52(%73, %arg78, %arg79) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %75:2 = call @Unknown53(%74) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %76 = mhlo.convolution(%75#0, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %77 = call @BatchNormTrainingOp54(%76, %arg83, %arg84) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %78:2 = call @Unknown55(%77, %72) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %79 = mhlo.convolution(%78#0, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %80 = call @BatchNormTrainingOp56(%79, %arg93, %arg94) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %81:2 = call @Unknown57(%80) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %82 = mhlo.convolution(%81#0, %25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %83 = call @BatchNormTrainingOp58(%82, %arg98, %arg99) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %84:2 = call @Unknown59(%83, %78#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %85 = mhlo.reduce(%84#0 init: %1) across dimensions = [3, 2] : (tensor<4x512x7x7xf16>, tensor<f16>) -> tensor<4x512xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %198 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %198 : tensor<f16>
    }
    %86 = call @Unknown60(%85) : (tensor<4x512xf16>) -> tensor<4x512xf16>
    %87 = "mhlo.dot_general"(%86, %27) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<1000x512xf16>) -> tensor<4x1000xf16>
    %88 = call @Unknown61(%arg103, %87) : (tensor<1000xf32>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %89 = mhlo.reduce(%88 init: %2) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %198 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      mhlo.return %198 : tensor<f16>
    }
    %90:2 = call @Unknown62(%89, %88) : (tensor<4xf16>, tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>)
    %91 = mhlo.reduce(%90#1 init: %1) across dimensions = [1] : (tensor<4x1000xf16>, tensor<f16>) -> tensor<4xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %198 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %198 : tensor<f16>
    }
    %92:3 = call @Unknown63(%91, %90#0, %28, %26, %arg1) : (tensor<4xf16>, tensor<4x1000xf16>, tensor<4xf16>, tensor<4x1000xf16>, tensor<4x1000xf32>) -> (tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>)
    %93 = "mhlo.dot"(%92#0, %27) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x1000xf16>, tensor<1000x512xf16>) -> tensor<4x512xf16>
    %94 = call @Unknown64(%93, %84#1) : (tensor<4x512xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %95:3 = call @BatchNormGradOp65(%82, %arg98, %94) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %96 = call @ConvBackwardDataOp66(%95#0, %25) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %97 = call @ConvBackwardFilterOp67(%81#0, %95#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %98 = call @Unknown68(%81#1, %96) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %99:3 = call @BatchNormGradOp69(%79, %arg93, %98) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %100 = call @ConvBackwardDataOp70(%99#0, %24) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %101 = call @ConvBackwardFilterOp71(%78#0, %99#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %102 = call @Unknown72(%94, %100, %78#1) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %103:3 = call @BatchNormGradOp73(%76, %arg83, %102) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %104 = call @ConvBackwardDataOp74(%103#0, %23) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %105 = call @ConvBackwardFilterOp75(%75#0, %103#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %106 = call @Unknown76(%75#1, %104) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %107:3 = call @BatchNormGradOp77(%73, %arg78, %106) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %108 = call @ConvBackwardDataOp78(%107#0, %22) : (tensor<4x512x7x7xf16>, tensor<512x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %109 = call @ConvBackwardFilterOp79(%70#0, %107#0) : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<512x256x3x3xf16>
    %110:3 = call @BatchNormGradOp80(%71, %arg88, %102) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %111 = call @ConvBackwardDataOp81(%110#0, %21) : (tensor<4x512x7x7xf16>, tensor<512x256x1x1xf16>) -> tensor<4x256x14x14xf16>
    %112 = call @ConvBackwardFilterOp82(%70#0, %110#0) : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<512x256x1x1xf16>
    %113 = call @Unknown83(%111, %108, %70#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %114:3 = call @BatchNormGradOp84(%68, %arg73, %113) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %115 = call @ConvBackwardDataOp85(%114#0, %20) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %116 = call @ConvBackwardFilterOp86(%67#0, %114#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %117 = call @Unknown87(%67#1, %115) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %118:3 = call @BatchNormGradOp88(%65, %arg68, %117) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %119 = call @ConvBackwardDataOp89(%118#0, %19) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %120 = call @ConvBackwardFilterOp90(%64#0, %118#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %121 = call @Unknown91(%113, %119, %64#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %122:3 = call @BatchNormGradOp92(%62, %arg58, %121) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %123 = call @ConvBackwardDataOp93(%122#0, %18) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %124 = call @ConvBackwardFilterOp94(%61#0, %122#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %125 = call @Unknown95(%61#1, %123) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %126:3 = call @BatchNormGradOp96(%59, %arg53, %125) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %127 = call @ConvBackwardDataOp97(%126#0, %17) : (tensor<4x256x14x14xf16>, tensor<256x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %128 = call @ConvBackwardFilterOp98(%56#0, %126#0) : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<256x128x3x3xf16>
    %129:3 = call @BatchNormGradOp99(%57, %arg63, %121) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %130 = call @ConvBackwardDataOp100(%129#0, %16) : (tensor<4x256x14x14xf16>, tensor<256x128x1x1xf16>) -> tensor<4x128x28x28xf16>
    %131 = call @ConvBackwardFilterOp101(%56#0, %129#0) : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<256x128x1x1xf16>
    %132 = call @Unknown102(%130, %127, %56#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %133:3 = call @BatchNormGradOp103(%54, %arg48, %132) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %134 = call @ConvBackwardDataOp104(%133#0, %15) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %135 = call @ConvBackwardFilterOp105(%53#0, %133#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %136 = call @Unknown106(%53#1, %134) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %137:3 = call @BatchNormGradOp107(%51, %arg43, %136) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %138 = call @ConvBackwardDataOp108(%137#0, %14) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %139 = call @ConvBackwardFilterOp109(%50#0, %137#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %140 = call @Unknown110(%132, %138, %50#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %141:3 = call @BatchNormGradOp111(%48, %arg33, %140) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %142 = call @ConvBackwardDataOp112(%141#0, %13) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %143 = call @ConvBackwardFilterOp113(%47#0, %141#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %144 = call @Unknown114(%47#1, %142) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %145:3 = call @BatchNormGradOp115(%45, %arg28, %144) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %146 = call @ConvBackwardDataOp116(%145#0, %12) : (tensor<4x128x28x28xf16>, tensor<128x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %147 = call @ConvBackwardFilterOp117(%42#0, %145#0) : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<128x64x3x3xf16>
    %148:3 = call @BatchNormGradOp118(%43, %arg38, %140) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %149 = call @ConvBackwardDataOp119(%148#0, %11) : (tensor<4x128x28x28xf16>, tensor<128x64x1x1xf16>) -> tensor<4x64x56x56xf16>
    %150 = call @ConvBackwardFilterOp120(%42#0, %148#0) : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<128x64x1x1xf16>
    %151 = call @Unknown121(%149, %146, %42#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %152:3 = call @BatchNormGradOp122(%40, %arg23, %151) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %153 = call @ConvBackwardDataOp123(%152#0, %10) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %154 = call @ConvBackwardFilterOp124(%39#0, %152#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %155 = call @Unknown125(%39#1, %153) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %156:3 = call @BatchNormGradOp126(%37, %arg18, %155) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %157 = call @ConvBackwardDataOp127(%156#0, %9) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %158 = call @ConvBackwardFilterOp128(%36#0, %156#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %159 = call @Unknown129(%151, %157, %36#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %160:3 = call @BatchNormGradOp130(%34, %arg13, %159) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %161 = call @ConvBackwardDataOp131(%160#0, %8) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %162 = call @ConvBackwardFilterOp132(%33#0, %160#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %163 = call @Unknown133(%33#1, %161) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %164:3 = call @BatchNormGradOp134(%31, %arg8, %163) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %165 = call @ConvBackwardDataOp135(%164#0, %7) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %166 = call @ConvBackwardFilterOp136(%30, %164#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %167 = call @Unknown137(%159, %165) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %168 = "mhlo.select_and_scatter"(%29#0, %167, %1) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %198 = mhlo.compare  GE, %arg104, %arg105 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %198 : tensor<i1>
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %198 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %198 : tensor<f16>
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<4x64x112x112xf16>
    %169 = call @Unknown138(%29#1, %168) : (tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %170:3 = call @BatchNormGradOp139(%5, %arg3, %169) : (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %171 = call @ConvBackwardFilterOp140(%3, %170#0) : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<64x3x7x7xf16>
    %172 = mhlo.reduce(%92#1 init: %0) across dimensions = [0, 1] : (tensor<4x1000xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg104: tensor<f32>, %arg105: tensor<f32>)  {
      %198 = mhlo.add %arg104, %arg105 : tensor<f32>
      mhlo.return %198 : tensor<f32>
    }
    %173 = call @Unknown141(%172) : (tensor<f32>) -> tensor<f32>
    %174 = call @Unknown142(%171) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %175 = call @Unknown143(%166) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %176 = call @Unknown144(%162) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %177 = call @Unknown145(%158) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %178 = call @Unknown146(%154) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %179 = call @Unknown147(%147) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %180 = call @Unknown148(%143) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %181 = call @Unknown149(%150) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %182 = call @Unknown150(%139) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %183 = call @Unknown151(%135) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %184 = call @Unknown152(%128) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %185 = call @Unknown153(%124) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %186 = call @Unknown154(%131) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %187 = call @Unknown155(%120) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %188 = call @Unknown156(%116) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %189 = call @Unknown157(%109) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %190 = call @Unknown158(%105) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %191 = call @Unknown159(%112) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %192 = call @Unknown160(%101) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %193 = call @Unknown161(%97) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %194 = call @MatmulOp162(%86, %92#0) : (tensor<4x512xf16>, tensor<4x1000xf16>) -> tensor<1000x512xf16>
    %195 = call @Unknown163(%194) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %196 = mhlo.reduce(%92#2 init: %0) across dimensions = [0] : (tensor<4x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg104: tensor<f32>, %arg105: tensor<f32>)  {
      %198 = mhlo.add %arg104, %arg105 : tensor<f32>
      mhlo.return %198 : tensor<f32>
    }
    %197 = call @Unknown164(%196) : (tensor<1000xf32>) -> tensor<1000xf32>
    return %173, %174, %170#1, %170#2, %175, %164#1, %164#2, %176, %160#1, %160#2, %177, %156#1, %156#2, %178, %152#1, %152#2, %179, %145#1, %145#2, %180, %141#1, %141#2, %181, %148#1, %148#2, %182, %137#1, %137#2, %183, %133#1, %133#2, %184, %126#1, %126#2, %185, %122#1, %122#2, %186, %129#1, %129#2, %187, %118#1, %118#2, %188, %114#1, %114#2, %189, %107#1, %107#2, %190, %103#1, %103#2, %191, %110#1, %110#2, %192, %99#1, %99#2, %193, %95#1, %95#2, %195, %197 : tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>
  }
}