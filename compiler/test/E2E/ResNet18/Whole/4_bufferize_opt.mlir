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
  func.func @main(%arg0: tensor<4x3x224x224xf32>, %arg1: tensor<4x1000xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64x64x3x3xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64x64x3x3xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64x64x3x3xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64xf32>, %arg22: tensor<64x64x3x3xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<128x64x3x3xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128xf32>, %arg32: tensor<128x128x3x3xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128x64x1x1xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128x3x3xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128x128x3x3xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<256x128x3x3xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256xf32>, %arg57: tensor<256x256x3x3xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256x128x1x1xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256x256x3x3xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<256x256x3x3xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<256xf32>, %arg77: tensor<512x256x3x3xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512xf32>, %arg82: tensor<512x512x3x3xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512x256x1x1xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512xf32>, %arg91: tensor<512xf32>, %arg92: tensor<512x512x3x3xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512xf32>, %arg97: tensor<512x512x3x3xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<512xf32>, %arg102: tensor<1000x512xf32>, %arg103: tensor<1000xf32>) -> (tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg0) : (tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16>
    %1 = call @Unknown1(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %2 = byre.compute @ConvOp_f16f16_f16(%0, %1) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16> -> tensor<4x64x112x112xf16>
    %3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%2, %arg3, %arg4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<4x64x112x112xf16>
    %4 = call @Unknown3(%arg7) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %5 = call @Unknown4(%arg12) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %6 = call @Unknown5(%arg17) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %7 = call @Unknown6(%arg22) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %8 = call @Unknown7(%arg37) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %9 = call @Unknown8(%arg27) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %10 = call @Unknown9(%arg32) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %11 = call @Unknown10(%arg42) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %12 = call @Unknown11(%arg47) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %13 = call @Unknown12(%arg62) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %14 = call @Unknown13(%arg52) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %15 = call @Unknown14(%arg57) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %16 = call @Unknown15(%arg67) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %17 = call @Unknown16(%arg72) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %18 = call @Unknown17(%arg87) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %19 = call @Unknown18(%arg77) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %20 = call @Unknown19(%arg82) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %21 = call @Unknown20(%arg92) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %22 = call @Unknown21(%arg97) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %23 = call @Unknown22(%arg1) : (tensor<4x1000xf32>) -> tensor<4x1000xf16>
    %24 = call @Unknown23(%arg102) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %25 = byre.compute @ReduceSumOp_f16_f16(%23) {dimensions = dense<1> : tensor<1xi64>} : tensor<4x1000xf16> -> tensor<4xf16>
    %26:2 = call @Unknown24(%3) : (tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>)
    %27 = byre.compute @PoolMaxOp_f16_f16(%26#0) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : tensor<4x64x112x112xf16> -> tensor<4x64x56x56xf16>
    %28 = byre.compute @ConvOp_f16f16_f16(%27, %4) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %29 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%28, %arg8, %arg9) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<4x64x56x56xf16>
    %30:2 = call @Unknown26(%29) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %31 = byre.compute @ConvOp_f16f16_f16(%30#0, %5) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %32 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%31, %arg13, %arg14) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<4x64x56x56xf16>
    %33:2 = call @Unknown28(%32, %27) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %34 = byre.compute @ConvOp_f16f16_f16(%33#0, %6) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %35 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%34, %arg18, %arg19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<4x64x56x56xf16>
    %36:2 = call @Unknown30(%35) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %37 = byre.compute @ConvOp_f16f16_f16(%36#0, %7) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %38 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%37, %arg23, %arg24) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<4x64x56x56xf16>
    %39:2 = call @Unknown32(%38, %33#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %40 = byre.compute @ConvOp_f16f16_f16(%39#0, %8) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16> -> tensor<4x128x28x28xf16>
    %41 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%40, %arg38, %arg39) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<4x128x28x28xf16>
    %42 = byre.compute @ConvOp_f16f16_f16(%39#0, %9) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16> -> tensor<4x128x28x28xf16>
    %43 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%42, %arg28, %arg29) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<4x128x28x28xf16>
    %44:2 = call @Unknown35(%43) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %45 = byre.compute @ConvOp_f16f16_f16(%44#0, %10) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<4x128x28x28xf16>
    %46 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%45, %arg33, %arg34) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<4x128x28x28xf16>
    %47:2 = call @Unknown37(%46, %41) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %48 = byre.compute @ConvOp_f16f16_f16(%47#0, %11) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<4x128x28x28xf16>
    %49 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%48, %arg43, %arg44) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<4x128x28x28xf16>
    %50:2 = call @Unknown39(%49) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %51 = byre.compute @ConvOp_f16f16_f16(%50#0, %12) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<4x128x28x28xf16>
    %52 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%51, %arg48, %arg49) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<4x128x28x28xf16>
    %53:2 = call @Unknown41(%52, %47#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %54 = byre.compute @ConvOp_f16f16_f16(%53#0, %13) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16> -> tensor<4x256x14x14xf16>
    %55 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%54, %arg63, %arg64) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<4x256x14x14xf16>
    %56 = byre.compute @ConvOp_f16f16_f16(%53#0, %14) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16> -> tensor<4x256x14x14xf16>
    %57 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%56, %arg53, %arg54) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<4x256x14x14xf16>
    %58:2 = call @Unknown44(%57) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %59 = byre.compute @ConvOp_f16f16_f16(%58#0, %15) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<4x256x14x14xf16>
    %60 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%59, %arg58, %arg59) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<4x256x14x14xf16>
    %61:2 = call @Unknown46(%60, %55) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %62 = byre.compute @ConvOp_f16f16_f16(%61#0, %16) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<4x256x14x14xf16>
    %63 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%62, %arg68, %arg69) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<4x256x14x14xf16>
    %64:2 = call @Unknown48(%63) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %65 = byre.compute @ConvOp_f16f16_f16(%64#0, %17) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<4x256x14x14xf16>
    %66 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%65, %arg73, %arg74) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<4x256x14x14xf16>
    %67:2 = call @Unknown50(%66, %61#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %68 = byre.compute @ConvOp_f16f16_f16(%67#0, %18) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16> -> tensor<4x512x7x7xf16>
    %69 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%68, %arg88, %arg89) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<4x512x7x7xf16>
    %70 = byre.compute @ConvOp_f16f16_f16(%67#0, %19) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16> -> tensor<4x512x7x7xf16>
    %71 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%70, %arg78, %arg79) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<4x512x7x7xf16>
    %72:2 = call @Unknown53(%71) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %73 = byre.compute @ConvOp_f16f16_f16(%72#0, %20) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<4x512x7x7xf16>
    %74 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%73, %arg83, %arg84) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<4x512x7x7xf16>
    %75:2 = call @Unknown55(%74, %69) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %76 = byre.compute @ConvOp_f16f16_f16(%75#0, %21) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<4x512x7x7xf16>
    %77 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%76, %arg93, %arg94) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<4x512x7x7xf16>
    %78:2 = call @Unknown57(%77) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %79 = byre.compute @ConvOp_f16f16_f16(%78#0, %22) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<4x512x7x7xf16>
    %80 = byre.compute @BatchNormTrainingOp_f16f32f32_f16(%79, %arg98, %arg99) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<4x512x7x7xf16>
    %81:2 = call @Unknown59(%80, %75#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %82 = byre.compute @ReduceSumOp_f16_f16(%81#0) {dimensions = dense<[3, 2]> : tensor<2xi64>} : tensor<4x512x7x7xf16> -> tensor<4x512xf16>
    %83 = call @Unknown60(%82) : (tensor<4x512xf16>) -> tensor<4x512xf16>
    %84 = byre.compute @MatmulOp_f16f16_f16(%83, %24) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : tensor<4x512xf16>, tensor<1000x512xf16> -> tensor<4x1000xf16>
    %85 = call @Unknown61(%arg103, %84) : (tensor<1000xf32>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %86 = byre.compute @ReduceMaxOp_f16_f16(%85) {dimensions = dense<1> : tensor<1xi64>} : tensor<4x1000xf16> -> tensor<4xf16>
    %87:2 = call @Unknown62(%86, %85) : (tensor<4xf16>, tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>)
    %88 = byre.compute @ReduceSumOp_f16_f16(%87#1) {dimensions = dense<1> : tensor<1xi64>} : tensor<4x1000xf16> -> tensor<4xf16>
    %89:3 = call @Unknown63(%88, %87#0, %25, %23, %arg1) : (tensor<4xf16>, tensor<4x1000xf16>, tensor<4xf16>, tensor<4x1000xf16>, tensor<4x1000xf32>) -> (tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>)
    %90 = byre.compute @MatmulOp_f16f16_f16(%89#0, %24) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : tensor<4x1000xf16>, tensor<1000x512xf16> -> tensor<4x512xf16>
    %91 = call @Unknown64(%90, %81#1) : (tensor<4x512xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %92:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%79, %arg98, %91) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16> -> tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %93 = byre.compute @ConvBackwardDataOp_f16f16_f16(%92#0, %22) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<4x512x7x7xf16>
    %94 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%78#0, %92#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16> -> tensor<512x512x3x3xf16>
    %95 = call @Unknown68(%78#1, %93) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %96:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%76, %arg93, %95) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16> -> tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %97 = byre.compute @ConvBackwardDataOp_f16f16_f16(%96#0, %21) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<4x512x7x7xf16>
    %98 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%75#0, %96#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16> -> tensor<512x512x3x3xf16>
    %99 = call @Unknown72(%91, %97, %75#1) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %100:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%73, %arg83, %99) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16> -> tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %101 = byre.compute @ConvBackwardDataOp_f16f16_f16(%100#0, %20) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<4x512x7x7xf16>
    %102 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%72#0, %100#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16> -> tensor<512x512x3x3xf16>
    %103 = call @Unknown76(%72#1, %101) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %104:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%70, %arg78, %103) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16> -> tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %105 = byre.compute @ConvBackwardDataOp_f16f16_f16(%104#0, %19) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<512x256x3x3xf16> -> tensor<4x256x14x14xf16>
    %106 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%67#0, %104#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16> -> tensor<512x256x3x3xf16>
    %107:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%68, %arg88, %99) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16> -> tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %108 = byre.compute @ConvBackwardDataOp_f16f16_f16(%107#0, %18) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x512x7x7xf16>, tensor<512x256x1x1xf16> -> tensor<4x256x14x14xf16>
    %109 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%67#0, %107#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16> -> tensor<512x256x1x1xf16>
    %110 = call @Unknown83(%108, %105, %67#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %111:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%65, %arg73, %110) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16> -> tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %112 = byre.compute @ConvBackwardDataOp_f16f16_f16(%111#0, %17) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<4x256x14x14xf16>
    %113 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%64#0, %111#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16> -> tensor<256x256x3x3xf16>
    %114 = call @Unknown87(%64#1, %112) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %115:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%62, %arg68, %114) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16> -> tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %116 = byre.compute @ConvBackwardDataOp_f16f16_f16(%115#0, %16) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<4x256x14x14xf16>
    %117 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%61#0, %115#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16> -> tensor<256x256x3x3xf16>
    %118 = call @Unknown91(%110, %116, %61#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %119:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%59, %arg58, %118) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16> -> tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %120 = byre.compute @ConvBackwardDataOp_f16f16_f16(%119#0, %15) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<4x256x14x14xf16>
    %121 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%58#0, %119#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16> -> tensor<256x256x3x3xf16>
    %122 = call @Unknown95(%58#1, %120) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %123:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%56, %arg53, %122) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16> -> tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %124 = byre.compute @ConvBackwardDataOp_f16f16_f16(%123#0, %14) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<256x128x3x3xf16> -> tensor<4x128x28x28xf16>
    %125 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%53#0, %123#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16> -> tensor<256x128x3x3xf16>
    %126:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%54, %arg63, %118) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16> -> tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %127 = byre.compute @ConvBackwardDataOp_f16f16_f16(%126#0, %13) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x256x14x14xf16>, tensor<256x128x1x1xf16> -> tensor<4x128x28x28xf16>
    %128 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%53#0, %126#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16> -> tensor<256x128x1x1xf16>
    %129 = call @Unknown102(%127, %124, %53#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %130:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%51, %arg48, %129) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16> -> tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %131 = byre.compute @ConvBackwardDataOp_f16f16_f16(%130#0, %12) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<4x128x28x28xf16>
    %132 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%50#0, %130#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16> -> tensor<128x128x3x3xf16>
    %133 = call @Unknown106(%50#1, %131) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %134:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%48, %arg43, %133) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16> -> tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %135 = byre.compute @ConvBackwardDataOp_f16f16_f16(%134#0, %11) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<4x128x28x28xf16>
    %136 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%47#0, %134#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16> -> tensor<128x128x3x3xf16>
    %137 = call @Unknown110(%129, %135, %47#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %138:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%45, %arg33, %137) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16> -> tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %139 = byre.compute @ConvBackwardDataOp_f16f16_f16(%138#0, %10) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<4x128x28x28xf16>
    %140 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%44#0, %138#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16> -> tensor<128x128x3x3xf16>
    %141 = call @Unknown114(%44#1, %139) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %142:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%42, %arg28, %141) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16> -> tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %143 = byre.compute @ConvBackwardDataOp_f16f16_f16(%142#0, %9) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<128x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %144 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%39#0, %142#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16> -> tensor<128x64x3x3xf16>
    %145:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%40, %arg38, %137) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16> -> tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %146 = byre.compute @ConvBackwardDataOp_f16f16_f16(%145#0, %8) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x128x28x28xf16>, tensor<128x64x1x1xf16> -> tensor<4x64x56x56xf16>
    %147 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%39#0, %145#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16> -> tensor<128x64x1x1xf16>
    %148 = call @Unknown121(%146, %143, %39#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %149:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%37, %arg23, %148) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16> -> tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %150 = byre.compute @ConvBackwardDataOp_f16f16_f16(%149#0, %7) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %151 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%36#0, %149#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16> -> tensor<64x64x3x3xf16>
    %152 = call @Unknown125(%36#1, %150) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %153:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%34, %arg18, %152) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16> -> tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %154 = byre.compute @ConvBackwardDataOp_f16f16_f16(%153#0, %6) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %155 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%33#0, %153#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16> -> tensor<64x64x3x3xf16>
    %156 = call @Unknown129(%148, %154, %33#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %157:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%31, %arg13, %156) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16> -> tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %158 = byre.compute @ConvBackwardDataOp_f16f16_f16(%157#0, %5) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %159 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%30#0, %157#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16> -> tensor<64x64x3x3xf16>
    %160 = call @Unknown133(%30#1, %158) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %161:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%28, %arg8, %160) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16> -> tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %162 = byre.compute @ConvBackwardDataOp_f16f16_f16(%161#0, %4) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<4x64x56x56xf16>
    %163 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%27, %161#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16> -> tensor<64x64x3x3xf16>
    %164 = call @Unknown137(%156, %162) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %165 = byre.compute @PoolMaxGradOp_f16f16_f16(%26#0, %164) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : tensor<4x64x112x112xf16>, tensor<4x64x56x56xf16> -> tensor<4x64x112x112xf16>
    %166 = call @Unknown138(%26#1, %165) : (tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %167:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%2, %arg3, %166) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<4x64x112x112xf16> -> tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
    %168 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%0, %167#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16> -> tensor<64x3x7x7xf16>
    %169 = byre.compute @ReduceSumOp_f32_f32(%89#1) {dimensions = dense<[0, 1]> : tensor<2xi64>} : tensor<4x1000xf32> -> tensor<f32>
    %170 = call @Unknown141(%169) : (tensor<f32>) -> tensor<f32>
    %171 = call @Unknown142(%168) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %172 = call @Unknown143(%163) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %173 = call @Unknown144(%159) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %174 = call @Unknown145(%155) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %175 = call @Unknown146(%151) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %176 = call @Unknown147(%144) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %177 = call @Unknown148(%140) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %178 = call @Unknown149(%147) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %179 = call @Unknown150(%136) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %180 = call @Unknown151(%132) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %181 = call @Unknown152(%125) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %182 = call @Unknown153(%121) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %183 = call @Unknown154(%128) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %184 = call @Unknown155(%117) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %185 = call @Unknown156(%113) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %186 = call @Unknown157(%106) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %187 = call @Unknown158(%102) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %188 = call @Unknown159(%109) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %189 = call @Unknown160(%98) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %190 = call @Unknown161(%94) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %191 = byre.compute @MatmulOp_f16f16_f16(%83, %89#0) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : tensor<4x512xf16>, tensor<4x1000xf16> -> tensor<1000x512xf16>
    %192 = call @Unknown163(%191) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %193 = byre.compute @ReduceSumOp_f32_f32(%89#2) {dimensions = dense<0> : tensor<1xi64>} : tensor<4x1000xf32> -> tensor<1000xf32>
    %194 = call @Unknown164(%193) : (tensor<1000xf32>) -> tensor<1000xf32>
    return %170, %171, %167#1, %167#2, %172, %161#1, %161#2, %173, %157#1, %157#2, %174, %153#1, %153#2, %175, %149#1, %149#2, %176, %142#1, %142#2, %177, %138#1, %138#2, %178, %145#1, %145#2, %179, %134#1, %134#2, %180, %130#1, %130#2, %181, %123#1, %123#2, %182, %119#1, %119#2, %183, %126#1, %126#2, %184, %115#1, %115#2, %185, %111#1, %111#2, %186, %104#1, %104#2, %187, %100#1, %100#2, %188, %107#1, %107#2, %189, %96#1, %96#2, %190, %92#1, %92#2, %192, %194 : tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>
  }
}