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
      %3 = arith.maxnumf %in, %cst : f16
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
      %3 = arith.maxnumf %in, %cst : f16
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
      %4 = arith.maxnumf %3, %cst : f16
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
      %3 = arith.maxnumf %in, %cst : f16
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
      %4 = arith.maxnumf %3, %cst : f16
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
      %3 = arith.maxnumf %in, %cst : f16
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
      %4 = arith.maxnumf %3, %cst : f16
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
      %3 = arith.maxnumf %in, %cst : f16
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
      %4 = arith.maxnumf %3, %cst : f16
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
      %3 = arith.maxnumf %in, %cst : f16
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
      %4 = arith.maxnumf %3, %cst : f16
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
      %3 = arith.maxnumf %in, %cst : f16
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
      %4 = arith.maxnumf %3, %cst : f16
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
      %3 = arith.maxnumf %in, %cst : f16
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
      %4 = arith.maxnumf %3, %cst : f16
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
      %3 = arith.maxnumf %in, %cst : f16
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
      %4 = arith.maxnumf %3, %cst : f16
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
    %2 = tensor.empty() : tensor<4x64x112x112xf16>
    %3 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%0, %1 : tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>) outs(%2 : tensor<4x64x112x112xf16>) : tensor<4x64x112x112xf16>
    %4 = tensor.empty() : tensor<4x64x112x112xf16>
    %5 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%3, %arg3, %arg4 : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) outs(%4 : tensor<4x64x112x112xf16>) : tensor<4x64x112x112xf16>
    %6 = call @Unknown3(%arg7) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %7 = call @Unknown4(%arg12) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %8 = call @Unknown5(%arg17) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %9 = call @Unknown6(%arg22) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %10 = call @Unknown7(%arg37) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %11 = call @Unknown8(%arg27) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %12 = call @Unknown9(%arg32) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %13 = call @Unknown10(%arg42) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %14 = call @Unknown11(%arg47) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %15 = call @Unknown12(%arg62) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %16 = call @Unknown13(%arg52) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %17 = call @Unknown14(%arg57) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %18 = call @Unknown15(%arg67) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %19 = call @Unknown16(%arg72) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %20 = call @Unknown17(%arg87) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %21 = call @Unknown18(%arg77) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %22 = call @Unknown19(%arg82) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %23 = call @Unknown20(%arg92) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %24 = call @Unknown21(%arg97) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %25 = call @Unknown22(%arg1) : (tensor<4x1000xf32>) -> tensor<4x1000xf16>
    %26 = call @Unknown23(%arg102) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %27 = tensor.empty() : tensor<4xf16>
    %28 = byre.compute_on_tensor @ReduceSumOp_f16_f16 {dimensions = dense<1> : tensor<1xi64>} ins(%25 : tensor<4x1000xf16>) outs(%27 : tensor<4xf16>) : tensor<4xf16>
    %29:2 = call @Unknown24(%5) : (tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>)
    %30 = tensor.empty() : tensor<4x64x56x56xf16>
    %31 = byre.compute_on_tensor @PoolMaxOp_f16_f16 {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} ins(%29#0 : tensor<4x64x112x112xf16>) outs(%30 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %32 = tensor.empty() : tensor<4x64x56x56xf16>
    %33 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%31, %6 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%32 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %34 = tensor.empty() : tensor<4x64x56x56xf16>
    %35 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%33, %arg8, %arg9 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%34 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %36:2 = call @Unknown26(%35) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %37 = tensor.empty() : tensor<4x64x56x56xf16>
    %38 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%36#0, %7 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%37 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %39 = tensor.empty() : tensor<4x64x56x56xf16>
    %40 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%38, %arg13, %arg14 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%39 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %41:2 = call @Unknown28(%40, %31) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %42 = tensor.empty() : tensor<4x64x56x56xf16>
    %43 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%41#0, %8 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%42 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %44 = tensor.empty() : tensor<4x64x56x56xf16>
    %45 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%43, %arg18, %arg19 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%44 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %46:2 = call @Unknown30(%45) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %47 = tensor.empty() : tensor<4x64x56x56xf16>
    %48 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%46#0, %9 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%47 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %49 = tensor.empty() : tensor<4x64x56x56xf16>
    %50 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%48, %arg23, %arg24 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%49 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %51:2 = call @Unknown32(%50, %41#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %52 = tensor.empty() : tensor<4x128x28x28xf16>
    %53 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%51#0, %10 : tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) outs(%52 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %54 = tensor.empty() : tensor<4x128x28x28xf16>
    %55 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%53, %arg38, %arg39 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%54 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %56 = tensor.empty() : tensor<4x128x28x28xf16>
    %57 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%51#0, %11 : tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) outs(%56 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %58 = tensor.empty() : tensor<4x128x28x28xf16>
    %59 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%57, %arg28, %arg29 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%58 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %60:2 = call @Unknown35(%59) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %61 = tensor.empty() : tensor<4x128x28x28xf16>
    %62 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%60#0, %12 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%61 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %63 = tensor.empty() : tensor<4x128x28x28xf16>
    %64 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%62, %arg33, %arg34 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%63 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %65:2 = call @Unknown37(%64, %55) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %66 = tensor.empty() : tensor<4x128x28x28xf16>
    %67 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%65#0, %13 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%66 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %68 = tensor.empty() : tensor<4x128x28x28xf16>
    %69 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%67, %arg43, %arg44 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%68 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %70:2 = call @Unknown39(%69) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %71 = tensor.empty() : tensor<4x128x28x28xf16>
    %72 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%70#0, %14 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%71 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %73 = tensor.empty() : tensor<4x128x28x28xf16>
    %74 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%72, %arg48, %arg49 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%73 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %75:2 = call @Unknown41(%74, %65#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %76 = tensor.empty() : tensor<4x256x14x14xf16>
    %77 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%75#0, %15 : tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) outs(%76 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %78 = tensor.empty() : tensor<4x256x14x14xf16>
    %79 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%77, %arg63, %arg64 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%78 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %80 = tensor.empty() : tensor<4x256x14x14xf16>
    %81 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%75#0, %16 : tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) outs(%80 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %82 = tensor.empty() : tensor<4x256x14x14xf16>
    %83 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%81, %arg53, %arg54 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%82 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %84:2 = call @Unknown44(%83) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %85 = tensor.empty() : tensor<4x256x14x14xf16>
    %86 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%84#0, %17 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%85 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %87 = tensor.empty() : tensor<4x256x14x14xf16>
    %88 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%86, %arg58, %arg59 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%87 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %89:2 = call @Unknown46(%88, %79) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %90 = tensor.empty() : tensor<4x256x14x14xf16>
    %91 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%89#0, %18 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%90 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %92 = tensor.empty() : tensor<4x256x14x14xf16>
    %93 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%91, %arg68, %arg69 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%92 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %94:2 = call @Unknown48(%93) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %95 = tensor.empty() : tensor<4x256x14x14xf16>
    %96 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%94#0, %19 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%95 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %97 = tensor.empty() : tensor<4x256x14x14xf16>
    %98 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%96, %arg73, %arg74 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%97 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %99:2 = call @Unknown50(%98, %89#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %100 = tensor.empty() : tensor<4x512x7x7xf16>
    %101 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%99#0, %20 : tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) outs(%100 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %102 = tensor.empty() : tensor<4x512x7x7xf16>
    %103 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%101, %arg88, %arg89 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%102 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %104 = tensor.empty() : tensor<4x512x7x7xf16>
    %105 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%99#0, %21 : tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) outs(%104 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %106 = tensor.empty() : tensor<4x512x7x7xf16>
    %107 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%105, %arg78, %arg79 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%106 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %108:2 = call @Unknown53(%107) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %109 = tensor.empty() : tensor<4x512x7x7xf16>
    %110 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%108#0, %22 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%109 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %111 = tensor.empty() : tensor<4x512x7x7xf16>
    %112 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%110, %arg83, %arg84 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%111 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %113:2 = call @Unknown55(%112, %103) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %114 = tensor.empty() : tensor<4x512x7x7xf16>
    %115 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%113#0, %23 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%114 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %116 = tensor.empty() : tensor<4x512x7x7xf16>
    %117 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%115, %arg93, %arg94 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%116 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %118:2 = call @Unknown57(%117) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %119 = tensor.empty() : tensor<4x512x7x7xf16>
    %120 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%118#0, %24 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%119 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %121 = tensor.empty() : tensor<4x512x7x7xf16>
    %122 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%120, %arg98, %arg99 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%121 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %123:2 = call @Unknown59(%122, %113#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %124 = tensor.empty() : tensor<4x512xf16>
    %125 = byre.compute_on_tensor @ReduceSumOp_f16_f16 {dimensions = dense<[3, 2]> : tensor<2xi64>} ins(%123#0 : tensor<4x512x7x7xf16>) outs(%124 : tensor<4x512xf16>) : tensor<4x512xf16>
    %126 = call @Unknown60(%125) : (tensor<4x512xf16>) -> tensor<4x512xf16>
    %127 = tensor.empty() : tensor<4x1000xf16>
    %128 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} ins(%126, %26 : tensor<4x512xf16>, tensor<1000x512xf16>) outs(%127 : tensor<4x1000xf16>) : tensor<4x1000xf16>
    %129 = call @Unknown61(%arg103, %128) : (tensor<1000xf32>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %130 = tensor.empty() : tensor<4xf16>
    %131 = byre.compute_on_tensor @ReduceMaxOp_f16_f16 {dimensions = dense<1> : tensor<1xi64>} ins(%129 : tensor<4x1000xf16>) outs(%130 : tensor<4xf16>) : tensor<4xf16>
    %132:2 = call @Unknown62(%131, %129) : (tensor<4xf16>, tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>)
    %133 = tensor.empty() : tensor<4xf16>
    %134 = byre.compute_on_tensor @ReduceSumOp_f16_f16 {dimensions = dense<1> : tensor<1xi64>} ins(%132#1 : tensor<4x1000xf16>) outs(%133 : tensor<4xf16>) : tensor<4xf16>
    %135:3 = call @Unknown63(%134, %132#0, %28, %25, %arg1) : (tensor<4xf16>, tensor<4x1000xf16>, tensor<4xf16>, tensor<4x1000xf16>, tensor<4x1000xf32>) -> (tensor<4x1000xf16>, tensor<4x1000xf32>, tensor<4x1000xf32>)
    %136 = tensor.empty() : tensor<4x512xf16>
    %137 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} ins(%135#0, %26 : tensor<4x1000xf16>, tensor<1000x512xf16>) outs(%136 : tensor<4x512xf16>) : tensor<4x512xf16>
    %138 = call @Unknown64(%137, %123#1) : (tensor<4x512xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %139 = tensor.empty() : tensor<4x512x7x7xf16>
    %140 = tensor.empty() : tensor<512xf32>
    %141 = tensor.empty() : tensor<512xf32>
    %142:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%120, %arg98, %138 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%139, %140, %141 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %143 = tensor.empty() : tensor<4x512x7x7xf16>
    %144 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%142#0, %24 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%143 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %145 = tensor.empty() : tensor<512x512x3x3xf16>
    %146 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%118#0, %142#0 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%145 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %147 = call @Unknown68(%118#1, %144) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %148 = tensor.empty() : tensor<4x512x7x7xf16>
    %149 = tensor.empty() : tensor<512xf32>
    %150 = tensor.empty() : tensor<512xf32>
    %151:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%115, %arg93, %147 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%148, %149, %150 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %152 = tensor.empty() : tensor<4x512x7x7xf16>
    %153 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%151#0, %23 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%152 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %154 = tensor.empty() : tensor<512x512x3x3xf16>
    %155 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%113#0, %151#0 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%154 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %156 = call @Unknown72(%138, %153, %113#1) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %157 = tensor.empty() : tensor<4x512x7x7xf16>
    %158 = tensor.empty() : tensor<512xf32>
    %159 = tensor.empty() : tensor<512xf32>
    %160:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%110, %arg83, %156 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%157, %158, %159 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %161 = tensor.empty() : tensor<4x512x7x7xf16>
    %162 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%160#0, %22 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%161 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %163 = tensor.empty() : tensor<512x512x3x3xf16>
    %164 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%108#0, %160#0 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%163 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %165 = call @Unknown76(%108#1, %162) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %166 = tensor.empty() : tensor<4x512x7x7xf16>
    %167 = tensor.empty() : tensor<512xf32>
    %168 = tensor.empty() : tensor<512xf32>
    %169:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%105, %arg78, %165 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%166, %167, %168 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %170 = tensor.empty() : tensor<4x256x14x14xf16>
    %171 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%169#0, %21 : tensor<4x512x7x7xf16>, tensor<512x256x3x3xf16>) outs(%170 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %172 = tensor.empty() : tensor<512x256x3x3xf16>
    %173 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%99#0, %169#0 : tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) outs(%172 : tensor<512x256x3x3xf16>) : tensor<512x256x3x3xf16>
    %174 = tensor.empty() : tensor<4x512x7x7xf16>
    %175 = tensor.empty() : tensor<512xf32>
    %176 = tensor.empty() : tensor<512xf32>
    %177:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%101, %arg88, %156 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%174, %175, %176 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %178 = tensor.empty() : tensor<4x256x14x14xf16>
    %179 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%177#0, %20 : tensor<4x512x7x7xf16>, tensor<512x256x1x1xf16>) outs(%178 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %180 = tensor.empty() : tensor<512x256x1x1xf16>
    %181 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%99#0, %177#0 : tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) outs(%180 : tensor<512x256x1x1xf16>) : tensor<512x256x1x1xf16>
    %182 = call @Unknown83(%179, %171, %99#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %183 = tensor.empty() : tensor<4x256x14x14xf16>
    %184 = tensor.empty() : tensor<256xf32>
    %185 = tensor.empty() : tensor<256xf32>
    %186:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%96, %arg73, %182 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%183, %184, %185 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %187 = tensor.empty() : tensor<4x256x14x14xf16>
    %188 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%186#0, %19 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%187 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %189 = tensor.empty() : tensor<256x256x3x3xf16>
    %190 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%94#0, %186#0 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%189 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %191 = call @Unknown87(%94#1, %188) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %192 = tensor.empty() : tensor<4x256x14x14xf16>
    %193 = tensor.empty() : tensor<256xf32>
    %194 = tensor.empty() : tensor<256xf32>
    %195:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%91, %arg68, %191 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%192, %193, %194 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %196 = tensor.empty() : tensor<4x256x14x14xf16>
    %197 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%195#0, %18 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%196 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %198 = tensor.empty() : tensor<256x256x3x3xf16>
    %199 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%89#0, %195#0 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%198 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %200 = call @Unknown91(%182, %197, %89#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %201 = tensor.empty() : tensor<4x256x14x14xf16>
    %202 = tensor.empty() : tensor<256xf32>
    %203 = tensor.empty() : tensor<256xf32>
    %204:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%86, %arg58, %200 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%201, %202, %203 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %205 = tensor.empty() : tensor<4x256x14x14xf16>
    %206 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%204#0, %17 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%205 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %207 = tensor.empty() : tensor<256x256x3x3xf16>
    %208 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%84#0, %204#0 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%207 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %209 = call @Unknown95(%84#1, %206) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %210 = tensor.empty() : tensor<4x256x14x14xf16>
    %211 = tensor.empty() : tensor<256xf32>
    %212 = tensor.empty() : tensor<256xf32>
    %213:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%81, %arg53, %209 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%210, %211, %212 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %214 = tensor.empty() : tensor<4x128x28x28xf16>
    %215 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%213#0, %16 : tensor<4x256x14x14xf16>, tensor<256x128x3x3xf16>) outs(%214 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %216 = tensor.empty() : tensor<256x128x3x3xf16>
    %217 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%75#0, %213#0 : tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) outs(%216 : tensor<256x128x3x3xf16>) : tensor<256x128x3x3xf16>
    %218 = tensor.empty() : tensor<4x256x14x14xf16>
    %219 = tensor.empty() : tensor<256xf32>
    %220 = tensor.empty() : tensor<256xf32>
    %221:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%77, %arg63, %200 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%218, %219, %220 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %222 = tensor.empty() : tensor<4x128x28x28xf16>
    %223 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%221#0, %15 : tensor<4x256x14x14xf16>, tensor<256x128x1x1xf16>) outs(%222 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %224 = tensor.empty() : tensor<256x128x1x1xf16>
    %225 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%75#0, %221#0 : tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) outs(%224 : tensor<256x128x1x1xf16>) : tensor<256x128x1x1xf16>
    %226 = call @Unknown102(%223, %215, %75#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %227 = tensor.empty() : tensor<4x128x28x28xf16>
    %228 = tensor.empty() : tensor<128xf32>
    %229 = tensor.empty() : tensor<128xf32>
    %230:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%72, %arg48, %226 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%227, %228, %229 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %231 = tensor.empty() : tensor<4x128x28x28xf16>
    %232 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%230#0, %14 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%231 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %233 = tensor.empty() : tensor<128x128x3x3xf16>
    %234 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%70#0, %230#0 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%233 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %235 = call @Unknown106(%70#1, %232) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %236 = tensor.empty() : tensor<4x128x28x28xf16>
    %237 = tensor.empty() : tensor<128xf32>
    %238 = tensor.empty() : tensor<128xf32>
    %239:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%67, %arg43, %235 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%236, %237, %238 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %240 = tensor.empty() : tensor<4x128x28x28xf16>
    %241 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%239#0, %13 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%240 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %242 = tensor.empty() : tensor<128x128x3x3xf16>
    %243 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%65#0, %239#0 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%242 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %244 = call @Unknown110(%226, %241, %65#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %245 = tensor.empty() : tensor<4x128x28x28xf16>
    %246 = tensor.empty() : tensor<128xf32>
    %247 = tensor.empty() : tensor<128xf32>
    %248:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%62, %arg33, %244 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%245, %246, %247 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %249 = tensor.empty() : tensor<4x128x28x28xf16>
    %250 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%248#0, %12 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%249 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %251 = tensor.empty() : tensor<128x128x3x3xf16>
    %252 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%60#0, %248#0 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%251 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %253 = call @Unknown114(%60#1, %250) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %254 = tensor.empty() : tensor<4x128x28x28xf16>
    %255 = tensor.empty() : tensor<128xf32>
    %256 = tensor.empty() : tensor<128xf32>
    %257:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%57, %arg28, %253 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%254, %255, %256 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %258 = tensor.empty() : tensor<4x64x56x56xf16>
    %259 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%257#0, %11 : tensor<4x128x28x28xf16>, tensor<128x64x3x3xf16>) outs(%258 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %260 = tensor.empty() : tensor<128x64x3x3xf16>
    %261 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%51#0, %257#0 : tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) outs(%260 : tensor<128x64x3x3xf16>) : tensor<128x64x3x3xf16>
    %262 = tensor.empty() : tensor<4x128x28x28xf16>
    %263 = tensor.empty() : tensor<128xf32>
    %264 = tensor.empty() : tensor<128xf32>
    %265:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%53, %arg38, %244 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%262, %263, %264 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %266 = tensor.empty() : tensor<4x64x56x56xf16>
    %267 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%265#0, %10 : tensor<4x128x28x28xf16>, tensor<128x64x1x1xf16>) outs(%266 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %268 = tensor.empty() : tensor<128x64x1x1xf16>
    %269 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%51#0, %265#0 : tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) outs(%268 : tensor<128x64x1x1xf16>) : tensor<128x64x1x1xf16>
    %270 = call @Unknown121(%267, %259, %51#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %271 = tensor.empty() : tensor<4x64x56x56xf16>
    %272 = tensor.empty() : tensor<64xf32>
    %273 = tensor.empty() : tensor<64xf32>
    %274:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%48, %arg23, %270 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) outs(%271, %272, %273 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %275 = tensor.empty() : tensor<4x64x56x56xf16>
    %276 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%274#0, %9 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%275 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %277 = tensor.empty() : tensor<64x64x3x3xf16>
    %278 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%46#0, %274#0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%277 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %279 = call @Unknown125(%46#1, %276) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %280 = tensor.empty() : tensor<4x64x56x56xf16>
    %281 = tensor.empty() : tensor<64xf32>
    %282 = tensor.empty() : tensor<64xf32>
    %283:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%43, %arg18, %279 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) outs(%280, %281, %282 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %284 = tensor.empty() : tensor<4x64x56x56xf16>
    %285 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%283#0, %8 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%284 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %286 = tensor.empty() : tensor<64x64x3x3xf16>
    %287 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%41#0, %283#0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%286 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %288 = call @Unknown129(%270, %285, %41#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %289 = tensor.empty() : tensor<4x64x56x56xf16>
    %290 = tensor.empty() : tensor<64xf32>
    %291 = tensor.empty() : tensor<64xf32>
    %292:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%38, %arg13, %288 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) outs(%289, %290, %291 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %293 = tensor.empty() : tensor<4x64x56x56xf16>
    %294 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%292#0, %7 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%293 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %295 = tensor.empty() : tensor<64x64x3x3xf16>
    %296 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%36#0, %292#0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%295 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %297 = call @Unknown133(%36#1, %294) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %298 = tensor.empty() : tensor<4x64x56x56xf16>
    %299 = tensor.empty() : tensor<64xf32>
    %300 = tensor.empty() : tensor<64xf32>
    %301:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%33, %arg8, %297 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) outs(%298, %299, %300 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %302 = tensor.empty() : tensor<4x64x56x56xf16>
    %303 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%301#0, %6 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%302 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %304 = tensor.empty() : tensor<64x64x3x3xf16>
    %305 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%31, %301#0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%304 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %306 = call @Unknown137(%288, %303) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %307 = tensor.empty() : tensor<4x64x112x112xf16>
    %308 = byre.compute_on_tensor @PoolMaxGradOp_f16f16_f16 {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} ins(%29#0, %306 : tensor<4x64x112x112xf16>, tensor<4x64x56x56xf16>) outs(%307 : tensor<4x64x112x112xf16>) : tensor<4x64x112x112xf16>
    %309 = call @Unknown138(%29#1, %308) : (tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %310 = tensor.empty() : tensor<4x64x112x112xf16>
    %311 = tensor.empty() : tensor<64xf32>
    %312 = tensor.empty() : tensor<64xf32>
    %313:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%3, %arg3, %309 : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<4x64x112x112xf16>) outs(%310, %311, %312 : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
    %314 = tensor.empty() : tensor<64x3x7x7xf16>
    %315 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%0, %313#0 : tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) outs(%314 : tensor<64x3x7x7xf16>) : tensor<64x3x7x7xf16>
    %316 = tensor.empty() : tensor<f32>
    %317 = byre.compute_on_tensor @ReduceSumOp_f32_f32 {dimensions = dense<[0, 1]> : tensor<2xi64>} ins(%135#1 : tensor<4x1000xf32>) outs(%316 : tensor<f32>) : tensor<f32>
    %318 = call @Unknown141(%317) : (tensor<f32>) -> tensor<f32>
    %319 = call @Unknown142(%315) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %320 = call @Unknown143(%305) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %321 = call @Unknown144(%296) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %322 = call @Unknown145(%287) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %323 = call @Unknown146(%278) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %324 = call @Unknown147(%261) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %325 = call @Unknown148(%252) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %326 = call @Unknown149(%269) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %327 = call @Unknown150(%243) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %328 = call @Unknown151(%234) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %329 = call @Unknown152(%217) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %330 = call @Unknown153(%208) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %331 = call @Unknown154(%225) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %332 = call @Unknown155(%199) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %333 = call @Unknown156(%190) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %334 = call @Unknown157(%173) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %335 = call @Unknown158(%164) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %336 = call @Unknown159(%181) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %337 = call @Unknown160(%155) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %338 = call @Unknown161(%146) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %339 = tensor.empty() : tensor<1000x512xf16>
    %340 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} ins(%126, %135#0 : tensor<4x512xf16>, tensor<4x1000xf16>) outs(%339 : tensor<1000x512xf16>) : tensor<1000x512xf16>
    %341 = call @Unknown163(%340) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %342 = tensor.empty() : tensor<1000xf32>
    %343 = byre.compute_on_tensor @ReduceSumOp_f32_f32 {dimensions = dense<0> : tensor<1xi64>} ins(%135#2 : tensor<4x1000xf32>) outs(%342 : tensor<1000xf32>) : tensor<1000xf32>
    %344 = call @Unknown164(%343) : (tensor<1000xf32>) -> tensor<1000xf32>
    return %318, %319, %313#1, %313#2, %320, %301#1, %301#2, %321, %292#1, %292#2, %322, %283#1, %283#2, %323, %274#1, %274#2, %324, %257#1, %257#2, %325, %248#1, %248#2, %326, %265#1, %265#2, %327, %239#1, %239#2, %328, %230#1, %230#2, %329, %213#1, %213#2, %330, %204#1, %204#2, %331, %221#1, %221#2, %332, %195#1, %195#2, %333, %186#1, %186#2, %334, %169#1, %169#2, %335, %160#1, %160#2, %336, %177#1, %177#2, %337, %151#1, %151#2, %338, %142#1, %142#2, %341, %344 : tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>
  }
}