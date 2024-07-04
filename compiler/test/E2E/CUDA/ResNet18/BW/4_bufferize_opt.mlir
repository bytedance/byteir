// RUN: byteir-opt %s -byteir-bufferize-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0)>
module {
  func.func private @Unknown0(%arg0: tensor<1x512xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 4.900000e+01 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg0 : tensor<1x512x7x7xf16>, tensor<1x512xf16>) outs(%0 : tensor<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %2 = arith.divf %in_1, %cst_0 : f16
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %2, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown4(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) outs(%0 : tensor<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown8(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>, %arg2: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) outs(%0 : tensor<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %2, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown12(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) outs(%0 : tensor<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x512x7x7xf16>
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown19(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%0 : tensor<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %2, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown23(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%0 : tensor<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown27(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%0 : tensor<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %2, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown31(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%0 : tensor<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown38(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%0 : tensor<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %2, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown42(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%0 : tensor<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown46(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%0 : tensor<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %2, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown50(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%0 : tensor<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown57(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %2, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown61(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown65(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in_0, %in_1 : f16
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %2, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown69(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown73(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%0 : tensor<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.addf %in, %in_0 : f16
      linalg.yield %2 : f16
    } -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown74(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x112x112xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) outs(%0 : tensor<1x64x112x112xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %2 = arith.cmpf ogt, %in, %cst : f16
      %3 = arith.select %2, %in_0, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x64x112x112xf16>
    return %1 : tensor<1x64x112x112xf16>
  }
  func.func private @Unknown77(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x3x7x7xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x3x7x7xf16>) outs(%0 : tensor<64x3x7x7xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x3x7x7xf32>
    return %1 : tensor<64x3x7x7xf32>
  }
  func.func private @Unknown78(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1x1000xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x1000xf16>) outs(%0 : tensor<1x1000xf32>) {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<1x1000xf32>
    return %1 : tensor<1x1000xf32>
  }
  func.func private @Unknown79(%arg0: tensor<1000xf32>) -> tensor<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1000xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg0 : tensor<1000xf32>) outs(%0 : tensor<1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.truncf %in : f32 to f16
      %3 = arith.extf %2 : f16 to f32
      linalg.yield %3 : f32
    } -> tensor<1000xf32>
    return %1 : tensor<1000xf32>
  }
  func.func private @Unknown80(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<1000x512xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1000x512xf16>) outs(%0 : tensor<1000x512xf32>) {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<1000x512xf32>
    return %1 : tensor<1000x512xf32>
  }
  func.func private @Unknown81(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf16>) outs(%0 : tensor<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x64x3x3xf32>
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown82(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf16>) outs(%0 : tensor<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x64x3x3xf32>
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown83(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf16>) outs(%0 : tensor<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x64x3x3xf32>
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown84(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x64x3x3xf16>) outs(%0 : tensor<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<64x64x3x3xf32>
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown85(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x3x3xf16>) outs(%0 : tensor<128x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x64x3x3xf32>
    return %1 : tensor<128x64x3x3xf32>
  }
  func.func private @Unknown86(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf16>) outs(%0 : tensor<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x128x3x3xf32>
    return %1 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown87(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x1x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x1x1xf16>) outs(%0 : tensor<128x64x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x64x1x1xf32>
    return %1 : tensor<128x64x1x1xf32>
  }
  func.func private @Unknown88(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf16>) outs(%0 : tensor<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x128x3x3xf32>
    return %1 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown89(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x128x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128x3x3xf16>) outs(%0 : tensor<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<128x128x3x3xf32>
    return %1 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown90(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x3x3xf16>) outs(%0 : tensor<256x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x128x3x3xf32>
    return %1 : tensor<256x128x3x3xf32>
  }
  func.func private @Unknown91(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf16>) outs(%0 : tensor<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x256x3x3xf32>
    return %1 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown92(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x1x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x1x1xf16>) outs(%0 : tensor<256x128x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x128x1x1xf32>
    return %1 : tensor<256x128x1x1xf32>
  }
  func.func private @Unknown93(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf16>) outs(%0 : tensor<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x256x3x3xf32>
    return %1 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown94(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x256x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x256x3x3xf16>) outs(%0 : tensor<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<256x256x3x3xf32>
    return %1 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown95(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x3x3xf16>) outs(%0 : tensor<512x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x256x3x3xf32>
    return %1 : tensor<512x256x3x3xf32>
  }
  func.func private @Unknown96(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf16>) outs(%0 : tensor<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x512x3x3xf32>
    return %1 : tensor<512x512x3x3xf32>
  }
  func.func private @Unknown97(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x1x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x1x1xf16>) outs(%0 : tensor<512x256x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x256x1x1xf32>
    return %1 : tensor<512x256x1x1xf32>
  }
  func.func private @Unknown98(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf16>) outs(%0 : tensor<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x512x3x3xf32>
    return %1 : tensor<512x512x3x3xf32>
  }
  func.func private @Unknown99(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x512x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x512x3x3xf16>) outs(%0 : tensor<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    } -> tensor<512x512x3x3xf32>
    return %1 : tensor<512x512x3x3xf32>
  }
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<256xf32>, %arg30: tensor<512xf32>, %arg31: tensor<512xf32>, %arg32: tensor<512xf32>, %arg33: tensor<512xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<512xf32>, %arg38: tensor<512xf32>, %arg39: tensor<512xf32>, %arg40: tensor<64xf32>, %arg41: tensor<64xf32>, %arg42: tensor<64xf32>, %arg43: tensor<64xf32>, %arg44: tensor<64xf32>, %arg45: tensor<64xf32>, %arg46: tensor<64xf32>, %arg47: tensor<64xf32>, %arg48: tensor<64xf32>, %arg49: tensor<64xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<128xf32>, %arg53: tensor<128xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<512xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<64x3x7x7xf16>, %arg81: tensor<1x3x224x224xf16>, %arg82: tensor<1x64x112x112xf16>, %arg83: tensor<1x64x112x112xf16>, %arg84: tensor<1x64x56x56xf16>, %arg85: tensor<64x64x3x3xf16>, %arg86: tensor<1x64x56x56xf16>, %arg87: tensor<1x64x56x56xf16>, %arg88: tensor<64x64x3x3xf16>, %arg89: tensor<1x64x56x56xf16>, %arg90: tensor<1x64x56x56xf16>, %arg91: tensor<64x64x3x3xf16>, %arg92: tensor<1x64x56x56xf16>, %arg93: tensor<1x64x56x56xf16>, %arg94: tensor<64x64x3x3xf16>, %arg95: tensor<1x64x56x56xf16>, %arg96: tensor<1x64x56x56xf16>, %arg97: tensor<128x64x3x3xf16>, %arg98: tensor<1x128x28x28xf16>, %arg99: tensor<1x128x28x28xf16>, %arg100: tensor<128x128x3x3xf16>, %arg101: tensor<1x128x28x28xf16>, %arg102: tensor<128x64x1x1xf16>, %arg103: tensor<1x128x28x28xf16>, %arg104: tensor<1x128x28x28xf16>, %arg105: tensor<128x128x3x3xf16>, %arg106: tensor<1x128x28x28xf16>, %arg107: tensor<1x128x28x28xf16>, %arg108: tensor<128x128x3x3xf16>, %arg109: tensor<1x128x28x28xf16>, %arg110: tensor<1x128x28x28xf16>, %arg111: tensor<256x128x3x3xf16>, %arg112: tensor<1x256x14x14xf16>, %arg113: tensor<1x256x14x14xf16>, %arg114: tensor<256x256x3x3xf16>, %arg115: tensor<1x256x14x14xf16>, %arg116: tensor<256x128x1x1xf16>, %arg117: tensor<1x256x14x14xf16>, %arg118: tensor<1x256x14x14xf16>, %arg119: tensor<256x256x3x3xf16>, %arg120: tensor<1x256x14x14xf16>, %arg121: tensor<1x256x14x14xf16>, %arg122: tensor<256x256x3x3xf16>, %arg123: tensor<1x256x14x14xf16>, %arg124: tensor<1x256x14x14xf16>, %arg125: tensor<512x256x3x3xf16>, %arg126: tensor<1x512x7x7xf16>, %arg127: tensor<1x512x7x7xf16>, %arg128: tensor<512x512x3x3xf16>, %arg129: tensor<1x512x7x7xf16>, %arg130: tensor<512x256x1x1xf16>, %arg131: tensor<1x512x7x7xf16>, %arg132: tensor<1x512x7x7xf16>, %arg133: tensor<512x512x3x3xf16>, %arg134: tensor<1x512x7x7xf16>, %arg135: tensor<1x512x7x7xf16>, %arg136: tensor<512x512x3x3xf16>, %arg137: tensor<1x512x7x7xf16>, %arg138: tensor<1x512x7x7xf16>, %arg139: tensor<1x512xf16>, %arg140: tensor<512x1000xf16>, %arg141: tensor<1x1000xf16>) -> (tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>) attributes {__placeholder__byre.entry_point} {
    %0 = tensor.empty() : tensor<1x512xf16>
    %1 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} ins(%arg141, %arg140 : tensor<1x1000xf16>, tensor<512x1000xf16>) outs(%0 : tensor<1x512xf16>) : tensor<1x512xf16>
    %2 = call @Unknown0(%1, %arg138) : (tensor<1x512xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %3 = tensor.empty() : tensor<1x512x7x7xf16>
    %4 = tensor.empty() : tensor<512xf32>
    %5 = tensor.empty() : tensor<512xf32>
    %6:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg137, %arg39, %2 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) outs(%3, %4, %5 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %7 = tensor.empty() : tensor<1x512x7x7xf16>
    %8 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%6#0, %arg136 : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%7 : tensor<1x512x7x7xf16>) : tensor<1x512x7x7xf16>
    %9 = tensor.empty() : tensor<512x512x3x3xf16>
    %10 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg135, %6#0 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) outs(%9 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %11 = call @Unknown4(%arg135, %8) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %12 = tensor.empty() : tensor<1x512x7x7xf16>
    %13 = tensor.empty() : tensor<512xf32>
    %14 = tensor.empty() : tensor<512xf32>
    %15:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg134, %arg37, %11 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) outs(%12, %13, %14 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %16 = tensor.empty() : tensor<1x512x7x7xf16>
    %17 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%15#0, %arg133 : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%16 : tensor<1x512x7x7xf16>) : tensor<1x512x7x7xf16>
    %18 = tensor.empty() : tensor<512x512x3x3xf16>
    %19 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg132, %15#0 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) outs(%18 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %20 = call @Unknown8(%2, %17, %arg132) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %21 = tensor.empty() : tensor<1x512x7x7xf16>
    %22 = tensor.empty() : tensor<512xf32>
    %23 = tensor.empty() : tensor<512xf32>
    %24:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg129, %arg33, %20 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) outs(%21, %22, %23 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %25 = tensor.empty() : tensor<1x512x7x7xf16>
    %26 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%24#0, %arg128 : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%25 : tensor<1x512x7x7xf16>) : tensor<1x512x7x7xf16>
    %27 = tensor.empty() : tensor<512x512x3x3xf16>
    %28 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg127, %24#0 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) outs(%27 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %29 = call @Unknown12(%arg127, %26) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %30 = tensor.empty() : tensor<1x512x7x7xf16>
    %31 = tensor.empty() : tensor<512xf32>
    %32 = tensor.empty() : tensor<512xf32>
    %33:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg126, %arg31, %29 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) outs(%30, %31, %32 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %34 = tensor.empty() : tensor<1x256x14x14xf16>
    %35 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%33#0, %arg125 : tensor<1x512x7x7xf16>, tensor<512x256x3x3xf16>) outs(%34 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %36 = tensor.empty() : tensor<512x256x3x3xf16>
    %37 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%arg124, %33#0 : tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) outs(%36 : tensor<512x256x3x3xf16>) : tensor<512x256x3x3xf16>
    %38 = tensor.empty() : tensor<1x512x7x7xf16>
    %39 = tensor.empty() : tensor<512xf32>
    %40 = tensor.empty() : tensor<512xf32>
    %41:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg131, %arg35, %20 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) outs(%38, %39, %40 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %42 = tensor.empty() : tensor<1x256x14x14xf16>
    %43 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%41#0, %arg130 : tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>) outs(%42 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %44 = tensor.empty() : tensor<512x256x1x1xf16>
    %45 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%arg124, %41#0 : tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) outs(%44 : tensor<512x256x1x1xf16>) : tensor<512x256x1x1xf16>
    %46 = call @Unknown19(%43, %35, %arg124) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %47 = tensor.empty() : tensor<1x256x14x14xf16>
    %48 = tensor.empty() : tensor<256xf32>
    %49 = tensor.empty() : tensor<256xf32>
    %50:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg123, %arg29, %46 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) outs(%47, %48, %49 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %51 = tensor.empty() : tensor<1x256x14x14xf16>
    %52 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%50#0, %arg122 : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%51 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %53 = tensor.empty() : tensor<256x256x3x3xf16>
    %54 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg121, %50#0 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%53 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %55 = call @Unknown23(%arg121, %52) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %56 = tensor.empty() : tensor<1x256x14x14xf16>
    %57 = tensor.empty() : tensor<256xf32>
    %58 = tensor.empty() : tensor<256xf32>
    %59:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg120, %arg27, %55 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) outs(%56, %57, %58 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %60 = tensor.empty() : tensor<1x256x14x14xf16>
    %61 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%59#0, %arg119 : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%60 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %62 = tensor.empty() : tensor<256x256x3x3xf16>
    %63 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg118, %59#0 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%62 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %64 = call @Unknown27(%46, %61, %arg118) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %65 = tensor.empty() : tensor<1x256x14x14xf16>
    %66 = tensor.empty() : tensor<256xf32>
    %67 = tensor.empty() : tensor<256xf32>
    %68:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg115, %arg23, %64 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) outs(%65, %66, %67 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %69 = tensor.empty() : tensor<1x256x14x14xf16>
    %70 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%68#0, %arg114 : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%69 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %71 = tensor.empty() : tensor<256x256x3x3xf16>
    %72 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg113, %68#0 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) outs(%71 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %73 = call @Unknown31(%arg113, %70) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %74 = tensor.empty() : tensor<1x256x14x14xf16>
    %75 = tensor.empty() : tensor<256xf32>
    %76 = tensor.empty() : tensor<256xf32>
    %77:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg112, %arg21, %73 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) outs(%74, %75, %76 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %78 = tensor.empty() : tensor<1x128x28x28xf16>
    %79 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%77#0, %arg111 : tensor<1x256x14x14xf16>, tensor<256x128x3x3xf16>) outs(%78 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %80 = tensor.empty() : tensor<256x128x3x3xf16>
    %81 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%arg110, %77#0 : tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) outs(%80 : tensor<256x128x3x3xf16>) : tensor<256x128x3x3xf16>
    %82 = tensor.empty() : tensor<1x256x14x14xf16>
    %83 = tensor.empty() : tensor<256xf32>
    %84 = tensor.empty() : tensor<256xf32>
    %85:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg117, %arg25, %64 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) outs(%82, %83, %84 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %86 = tensor.empty() : tensor<1x128x28x28xf16>
    %87 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%85#0, %arg116 : tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>) outs(%86 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %88 = tensor.empty() : tensor<256x128x1x1xf16>
    %89 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%arg110, %85#0 : tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) outs(%88 : tensor<256x128x1x1xf16>) : tensor<256x128x1x1xf16>
    %90 = call @Unknown38(%87, %79, %arg110) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %91 = tensor.empty() : tensor<1x128x28x28xf16>
    %92 = tensor.empty() : tensor<128xf32>
    %93 = tensor.empty() : tensor<128xf32>
    %94:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg109, %arg19, %90 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) outs(%91, %92, %93 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %95 = tensor.empty() : tensor<1x128x28x28xf16>
    %96 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%94#0, %arg108 : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%95 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %97 = tensor.empty() : tensor<128x128x3x3xf16>
    %98 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg107, %94#0 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%97 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %99 = call @Unknown42(%arg107, %96) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %100 = tensor.empty() : tensor<1x128x28x28xf16>
    %101 = tensor.empty() : tensor<128xf32>
    %102 = tensor.empty() : tensor<128xf32>
    %103:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg106, %arg17, %99 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) outs(%100, %101, %102 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %104 = tensor.empty() : tensor<1x128x28x28xf16>
    %105 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%103#0, %arg105 : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%104 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %106 = tensor.empty() : tensor<128x128x3x3xf16>
    %107 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg104, %103#0 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%106 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %108 = call @Unknown46(%90, %105, %arg104) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %109 = tensor.empty() : tensor<1x128x28x28xf16>
    %110 = tensor.empty() : tensor<128xf32>
    %111 = tensor.empty() : tensor<128xf32>
    %112:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg101, %arg13, %108 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) outs(%109, %110, %111 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %113 = tensor.empty() : tensor<1x128x28x28xf16>
    %114 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%112#0, %arg100 : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%113 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %115 = tensor.empty() : tensor<128x128x3x3xf16>
    %116 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg99, %112#0 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) outs(%115 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %117 = call @Unknown50(%arg99, %114) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %118 = tensor.empty() : tensor<1x128x28x28xf16>
    %119 = tensor.empty() : tensor<128xf32>
    %120 = tensor.empty() : tensor<128xf32>
    %121:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg98, %arg11, %117 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) outs(%118, %119, %120 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %122 = tensor.empty() : tensor<1x64x56x56xf16>
    %123 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%121#0, %arg97 : tensor<1x128x28x28xf16>, tensor<128x64x3x3xf16>) outs(%122 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %124 = tensor.empty() : tensor<128x64x3x3xf16>
    %125 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%arg96, %121#0 : tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) outs(%124 : tensor<128x64x3x3xf16>) : tensor<128x64x3x3xf16>
    %126 = tensor.empty() : tensor<1x128x28x28xf16>
    %127 = tensor.empty() : tensor<128xf32>
    %128 = tensor.empty() : tensor<128xf32>
    %129:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg103, %arg15, %108 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) outs(%126, %127, %128 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %130 = tensor.empty() : tensor<1x64x56x56xf16>
    %131 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%129#0, %arg102 : tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>) outs(%130 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %132 = tensor.empty() : tensor<128x64x1x1xf16>
    %133 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%arg96, %129#0 : tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) outs(%132 : tensor<128x64x1x1xf16>) : tensor<128x64x1x1xf16>
    %134 = call @Unknown57(%131, %123, %arg96) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %135 = tensor.empty() : tensor<1x64x56x56xf16>
    %136 = tensor.empty() : tensor<64xf32>
    %137 = tensor.empty() : tensor<64xf32>
    %138:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg95, %arg9, %134 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16>) outs(%135, %136, %137 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %139 = tensor.empty() : tensor<1x64x56x56xf16>
    %140 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%138#0, %arg94 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%139 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %141 = tensor.empty() : tensor<64x64x3x3xf16>
    %142 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg93, %138#0 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%141 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %143 = call @Unknown61(%arg93, %140) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %144 = tensor.empty() : tensor<1x64x56x56xf16>
    %145 = tensor.empty() : tensor<64xf32>
    %146 = tensor.empty() : tensor<64xf32>
    %147:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg92, %arg7, %143 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16>) outs(%144, %145, %146 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %148 = tensor.empty() : tensor<1x64x56x56xf16>
    %149 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%147#0, %arg91 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%148 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %150 = tensor.empty() : tensor<64x64x3x3xf16>
    %151 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg90, %147#0 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%150 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %152 = call @Unknown65(%134, %149, %arg90) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %153 = tensor.empty() : tensor<1x64x56x56xf16>
    %154 = tensor.empty() : tensor<64xf32>
    %155 = tensor.empty() : tensor<64xf32>
    %156:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg89, %arg5, %152 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16>) outs(%153, %154, %155 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %157 = tensor.empty() : tensor<1x64x56x56xf16>
    %158 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%156#0, %arg88 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%157 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %159 = tensor.empty() : tensor<64x64x3x3xf16>
    %160 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg87, %156#0 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%159 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %161 = call @Unknown69(%arg87, %158) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %162 = tensor.empty() : tensor<1x64x56x56xf16>
    %163 = tensor.empty() : tensor<64xf32>
    %164 = tensor.empty() : tensor<64xf32>
    %165:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg86, %arg3, %161 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16>) outs(%162, %163, %164 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %166 = tensor.empty() : tensor<1x64x56x56xf16>
    %167 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%165#0, %arg85 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%166 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %168 = tensor.empty() : tensor<64x64x3x3xf16>
    %169 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%arg84, %165#0 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) outs(%168 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %170 = call @Unknown73(%152, %167) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %171 = tensor.empty() : tensor<1x64x112x112xf16>
    %172 = byre.compute_on_tensor @PoolMaxGradOp_f16f16_f16 {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} ins(%arg83, %170 : tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>) outs(%171 : tensor<1x64x112x112xf16>) : tensor<1x64x112x112xf16>
    %173 = call @Unknown74(%arg83, %172) : (tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %174 = tensor.empty() : tensor<1x64x112x112xf16>
    %175 = tensor.empty() : tensor<64xf32>
    %176 = tensor.empty() : tensor<64xf32>
    %177:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%arg82, %arg1, %173 : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<1x64x112x112xf16>) outs(%174, %175, %176 : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
    %178 = tensor.empty() : tensor<64x3x7x7xf16>
    %179 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%arg81, %177#0 : tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>) outs(%178 : tensor<64x3x7x7xf16>) : tensor<64x3x7x7xf16>
    %180 = call @Unknown77(%179) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %181 = call @Unknown78(%arg141) : (tensor<1x1000xf16>) -> tensor<1x1000xf32>
    %182 = tensor.empty() : tensor<1000xf32>
    %183 = byre.compute_on_tensor @ReduceSumOp_f32_f32 {dimensions = dense<0> : tensor<1xi64>} ins(%181 : tensor<1x1000xf32>) outs(%182 : tensor<1000xf32>) : tensor<1000xf32>
    %184 = call @Unknown79(%183) : (tensor<1000xf32>) -> tensor<1000xf32>
    %collapsed = tensor.collapse_shape %arg141 [[0, 1]] : tensor<1x1000xf16> into tensor<1000xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1000, 1] : tensor<1000xf16> into tensor<1000x1xf16>
    %185 = tensor.empty() : tensor<1000x512xf16>
    %186 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} ins(%expanded, %arg139 : tensor<1000x1xf16>, tensor<1x512xf16>) outs(%185 : tensor<1000x512xf16>) : tensor<1000x512xf16>
    %187 = call @Unknown80(%186) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %188 = call @Unknown81(%169) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %189 = call @Unknown82(%160) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %190 = call @Unknown83(%151) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %191 = call @Unknown84(%142) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %192 = call @Unknown85(%125) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %193 = call @Unknown86(%116) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %194 = call @Unknown87(%133) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %195 = call @Unknown88(%107) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %196 = call @Unknown89(%98) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %197 = call @Unknown90(%81) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %198 = call @Unknown91(%72) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %199 = call @Unknown92(%89) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %200 = call @Unknown93(%63) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %201 = call @Unknown94(%54) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %202 = call @Unknown95(%37) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %203 = call @Unknown96(%28) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %204 = call @Unknown97(%45) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %205 = call @Unknown98(%19) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %206 = call @Unknown99(%10) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %177#2, %177#1, %180, %184, %187, %165#2, %165#1, %156#2, %156#1, %188, %189, %147#2, %147#1, %138#2, %138#1, %190, %191, %121#2, %121#1, %112#2, %112#1, %192, %193, %194, %129#2, %129#1, %103#2, %103#1, %94#2, %94#1, %195, %196, %77#2, %77#1, %68#2, %68#1, %197, %198, %199, %85#2, %85#1, %59#2, %59#1, %50#2, %50#1, %200, %201, %33#2, %33#1, %24#2, %24#1, %202, %203, %204, %41#2, %41#1, %15#2, %15#1, %6#2, %6#1, %205, %206 : tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>
  }
}
