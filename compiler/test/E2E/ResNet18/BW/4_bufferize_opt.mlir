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
    %0 = byre.compute @MatmulOp_f16f16_f16(%arg141, %arg140) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : tensor<1x1000xf16>, tensor<512x1000xf16> -> tensor<1x512xf16>
    %1 = call @Unknown0(%0, %arg138) : (tensor<1x512xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %2:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg137, %arg39, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %3 = byre.compute @ConvBackwardDataOp_f16f16_f16(%2#0, %arg136) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<1x512x7x7xf16>
    %4 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg135, %2#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16> -> tensor<512x512x3x3xf16>
    %5 = call @Unknown4(%arg135, %3) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %6:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg134, %arg37, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %7 = byre.compute @ConvBackwardDataOp_f16f16_f16(%6#0, %arg133) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<1x512x7x7xf16>
    %8 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg132, %6#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16> -> tensor<512x512x3x3xf16>
    %9 = call @Unknown8(%1, %7, %arg132) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %10:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg129, %arg33, %9) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %11 = byre.compute @ConvBackwardDataOp_f16f16_f16(%10#0, %arg128) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<1x512x7x7xf16>
    %12 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg127, %10#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16> -> tensor<512x512x3x3xf16>
    %13 = call @Unknown12(%arg127, %11) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %14:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg126, %arg31, %13) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %15 = byre.compute @ConvBackwardDataOp_f16f16_f16(%14#0, %arg125) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<512x256x3x3xf16> -> tensor<1x256x14x14xf16>
    %16 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg124, %14#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16> -> tensor<512x256x3x3xf16>
    %17:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg131, %arg35, %9) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %18 = byre.compute @ConvBackwardDataOp_f16f16_f16(%17#0, %arg130) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16> -> tensor<1x256x14x14xf16>
    %19 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg124, %17#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16> -> tensor<512x256x1x1xf16>
    %20 = call @Unknown19(%18, %15, %arg124) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %21:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg123, %arg29, %20) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %22 = byre.compute @ConvBackwardDataOp_f16f16_f16(%21#0, %arg122) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<1x256x14x14xf16>
    %23 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg121, %21#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16> -> tensor<256x256x3x3xf16>
    %24 = call @Unknown23(%arg121, %22) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %25:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg120, %arg27, %24) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %26 = byre.compute @ConvBackwardDataOp_f16f16_f16(%25#0, %arg119) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<1x256x14x14xf16>
    %27 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg118, %25#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16> -> tensor<256x256x3x3xf16>
    %28 = call @Unknown27(%20, %26, %arg118) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %29:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg115, %arg23, %28) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %30 = byre.compute @ConvBackwardDataOp_f16f16_f16(%29#0, %arg114) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<1x256x14x14xf16>
    %31 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg113, %29#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16> -> tensor<256x256x3x3xf16>
    %32 = call @Unknown31(%arg113, %30) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %33:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg112, %arg21, %32) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %34 = byre.compute @ConvBackwardDataOp_f16f16_f16(%33#0, %arg111) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<256x128x3x3xf16> -> tensor<1x128x28x28xf16>
    %35 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg110, %33#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16> -> tensor<256x128x3x3xf16>
    %36:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg117, %arg25, %28) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %37 = byre.compute @ConvBackwardDataOp_f16f16_f16(%36#0, %arg116) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16> -> tensor<1x128x28x28xf16>
    %38 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg110, %36#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16> -> tensor<256x128x1x1xf16>
    %39 = call @Unknown38(%37, %34, %arg110) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %40:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg109, %arg19, %39) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %41 = byre.compute @ConvBackwardDataOp_f16f16_f16(%40#0, %arg108) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<1x128x28x28xf16>
    %42 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg107, %40#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16> -> tensor<128x128x3x3xf16>
    %43 = call @Unknown42(%arg107, %41) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %44:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg106, %arg17, %43) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %45 = byre.compute @ConvBackwardDataOp_f16f16_f16(%44#0, %arg105) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<1x128x28x28xf16>
    %46 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg104, %44#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16> -> tensor<128x128x3x3xf16>
    %47 = call @Unknown46(%39, %45, %arg104) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %48:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg101, %arg13, %47) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %49 = byre.compute @ConvBackwardDataOp_f16f16_f16(%48#0, %arg100) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<1x128x28x28xf16>
    %50 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg99, %48#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16> -> tensor<128x128x3x3xf16>
    %51 = call @Unknown50(%arg99, %49) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %52:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg98, %arg11, %51) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %53 = byre.compute @ConvBackwardDataOp_f16f16_f16(%52#0, %arg97) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<128x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %54 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg96, %52#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16> -> tensor<128x64x3x3xf16>
    %55:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg103, %arg15, %47) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %56 = byre.compute @ConvBackwardDataOp_f16f16_f16(%55#0, %arg102) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16> -> tensor<1x64x56x56xf16>
    %57 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg96, %55#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16> -> tensor<128x64x1x1xf16>
    %58 = call @Unknown57(%56, %53, %arg96) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %59:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg95, %arg9, %58) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16> -> tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %60 = byre.compute @ConvBackwardDataOp_f16f16_f16(%59#0, %arg94) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %61 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg93, %59#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16> -> tensor<64x64x3x3xf16>
    %62 = call @Unknown61(%arg93, %60) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %63:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg92, %arg7, %62) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16> -> tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %64 = byre.compute @ConvBackwardDataOp_f16f16_f16(%63#0, %arg91) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %65 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg90, %63#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16> -> tensor<64x64x3x3xf16>
    %66 = call @Unknown65(%58, %64, %arg90) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %67:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg89, %arg5, %66) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16> -> tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %68 = byre.compute @ConvBackwardDataOp_f16f16_f16(%67#0, %arg88) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %69 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg87, %67#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16> -> tensor<64x64x3x3xf16>
    %70 = call @Unknown69(%arg87, %68) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %71:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg86, %arg3, %70) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16> -> tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %72 = byre.compute @ConvBackwardDataOp_f16f16_f16(%71#0, %arg85) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %73 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg84, %71#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16> -> tensor<64x64x3x3xf16>
    %74 = call @Unknown73(%66, %72) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %75 = byre.compute @PoolMaxGradOp_f16f16_f16(%arg83, %74) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16> -> tensor<1x64x112x112xf16>
    %76 = call @Unknown74(%arg83, %75) : (tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %77:3 = byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg82, %arg1, %76) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<1x64x112x112xf16> -> tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
    %78 = byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg81, %77#0) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16> -> tensor<64x3x7x7xf16>
    %79 = call @Unknown77(%78) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %80 = call @Unknown78(%arg141) : (tensor<1x1000xf16>) -> tensor<1x1000xf32>
    %81 = byre.compute @ReduceSumOp_f32_f32(%80) {dimensions = dense<0> : tensor<1xi64>} : tensor<1x1000xf32> -> tensor<1000xf32>
    %82 = call @Unknown79(%81) : (tensor<1000xf32>) -> tensor<1000xf32>
    %collapsed = tensor.collapse_shape %arg141 [[0, 1]] : tensor<1x1000xf16> into tensor<1000xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] : tensor<1000xf16> into tensor<1000x1xf16>
    %83 = byre.compute @MatmulOp_f16f16_f16(%expanded, %arg139) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : tensor<1000x1xf16>, tensor<1x512xf16> -> tensor<1000x512xf16>
    %84 = call @Unknown80(%83) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %85 = call @Unknown81(%73) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %86 = call @Unknown82(%69) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %87 = call @Unknown83(%65) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %88 = call @Unknown84(%61) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %89 = call @Unknown85(%54) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %90 = call @Unknown86(%50) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %91 = call @Unknown87(%57) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %92 = call @Unknown88(%46) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %93 = call @Unknown89(%42) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %94 = call @Unknown90(%35) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %95 = call @Unknown91(%31) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %96 = call @Unknown92(%38) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %97 = call @Unknown93(%27) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %98 = call @Unknown94(%23) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %99 = call @Unknown95(%16) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %100 = call @Unknown96(%12) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %101 = call @Unknown97(%19) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %102 = call @Unknown98(%8) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %103 = call @Unknown99(%4) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %77#2, %77#1, %79, %82, %84, %71#2, %71#1, %67#2, %67#1, %85, %86, %63#2, %63#1, %59#2, %59#1, %87, %88, %52#2, %52#1, %48#2, %48#1, %89, %90, %91, %55#2, %55#1, %44#2, %44#1, %40#2, %40#1, %92, %93, %33#2, %33#1, %29#2, %29#1, %94, %95, %96, %36#2, %36#1, %25#2, %25#1, %21#2, %21#1, %97, %98, %14#2, %14#1, %10#2, %10#1, %99, %100, %101, %17#2, %17#1, %6#2, %6#1, %2#2, %2#1, %102, %103 : tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>
  }
}