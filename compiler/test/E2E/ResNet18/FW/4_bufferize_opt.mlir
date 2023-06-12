// RUN: byteir-opt %s -byteir-bufferize-opt | FileCheck %s

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
  func.func private @Unknown18(%arg0: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<128x64x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x64x3x3xf32>) outs(%0 : tensor<128x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
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
  func.func private @Unknown32(%arg0: tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x128x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<256x128x3x3xf32>) outs(%0 : tensor<256x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
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
  func.func private @Unknown46(%arg0: tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<512x256x3x3xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x256x3x3xf32>) outs(%0 : tensor<512x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
    } -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
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
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<1000xf32>, %arg4: tensor<1000x512xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64x64x3x3xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64x64x3x3xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x64x3x3xf32>, %arg22: tensor<128x128x3x3xf32>, %arg23: tensor<128x64x1x1xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128x128x3x3xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x128x3x3xf32>, %arg37: tensor<256x256x3x3xf32>, %arg38: tensor<256x128x1x1xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<256xf32>, %arg44: tensor<256xf32>, %arg45: tensor<256x256x3x3xf32>, %arg46: tensor<256x256x3x3xf32>, %arg47: tensor<512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<512x256x3x3xf32>, %arg52: tensor<512x512x3x3xf32>, %arg53: tensor<512x256x1x1xf32>, %arg54: tensor<512xf32>, %arg55: tensor<512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<512xf32>, %arg59: tensor<512xf32>, %arg60: tensor<512x512x3x3xf32>, %arg61: tensor<512x512x3x3xf32>, %arg62: tensor<i64>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<i64>, %arg66: tensor<64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<i64>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<i64>, %arg72: tensor<64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<i64>, %arg75: tensor<64xf32>, %arg76: tensor<64xf32>, %arg77: tensor<i64>, %arg78: tensor<128xf32>, %arg79: tensor<128xf32>, %arg80: tensor<i64>, %arg81: tensor<128xf32>, %arg82: tensor<128xf32>, %arg83: tensor<i64>, %arg84: tensor<128xf32>, %arg85: tensor<128xf32>, %arg86: tensor<i64>, %arg87: tensor<128xf32>, %arg88: tensor<128xf32>, %arg89: tensor<i64>, %arg90: tensor<128xf32>, %arg91: tensor<128xf32>, %arg92: tensor<i64>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<i64>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<i64>, %arg99: tensor<256xf32>, %arg100: tensor<256xf32>, %arg101: tensor<i64>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<i64>, %arg105: tensor<256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<i64>, %arg108: tensor<512xf32>, %arg109: tensor<512xf32>, %arg110: tensor<i64>, %arg111: tensor<512xf32>, %arg112: tensor<512xf32>, %arg113: tensor<i64>, %arg114: tensor<512xf32>, %arg115: tensor<512xf32>, %arg116: tensor<i64>, %arg117: tensor<512xf32>, %arg118: tensor<512xf32>, %arg119: tensor<i64>, %arg120: tensor<512xf32>, %arg121: tensor<512xf32>, %arg122: tensor<1x3x224x224xf32>) -> (tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg122) : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16>
    %1 = call @Unknown1(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %2 = byre.compute @ConvOp_f16f16_f16(%0, %1) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16> -> tensor<1x64x112x112xf16>
    %3:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%2, %arg1, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
    %4 = call @Unknown3(%3#0) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %5 = byre.compute @PoolMaxOp_f16_f16(%4) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : tensor<1x64x112x112xf16> -> tensor<1x64x56x56xf16>
    %6 = call @Unknown4(%arg9) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %7 = byre.compute @ConvOp_f16f16_f16(%5, %6) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %8:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%7, %arg6, %arg5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %9 = call @Unknown6(%8#0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %10 = call @Unknown7(%arg10) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %11 = byre.compute @ConvOp_f16f16_f16(%9, %10) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %12:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%11, %arg8, %arg7) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %13 = call @Unknown9(%12#0, %5) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %14 = call @Unknown10(%arg15) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %15 = byre.compute @ConvOp_f16f16_f16(%13, %14) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %16:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%15, %arg12, %arg11) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %17 = call @Unknown12(%16#0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %18 = call @Unknown13(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %19 = byre.compute @ConvOp_f16f16_f16(%17, %18) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x56x56xf16>
    %20:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%19, %arg14, %arg13) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32> -> tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %21 = call @Unknown15(%20#0, %13) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %22 = call @Unknown16(%arg23) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %23 = byre.compute @ConvOp_f16f16_f16(%21, %22) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16> -> tensor<1x128x28x28xf16>
    %24:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%23, %arg25, %arg24) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %25 = call @Unknown18(%arg21) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %26 = byre.compute @ConvOp_f16f16_f16(%21, %25) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16> -> tensor<1x128x28x28xf16>
    %27:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%26, %arg18, %arg17) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %28 = call @Unknown20(%27#0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %29 = call @Unknown21(%arg22) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %30 = byre.compute @ConvOp_f16f16_f16(%28, %29) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<1x128x28x28xf16>
    %31:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%30, %arg20, %arg19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %32 = call @Unknown23(%31#0, %24#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %33 = call @Unknown24(%arg30) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %34 = byre.compute @ConvOp_f16f16_f16(%32, %33) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<1x128x28x28xf16>
    %35:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%34, %arg27, %arg26) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %36 = call @Unknown26(%35#0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %37 = call @Unknown27(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %38 = byre.compute @ConvOp_f16f16_f16(%36, %37) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16> -> tensor<1x128x28x28xf16>
    %39:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%38, %arg29, %arg28) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32> -> tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %40 = call @Unknown29(%39#0, %32) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %41 = call @Unknown30(%arg38) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %42 = byre.compute @ConvOp_f16f16_f16(%40, %41) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16> -> tensor<1x256x14x14xf16>
    %43:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%42, %arg40, %arg39) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %44 = call @Unknown32(%arg36) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %45 = byre.compute @ConvOp_f16f16_f16(%40, %44) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16> -> tensor<1x256x14x14xf16>
    %46:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%45, %arg33, %arg32) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %47 = call @Unknown34(%46#0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %48 = call @Unknown35(%arg37) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %49 = byre.compute @ConvOp_f16f16_f16(%47, %48) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<1x256x14x14xf16>
    %50:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%49, %arg35, %arg34) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %51 = call @Unknown37(%50#0, %43#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %52 = call @Unknown38(%arg45) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %53 = byre.compute @ConvOp_f16f16_f16(%51, %52) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<1x256x14x14xf16>
    %54:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%53, %arg42, %arg41) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %55 = call @Unknown40(%54#0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %56 = call @Unknown41(%arg46) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %57 = byre.compute @ConvOp_f16f16_f16(%55, %56) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16> -> tensor<1x256x14x14xf16>
    %58:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%57, %arg44, %arg43) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32> -> tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %59 = call @Unknown43(%58#0, %51) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %60 = call @Unknown44(%arg53) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %61 = byre.compute @ConvOp_f16f16_f16(%59, %60) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16> -> tensor<1x512x7x7xf16>
    %62:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%61, %arg55, %arg54) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %63 = call @Unknown46(%arg51) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %64 = byre.compute @ConvOp_f16f16_f16(%59, %63) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16> -> tensor<1x512x7x7xf16>
    %65:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%64, %arg48, %arg47) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %66 = call @Unknown48(%65#0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %67 = call @Unknown49(%arg52) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %68 = byre.compute @ConvOp_f16f16_f16(%66, %67) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<1x512x7x7xf16>
    %69:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%68, %arg50, %arg49) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %70 = call @Unknown51(%69#0, %62#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %71 = call @Unknown52(%arg60) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %72 = byre.compute @ConvOp_f16f16_f16(%70, %71) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<1x512x7x7xf16>
    %73:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%72, %arg57, %arg56) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %74 = call @Unknown54(%73#0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %75 = call @Unknown55(%arg61) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %76 = byre.compute @ConvOp_f16f16_f16(%74, %75) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16> -> tensor<1x512x7x7xf16>
    %77:3 = byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%76, %arg59, %arg58) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32> -> tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %78 = call @Unknown57(%77#0, %70) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %79 = byre.compute @ReduceSumOp_f16_f16(%78) {dimensions = dense<[3, 2]> : tensor<2xi64>} : tensor<1x512x7x7xf16> -> tensor<1x512xf16>
    %80 = call @Unknown58(%79) : (tensor<1x512xf16>) -> tensor<1x512xf16>
    %81 = call @Unknown59(%arg4) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %82 = byre.compute @TransposeOp_f16_f16(%81) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : tensor<1000x512xf16> -> tensor<512x1000xf16>
    %83 = byre.compute @MatmulOp_f16f16_f16(%80, %81) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : tensor<1x512xf16>, tensor<1000x512xf16> -> tensor<1x1000xf16>
    %84 = call @Unknown60(%arg3, %83) : (tensor<1000xf32>, tensor<1x1000xf16>) -> tensor<1x1000xf16>
    %85 = call @Unknown61(%3#1, %arg63) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %86 = call @Unknown62(%3#2, %arg64) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %87 = call @Unknown63(%8#1, %arg66) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %88 = call @Unknown64(%8#2, %arg67) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %89 = call @Unknown65(%12#1, %arg69) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %90 = call @Unknown66(%12#2, %arg70) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %91 = call @Unknown67(%16#1, %arg72) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %92 = call @Unknown68(%16#2, %arg73) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %93 = call @Unknown69(%20#1, %arg75) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %94 = call @Unknown70(%20#2, %arg76) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %95 = call @Unknown71(%27#1, %arg78) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %96 = call @Unknown72(%27#2, %arg79) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %97 = call @Unknown73(%31#1, %arg81) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %98 = call @Unknown74(%31#2, %arg82) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %99 = call @Unknown75(%24#1, %arg84) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %100 = call @Unknown76(%24#2, %arg85) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %101 = call @Unknown77(%35#1, %arg87) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %102 = call @Unknown78(%35#2, %arg88) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %103 = call @Unknown79(%39#1, %arg90) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %104 = call @Unknown80(%39#2, %arg91) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %105 = call @Unknown81(%46#1, %arg93) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %106 = call @Unknown82(%46#2, %arg94) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %107 = call @Unknown83(%50#1, %arg96) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %108 = call @Unknown84(%50#2, %arg97) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %109 = call @Unknown85(%43#1, %arg99) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %110 = call @Unknown86(%43#2, %arg100) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %111 = call @Unknown87(%54#1, %arg102) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %112 = call @Unknown88(%54#2, %arg103) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %113 = call @Unknown89(%58#1, %arg105) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %114 = call @Unknown90(%58#2, %arg106) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %115 = call @Unknown91(%65#1, %arg108) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %116 = call @Unknown92(%65#2, %arg109) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %117 = call @Unknown93(%69#1, %arg111) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %118 = call @Unknown94(%69#2, %arg112) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %119 = call @Unknown95(%62#1, %arg114) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %120 = call @Unknown96(%62#2, %arg115) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %121 = call @Unknown97(%73#1, %arg117) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %122 = call @Unknown98(%73#2, %arg118) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %123 = call @Unknown99(%77#1, %arg120) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %124 = call @Unknown100(%77#2, %arg121) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    return %84, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %1, %0, %2, %4, %5, %6, %7, %9, %10, %11, %13, %14, %15, %17, %18, %19, %21, %25, %26, %28, %29, %30, %22, %23, %32, %33, %34, %36, %37, %38, %40, %44, %45, %47, %48, %49, %41, %42, %51, %52, %53, %55, %56, %57, %59, %63, %64, %66, %67, %68, %60, %61, %70, %71, %72, %74, %75, %76, %78, %80, %82 : tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>
  }
}