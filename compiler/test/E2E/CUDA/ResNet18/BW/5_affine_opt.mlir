// RUN: byteir-opt %s -affine-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0)>
module {
  func.func private @Unknown0(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.900000e+01 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg0 : memref<1x512x7x7xf16>, memref<1x512xf16>) outs(%alloc : memref<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %0 = arith.divf %in_1, %cst : f16
      %1 = arith.cmpf ogt, %in, %cst_0 : f16
      %2 = arith.select %1, %0, %cst_0 : f16
      linalg.yield %2 : f16
    }
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @Unknown4(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) outs(%alloc : memref<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @Unknown8(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) outs(%alloc : memref<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.cmpf ogt, %in, %cst : f16
      %2 = arith.select %1, %0, %cst : f16
      linalg.yield %2 : f16
    }
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @Unknown12(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) outs(%alloc : memref<1x512x7x7xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @Unknown19(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%alloc : memref<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.cmpf ogt, %in, %cst : f16
      %2 = arith.select %1, %0, %cst : f16
      linalg.yield %2 : f16
    }
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @Unknown23(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%alloc : memref<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @Unknown27(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%alloc : memref<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.cmpf ogt, %in, %cst : f16
      %2 = arith.select %1, %0, %cst : f16
      linalg.yield %2 : f16
    }
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @Unknown31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%alloc : memref<1x256x14x14xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @Unknown38(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%alloc : memref<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.cmpf ogt, %in, %cst : f16
      %2 = arith.select %1, %0, %cst : f16
      linalg.yield %2 : f16
    }
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @Unknown42(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%alloc : memref<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @Unknown46(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%alloc : memref<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.cmpf ogt, %in, %cst : f16
      %2 = arith.select %1, %0, %cst : f16
      linalg.yield %2 : f16
    }
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @Unknown50(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%alloc : memref<1x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @Unknown57(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%alloc : memref<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.cmpf ogt, %in, %cst : f16
      %2 = arith.select %1, %0, %cst : f16
      linalg.yield %2 : f16
    }
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @Unknown61(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%alloc : memref<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @Unknown65(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%alloc : memref<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.cmpf ogt, %in, %cst : f16
      %2 = arith.select %1, %0, %cst : f16
      linalg.yield %2 : f16
    }
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @Unknown69(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%alloc : memref<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @Unknown73(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%alloc : memref<1x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.addf %in, %in_0 : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @Unknown74(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<1x64x112x112xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x112x112xf16>, memref<1x64x112x112xf16>) outs(%alloc : memref<1x64x112x112xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.cmpf ogt, %in, %cst : f16
      %1 = arith.select %0, %in_0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<1x64x112x112xf16>
  }
  func.func private @Unknown77(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x3x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x3x7x7xf16>) outs(%alloc : memref<64x3x7x7xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x3x7x7xf32>
  }
  func.func private @Unknown78(%arg0: memref<1x1000xf16>) -> memref<1x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<1x1000xf32>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x1000xf16>) outs(%alloc : memref<1x1000xf32>) {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<1x1000xf32>
  }
  func.func private @Unknown79(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<1000xf32>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg0 : memref<1000xf32>) outs(%alloc : memref<1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.truncf %in : f32 to f16
      %1 = arith.extf %0 : f16 to f32
      linalg.yield %1 : f32
    }
    return %alloc : memref<1000xf32>
  }
  func.func private @Unknown80(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<1000x512xf32>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1000x512xf16>) outs(%alloc : memref<1000x512xf32>) {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<1000x512xf32>
  }
  func.func private @Unknown81(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%alloc : memref<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown82(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%alloc : memref<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown83(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%alloc : memref<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown84(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%alloc : memref<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown85(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x3x3xf16>) outs(%alloc : memref<128x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x64x3x3xf32>
  }
  func.func private @Unknown86(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%alloc : memref<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown87(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x64x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x1x1xf16>) outs(%alloc : memref<128x64x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x64x1x1xf32>
  }
  func.func private @Unknown88(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%alloc : memref<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown89(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%alloc : memref<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown90(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x128x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x3x3xf16>) outs(%alloc : memref<256x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x128x3x3xf32>
  }
  func.func private @Unknown91(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%alloc : memref<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown92(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x128x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x1x1xf16>) outs(%alloc : memref<256x128x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x128x1x1xf32>
  }
  func.func private @Unknown93(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%alloc : memref<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown94(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%alloc : memref<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown95(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x256x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x3x3xf16>) outs(%alloc : memref<512x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x256x3x3xf32>
  }
  func.func private @Unknown96(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%alloc : memref<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown97(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x256x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x1x1xf16>) outs(%alloc : memref<512x256x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x256x1x1xf32>
  }
  func.func private @Unknown98(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%alloc : memref<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown99(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%alloc : memref<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func @main(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<128xf32>, %arg11: memref<128xf32>, %arg12: memref<128xf32>, %arg13: memref<128xf32>, %arg14: memref<128xf32>, %arg15: memref<128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<128xf32>, %arg20: memref<256xf32>, %arg21: memref<256xf32>, %arg22: memref<256xf32>, %arg23: memref<256xf32>, %arg24: memref<256xf32>, %arg25: memref<256xf32>, %arg26: memref<256xf32>, %arg27: memref<256xf32>, %arg28: memref<256xf32>, %arg29: memref<256xf32>, %arg30: memref<512xf32>, %arg31: memref<512xf32>, %arg32: memref<512xf32>, %arg33: memref<512xf32>, %arg34: memref<512xf32>, %arg35: memref<512xf32>, %arg36: memref<512xf32>, %arg37: memref<512xf32>, %arg38: memref<512xf32>, %arg39: memref<512xf32>, %arg40: memref<64xf32>, %arg41: memref<64xf32>, %arg42: memref<64xf32>, %arg43: memref<64xf32>, %arg44: memref<64xf32>, %arg45: memref<64xf32>, %arg46: memref<64xf32>, %arg47: memref<64xf32>, %arg48: memref<64xf32>, %arg49: memref<64xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<128xf32>, %arg53: memref<128xf32>, %arg54: memref<128xf32>, %arg55: memref<128xf32>, %arg56: memref<128xf32>, %arg57: memref<128xf32>, %arg58: memref<128xf32>, %arg59: memref<128xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<512xf32>, %arg71: memref<512xf32>, %arg72: memref<512xf32>, %arg73: memref<512xf32>, %arg74: memref<512xf32>, %arg75: memref<512xf32>, %arg76: memref<512xf32>, %arg77: memref<512xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<64x3x7x7xf16>, %arg81: memref<1x3x224x224xf16>, %arg82: memref<1x64x112x112xf16>, %arg83: memref<1x64x112x112xf16>, %arg84: memref<1x64x56x56xf16>, %arg85: memref<64x64x3x3xf16>, %arg86: memref<1x64x56x56xf16>, %arg87: memref<1x64x56x56xf16>, %arg88: memref<64x64x3x3xf16>, %arg89: memref<1x64x56x56xf16>, %arg90: memref<1x64x56x56xf16>, %arg91: memref<64x64x3x3xf16>, %arg92: memref<1x64x56x56xf16>, %arg93: memref<1x64x56x56xf16>, %arg94: memref<64x64x3x3xf16>, %arg95: memref<1x64x56x56xf16>, %arg96: memref<1x64x56x56xf16>, %arg97: memref<128x64x3x3xf16>, %arg98: memref<1x128x28x28xf16>, %arg99: memref<1x128x28x28xf16>, %arg100: memref<128x128x3x3xf16>, %arg101: memref<1x128x28x28xf16>, %arg102: memref<128x64x1x1xf16>, %arg103: memref<1x128x28x28xf16>, %arg104: memref<1x128x28x28xf16>, %arg105: memref<128x128x3x3xf16>, %arg106: memref<1x128x28x28xf16>, %arg107: memref<1x128x28x28xf16>, %arg108: memref<128x128x3x3xf16>, %arg109: memref<1x128x28x28xf16>, %arg110: memref<1x128x28x28xf16>, %arg111: memref<256x128x3x3xf16>, %arg112: memref<1x256x14x14xf16>, %arg113: memref<1x256x14x14xf16>, %arg114: memref<256x256x3x3xf16>, %arg115: memref<1x256x14x14xf16>, %arg116: memref<256x128x1x1xf16>, %arg117: memref<1x256x14x14xf16>, %arg118: memref<1x256x14x14xf16>, %arg119: memref<256x256x3x3xf16>, %arg120: memref<1x256x14x14xf16>, %arg121: memref<1x256x14x14xf16>, %arg122: memref<256x256x3x3xf16>, %arg123: memref<1x256x14x14xf16>, %arg124: memref<1x256x14x14xf16>, %arg125: memref<512x256x3x3xf16>, %arg126: memref<1x512x7x7xf16>, %arg127: memref<1x512x7x7xf16>, %arg128: memref<512x512x3x3xf16>, %arg129: memref<1x512x7x7xf16>, %arg130: memref<512x256x1x1xf16>, %arg131: memref<1x512x7x7xf16>, %arg132: memref<1x512x7x7xf16>, %arg133: memref<512x512x3x3xf16>, %arg134: memref<1x512x7x7xf16>, %arg135: memref<1x512x7x7xf16>, %arg136: memref<512x512x3x3xf16>, %arg137: memref<1x512x7x7xf16>, %arg138: memref<1x512x7x7xf16>, %arg139: memref<1x512xf16>, %arg140: memref<512x1000xf16>, %arg141: memref<1x1000xf16>) -> (memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>) attributes {__placeholder__byre.entry_point} {
    %alloc = memref.alloc() : memref<1x512xf16>
    byre.compute @MatmulOp_f16f16_f16(%arg141, %arg140, %alloc) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<1x1000xf16>, memref<512x1000xf16>, memref<1x512xf16>
    %0 = call @Unknown0(%alloc, %arg138) : (memref<1x512xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_1 = memref.alloc() : memref<512xf32>
    %alloc_2 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg137, %arg39, %0, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_3 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_0, %arg136, %alloc_3) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %alloc_4 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg135, %alloc_0, %alloc_4) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>
    %1 = call @Unknown4(%arg135, %alloc_3) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_6 = memref.alloc() : memref<512xf32>
    %alloc_7 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg134, %arg37, %1, %alloc_5, %alloc_6, %alloc_7) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_8 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_5, %arg133, %alloc_8) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %alloc_9 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg132, %alloc_5, %alloc_9) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>
    %2 = call @Unknown8(%0, %alloc_8, %arg132) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %alloc_10 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_11 = memref.alloc() : memref<512xf32>
    %alloc_12 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg129, %arg33, %2, %alloc_10, %alloc_11, %alloc_12) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_13 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_10, %arg128, %alloc_13) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %alloc_14 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg127, %alloc_10, %alloc_14) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>
    %3 = call @Unknown12(%arg127, %alloc_13) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %alloc_15 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_16 = memref.alloc() : memref<512xf32>
    %alloc_17 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg126, %arg31, %3, %alloc_15, %alloc_16, %alloc_17) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_18 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_15, %arg125, %alloc_18) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x256x3x3xf16>, memref<1x256x14x14xf16>
    %alloc_19 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg124, %alloc_15, %alloc_19) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<512x256x3x3xf16>
    %alloc_20 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_21 = memref.alloc() : memref<512xf32>
    %alloc_22 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg131, %arg35, %2, %alloc_20, %alloc_21, %alloc_22) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_23 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_20, %arg130, %alloc_23) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x256x14x14xf16>
    %alloc_24 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg124, %alloc_20, %alloc_24) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>
    %4 = call @Unknown19(%alloc_23, %alloc_18, %arg124) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %alloc_25 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_26 = memref.alloc() : memref<256xf32>
    %alloc_27 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg123, %arg29, %4, %alloc_25, %alloc_26, %alloc_27) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_28 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_25, %arg122, %alloc_28) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %alloc_29 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg121, %alloc_25, %alloc_29) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>
    %5 = call @Unknown23(%arg121, %alloc_28) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %alloc_30 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_31 = memref.alloc() : memref<256xf32>
    %alloc_32 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg120, %arg27, %5, %alloc_30, %alloc_31, %alloc_32) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_33 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_30, %arg119, %alloc_33) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %alloc_34 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg118, %alloc_30, %alloc_34) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>
    %6 = call @Unknown27(%4, %alloc_33, %arg118) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %alloc_35 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_36 = memref.alloc() : memref<256xf32>
    %alloc_37 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg115, %arg23, %6, %alloc_35, %alloc_36, %alloc_37) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_38 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_35, %arg114, %alloc_38) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %alloc_39 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg113, %alloc_35, %alloc_39) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>
    %7 = call @Unknown31(%arg113, %alloc_38) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %alloc_40 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_41 = memref.alloc() : memref<256xf32>
    %alloc_42 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg112, %arg21, %7, %alloc_40, %alloc_41, %alloc_42) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_43 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_40, %arg111, %alloc_43) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x128x3x3xf16>, memref<1x128x28x28xf16>
    %alloc_44 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg110, %alloc_40, %alloc_44) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<256x128x3x3xf16>
    %alloc_45 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_46 = memref.alloc() : memref<256xf32>
    %alloc_47 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg117, %arg25, %6, %alloc_45, %alloc_46, %alloc_47) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_48 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_45, %arg116, %alloc_48) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x128x28x28xf16>
    %alloc_49 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg110, %alloc_45, %alloc_49) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>
    %8 = call @Unknown38(%alloc_48, %alloc_43, %arg110) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %alloc_50 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_51 = memref.alloc() : memref<128xf32>
    %alloc_52 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg109, %arg19, %8, %alloc_50, %alloc_51, %alloc_52) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_53 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_50, %arg108, %alloc_53) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %alloc_54 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg107, %alloc_50, %alloc_54) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>
    %9 = call @Unknown42(%arg107, %alloc_53) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %alloc_55 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_56 = memref.alloc() : memref<128xf32>
    %alloc_57 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg106, %arg17, %9, %alloc_55, %alloc_56, %alloc_57) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_58 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_55, %arg105, %alloc_58) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %alloc_59 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg104, %alloc_55, %alloc_59) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>
    %10 = call @Unknown46(%8, %alloc_58, %arg104) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %alloc_60 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_61 = memref.alloc() : memref<128xf32>
    %alloc_62 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg101, %arg13, %10, %alloc_60, %alloc_61, %alloc_62) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_63 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_60, %arg100, %alloc_63) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %alloc_64 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg99, %alloc_60, %alloc_64) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>
    %11 = call @Unknown50(%arg99, %alloc_63) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %alloc_65 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_66 = memref.alloc() : memref<128xf32>
    %alloc_67 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg98, %arg11, %11, %alloc_65, %alloc_66, %alloc_67) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_68 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_65, %arg97, %alloc_68) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_69 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg96, %alloc_65, %alloc_69) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<128x64x3x3xf16>
    %alloc_70 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_71 = memref.alloc() : memref<128xf32>
    %alloc_72 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg103, %arg15, %10, %alloc_70, %alloc_71, %alloc_72) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_73 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_70, %arg102, %alloc_73) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x64x56x56xf16>
    %alloc_74 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg96, %alloc_70, %alloc_74) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>
    %12 = call @Unknown57(%alloc_73, %alloc_68, %arg96) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %alloc_75 = memref.alloc() : memref<1x64x56x56xf16>
    %alloc_76 = memref.alloc() : memref<64xf32>
    %alloc_77 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg95, %arg9, %12, %alloc_75, %alloc_76, %alloc_77) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_78 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_75, %arg94, %alloc_78) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_79 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg93, %alloc_75, %alloc_79) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>
    %13 = call @Unknown61(%arg93, %alloc_78) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %alloc_80 = memref.alloc() : memref<1x64x56x56xf16>
    %alloc_81 = memref.alloc() : memref<64xf32>
    %alloc_82 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg92, %arg7, %13, %alloc_80, %alloc_81, %alloc_82) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_83 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_80, %arg91, %alloc_83) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_84 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg90, %alloc_80, %alloc_84) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>
    %14 = call @Unknown65(%12, %alloc_83, %arg90) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %alloc_85 = memref.alloc() : memref<1x64x56x56xf16>
    %alloc_86 = memref.alloc() : memref<64xf32>
    %alloc_87 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg89, %arg5, %14, %alloc_85, %alloc_86, %alloc_87) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_88 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_85, %arg88, %alloc_88) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_89 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg87, %alloc_85, %alloc_89) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>
    %15 = call @Unknown69(%arg87, %alloc_88) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %alloc_90 = memref.alloc() : memref<1x64x56x56xf16>
    %alloc_91 = memref.alloc() : memref<64xf32>
    %alloc_92 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg86, %arg3, %15, %alloc_90, %alloc_91, %alloc_92) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_93 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_90, %arg85, %alloc_93) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_94 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg84, %alloc_90, %alloc_94) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>
    %16 = call @Unknown73(%14, %alloc_93) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %alloc_95 = memref.alloc() : memref<1x64x112x112xf16>
    byre.compute @PoolMaxGradOp_f16f16_f16(%arg83, %16, %alloc_95) {memory_effects = [1 : i32, 1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<1x64x112x112xf16>
    %17 = call @Unknown74(%arg83, %alloc_95) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    %alloc_96 = memref.alloc() : memref<1x64x112x112xf16>
    %alloc_97 = memref.alloc() : memref<64xf32>
    %alloc_98 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg82, %arg1, %17, %alloc_96, %alloc_97, %alloc_98) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x112x112xf16>, memref<64xf32>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    %alloc_99 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg81, %alloc_96, %alloc_99) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<64x3x7x7xf16>
    %18 = call @Unknown77(%alloc_99) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %19 = call @Unknown78(%arg141) : (memref<1x1000xf16>) -> memref<1x1000xf32>
    %alloc_100 = memref.alloc() : memref<1000xf32>
    byre.compute @ReduceSumOp_f32_f32(%19, %alloc_100) {dimensions = dense<0> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<1x1000xf32>, memref<1000xf32>
    %20 = call @Unknown79(%alloc_100) : (memref<1000xf32>) -> memref<1000xf32>
    %collapse_shape = memref.collapse_shape %arg141 [[0, 1]] : memref<1x1000xf16> into memref<1000xf16>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [1000, 1] : memref<1000xf16> into memref<1000x1xf16>
    %alloc_101 = memref.alloc() : memref<1000x512xf16>
    byre.compute @MatmulOp_f16f16_f16(%expand_shape, %arg139, %alloc_101) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<1000x1xf16>, memref<1x512xf16>, memref<1000x512xf16>
    %21 = call @Unknown80(%alloc_101) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %22 = call @Unknown81(%alloc_94) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %23 = call @Unknown82(%alloc_89) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %24 = call @Unknown83(%alloc_84) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %25 = call @Unknown84(%alloc_79) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %26 = call @Unknown85(%alloc_69) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %27 = call @Unknown86(%alloc_64) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %28 = call @Unknown87(%alloc_74) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %29 = call @Unknown88(%alloc_59) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %30 = call @Unknown89(%alloc_54) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %31 = call @Unknown90(%alloc_44) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %32 = call @Unknown91(%alloc_39) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %33 = call @Unknown92(%alloc_49) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %34 = call @Unknown93(%alloc_34) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %35 = call @Unknown94(%alloc_29) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %36 = call @Unknown95(%alloc_19) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %37 = call @Unknown96(%alloc_14) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %38 = call @Unknown97(%alloc_24) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %39 = call @Unknown98(%alloc_9) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %40 = call @Unknown99(%alloc_4) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    return %alloc_98, %alloc_97, %18, %20, %21, %alloc_92, %alloc_91, %alloc_87, %alloc_86, %22, %23, %alloc_82, %alloc_81, %alloc_77, %alloc_76, %24, %25, %alloc_67, %alloc_66, %alloc_62, %alloc_61, %26, %27, %28, %alloc_72, %alloc_71, %alloc_57, %alloc_56, %alloc_52, %alloc_51, %29, %30, %alloc_42, %alloc_41, %alloc_37, %alloc_36, %31, %32, %33, %alloc_47, %alloc_46, %alloc_32, %alloc_31, %alloc_27, %alloc_26, %34, %35, %alloc_17, %alloc_16, %alloc_12, %alloc_11, %36, %37, %38, %alloc_22, %alloc_21, %alloc_7, %alloc_6, %alloc_2, %alloc_1, %39, %40 : memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>
  }
}