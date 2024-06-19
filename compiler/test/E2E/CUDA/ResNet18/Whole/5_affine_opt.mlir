// RUN: byteir-opt %s -affine-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map5 = affine_map<() -> ()>
#map6 = affine_map<(d0) -> (d0)>
module @IrToMhlo.2452 {
  func.func private @Unknown0(%arg0: memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<4x3x224x224xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x3x224x224xf32>) outs(%alloc : memref<4x3x224x224xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x3x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x3x7x7xf32>) outs(%alloc : memref<64x3x7x7xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<64x3x7x7xf16>
  }
  func.func private @Unknown3(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%alloc : memref<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%alloc : memref<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown5(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%alloc : memref<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown6(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%alloc : memref<64x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown7(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x64x1x1xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x1x1xf32>) outs(%alloc : memref<128x64x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<128x64x1x1xf16>
  }
  func.func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x64x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x3x3xf32>) outs(%alloc : memref<128x64x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<128x64x3x3xf16>
  }
  func.func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%alloc : memref<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown10(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%alloc : memref<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown11(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%alloc : memref<128x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown12(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x128x1x1xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x1x1xf32>) outs(%alloc : memref<256x128x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<256x128x1x1xf16>
  }
  func.func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x128x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x3x3xf32>) outs(%alloc : memref<256x128x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<256x128x3x3xf16>
  }
  func.func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%alloc : memref<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown15(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%alloc : memref<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown16(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%alloc : memref<256x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown17(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x256x1x1xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x1x1xf32>) outs(%alloc : memref<512x256x1x1xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<512x256x1x1xf16>
  }
  func.func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x256x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x3x3xf32>) outs(%alloc : memref<512x256x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<512x256x3x3xf16>
  }
  func.func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%alloc : memref<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown20(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%alloc : memref<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown21(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%alloc : memref<512x512x3x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown22(%arg0: memref<4x1000xf32>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -2.500000e-01 : f32
    %alloc = memref.alloc() : memref<4x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<4x1000xf32>) outs(%alloc : memref<4x1000xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.mulf %in, %cst : f32
      %1 = arith.truncf %0 : f32 to f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown23(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<1000x512xf16>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1000x512xf32>) outs(%alloc : memref<1000x512xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<1000x512xf16>
  }
  func.func private @Unknown24(%arg0: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    %alloc_0 = memref.alloc() : memref<4x64x112x112xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x112x112xf16>) outs(%alloc, %alloc_0 : memref<4x64x112x112xf16>, memref<4x64x112x112xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
  }
  func.func private @Unknown26(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x56x56xf16>) outs(%alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown28(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: i1):
      %0 = arith.addf %in, %in_1 : f16
      %1 = arith.maxnumf %0, %cst : f16
      %2 = arith.cmpf ogt, %1, %cst : f16
      linalg.yield %1, %2 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown30(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x56x56xf16>) outs(%alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown32(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: i1):
      %0 = arith.addf %in, %in_1 : f16
      %1 = arith.maxnumf %0, %cst : f16
      %2 = arith.cmpf ogt, %1, %cst : f16
      linalg.yield %1, %2 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown35(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x128x28x28xf16>) outs(%alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown37(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: i1):
      %0 = arith.addf %in, %in_1 : f16
      %1 = arith.maxnumf %0, %cst : f16
      %2 = arith.cmpf ogt, %1, %cst : f16
      linalg.yield %1, %2 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown39(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x128x28x28xf16>) outs(%alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown41(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: i1):
      %0 = arith.addf %in, %in_1 : f16
      %1 = arith.maxnumf %0, %cst : f16
      %2 = arith.cmpf ogt, %1, %cst : f16
      linalg.yield %1, %2 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown44(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x256x14x14xf16>) outs(%alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown46(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: i1):
      %0 = arith.addf %in, %in_1 : f16
      %1 = arith.maxnumf %0, %cst : f16
      %2 = arith.cmpf ogt, %1, %cst : f16
      linalg.yield %1, %2 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown48(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x256x14x14xf16>) outs(%alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown50(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: i1):
      %0 = arith.addf %in, %in_1 : f16
      %1 = arith.maxnumf %0, %cst : f16
      %2 = arith.cmpf ogt, %1, %cst : f16
      linalg.yield %1, %2 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown53(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x512x7x7xf16>) outs(%alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown55(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) outs(%alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: i1):
      %0 = arith.addf %in, %in_1 : f16
      %1 = arith.maxnumf %0, %cst : f16
      %2 = arith.cmpf ogt, %1, %cst : f16
      linalg.yield %1, %2 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown57(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x512x7x7xf16>) outs(%alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) {
    ^bb0(%in: f16, %out: f16, %out_1: i1):
      %0 = arith.maxnumf %in, %cst : f16
      %1 = arith.cmpf ogt, %0, %cst : f16
      linalg.yield %0, %1 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown59(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) outs(%alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: i1):
      %0 = arith.addf %in, %in_1 : f16
      %1 = arith.maxnumf %0, %cst : f16
      %2 = arith.cmpf ogt, %1, %cst : f16
      linalg.yield %1, %2 : f16, i1
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown60(%arg0: memref<4x512xf16>) -> memref<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %alloc = memref.alloc() : memref<4x512xf16>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<4x512xf16>) outs(%alloc : memref<4x512xf16>) {
    ^bb0(%in: f16, %out: f16):
      %0 = arith.mulf %in, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x512xf16>
  }
  func.func private @Unknown61(%arg0: memref<1000xf32>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<4x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<4x1000xf16>, memref<1000xf32>) outs(%alloc : memref<4x1000xf16>) {
    ^bb0(%in: f16, %in_0: f32, %out: f16):
      %0 = arith.truncf %in_0 : f32 to f16
      %1 = arith.addf %in, %0 : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown62(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<4x1000xf16>
    %alloc_0 = memref.alloc() : memref<4x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map3, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<4x1000xf16>, memref<4xf16>) outs(%alloc, %alloc_0 : memref<4x1000xf16>, memref<4x1000xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16, %out_2: f16):
      %0 = arith.subf %in, %in_1 : f16
      %1 = math.exp %0 : f16
      linalg.yield %0, %1 : f16, f16
    }
    return %alloc, %alloc_0 : memref<4x1000xf16>, memref<4x1000xf16>
  }
  func.func private @Unknown63(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>, %arg4: memref<4x1000xf32>) -> (memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>) attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<4x1000xf16>
    %alloc_0 = memref.alloc() : memref<4x1000xf32>
    %alloc_1 = memref.alloc() : memref<4x1000xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map3, #map3, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg1, %arg0, %arg2, %arg4 : memref<4x1000xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4xf16>, memref<4x1000xf32>) outs(%alloc, %alloc_0, %alloc_1 : memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>) {
    ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %in_5: f32, %out: f16, %out_6: f32, %out_7: f32):
      %0 = math.log %in_3 : f16
      %1 = arith.subf %in_2, %0 : f16
      %2 = math.exp %1 : f16
      %3 = arith.mulf %2, %in_4 : f16
      %4 = arith.subf %in, %3 : f16
      %5 = arith.extf %1 : f16 to f32
      %6 = arith.mulf %5, %in_5 : f32
      %7 = arith.extf %4 : f16 to f32
      linalg.yield %4, %6, %7 : f16, f32, f32
    }
    return %alloc, %alloc_0, %alloc_1 : memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>
  }
  func.func private @Unknown64(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.900000e+01 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg0 : memref<4x512x7x7xi1>, memref<4x512xf16>) outs(%alloc : memref<4x512x7x7xf16>) {
    ^bb0(%in: i1, %in_1: f16, %out: f16):
      %0 = arith.divf %in_1, %cst : f16
      %1 = arith.select %in, %0, %cst_0 : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown68(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) outs(%alloc : memref<4x512x7x7xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown72(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) outs(%alloc : memref<4x512x7x7xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.select %in, %0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown76(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) outs(%alloc : memref<4x512x7x7xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown83(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%alloc : memref<4x256x14x14xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.select %in, %0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown87(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) outs(%alloc : memref<4x256x14x14xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown91(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%alloc : memref<4x256x14x14xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.select %in, %0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown95(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) outs(%alloc : memref<4x256x14x14xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown102(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%alloc : memref<4x128x28x28xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.select %in, %0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown106(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) outs(%alloc : memref<4x128x28x28xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown110(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%alloc : memref<4x128x28x28xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.select %in, %0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown114(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) outs(%alloc : memref<4x128x28x28xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown121(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%alloc : memref<4x64x56x56xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.select %in, %0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown125(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) outs(%alloc : memref<4x64x56x56xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown129(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%alloc : memref<4x64x56x56xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %0 = arith.addf %in_0, %in_1 : f16
      %1 = arith.select %in, %0, %cst : f16
      linalg.yield %1 : f16
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown133(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) outs(%alloc : memref<4x64x56x56xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown137(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%alloc : memref<4x64x56x56xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.addf %in, %in_0 : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown138(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x112x112xi1>, memref<4x64x112x112xf16>) outs(%alloc : memref<4x64x112x112xf16>) {
    ^bb0(%in: i1, %in_0: f16, %out: f16):
      %0 = arith.select %in, %in_0, %cst : f16
      linalg.yield %0 : f16
    }
    return %alloc : memref<4x64x112x112xf16>
  }
  func.func private @Unknown141(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.000000e+00 : f32
    %alloc = memref.alloc() : memref<f32>
    linalg.generic {indexing_maps = [#map5, #map5], iterator_types = []} ins(%arg0 : memref<f32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.negf %in : f32
      %1 = arith.divf %0, %cst : f32
      linalg.yield %1 : f32
    }
    return %alloc : memref<f32>
  }
  func.func private @Unknown142(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x3x7x7xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x3x7x7xf16>) outs(%alloc : memref<64x3x7x7xf32>) attrs =  {xla_shape = "f32[64,3,7,7]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x3x7x7xf32>
  }
  func.func private @Unknown143(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%alloc : memref<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown144(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%alloc : memref<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown145(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%alloc : memref<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown146(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%alloc : memref<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown147(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x64x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x3x3xf16>) outs(%alloc : memref<128x64x3x3xf32>) attrs =  {xla_shape = "f32[128,64,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x64x3x3xf32>
  }
  func.func private @Unknown148(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%alloc : memref<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown149(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x64x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x1x1xf16>) outs(%alloc : memref<128x64x1x1xf32>) attrs =  {xla_shape = "f32[128,64,1,1]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x64x1x1xf32>
  }
  func.func private @Unknown150(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%alloc : memref<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown151(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%alloc : memref<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown152(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x128x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x3x3xf16>) outs(%alloc : memref<256x128x3x3xf32>) attrs =  {xla_shape = "f32[256,128,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x128x3x3xf32>
  }
  func.func private @Unknown153(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%alloc : memref<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown154(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x128x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x1x1xf16>) outs(%alloc : memref<256x128x1x1xf32>) attrs =  {xla_shape = "f32[256,128,1,1]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x128x1x1xf32>
  }
  func.func private @Unknown155(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%alloc : memref<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown156(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%alloc : memref<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown157(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x256x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x3x3xf16>) outs(%alloc : memref<512x256x3x3xf32>) attrs =  {xla_shape = "f32[512,256,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x256x3x3xf32>
  }
  func.func private @Unknown158(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%alloc : memref<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown159(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x256x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x1x1xf16>) outs(%alloc : memref<512x256x1x1xf32>) attrs =  {xla_shape = "f32[512,256,1,1]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x256x1x1xf32>
  }
  func.func private @Unknown160(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%alloc : memref<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown161(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%alloc : memref<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown163(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<1000x512xf32>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1000x512xf16>) outs(%alloc : memref<1000x512xf32>) attrs =  {xla_shape = "f32[1000,512]{0,1}"} {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<1000x512xf32>
  }
  func.func private @Unknown164(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<1000xf32>
    linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel"]} ins(%arg0 : memref<1000xf32>) outs(%alloc : memref<1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.truncf %in : f32 to f16
      %1 = arith.extf %0 : f16 to f32
      linalg.yield %1 : f32
    }
    return %alloc : memref<1000xf32>
  }
  func.func @main(%arg0: memref<4x3x224x224xf32>, %arg1: memref<4x1000xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64x64x3x3xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<64xf32>, %arg11: memref<64xf32>, %arg12: memref<64x64x3x3xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64xf32>, %arg16: memref<64xf32>, %arg17: memref<64x64x3x3xf32>, %arg18: memref<64xf32>, %arg19: memref<64xf32>, %arg20: memref<64xf32>, %arg21: memref<64xf32>, %arg22: memref<64x64x3x3xf32>, %arg23: memref<64xf32>, %arg24: memref<64xf32>, %arg25: memref<64xf32>, %arg26: memref<64xf32>, %arg27: memref<128x64x3x3xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128xf32>, %arg31: memref<128xf32>, %arg32: memref<128x128x3x3xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<128xf32>, %arg37: memref<128x64x1x1xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128x3x3xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<128xf32>, %arg47: memref<128x128x3x3xf32>, %arg48: memref<128xf32>, %arg49: memref<128xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<256x128x3x3xf32>, %arg53: memref<256xf32>, %arg54: memref<256xf32>, %arg55: memref<256xf32>, %arg56: memref<256xf32>, %arg57: memref<256x256x3x3xf32>, %arg58: memref<256xf32>, %arg59: memref<256xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256x128x1x1xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256x256x3x3xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<256xf32>, %arg71: memref<256xf32>, %arg72: memref<256x256x3x3xf32>, %arg73: memref<256xf32>, %arg74: memref<256xf32>, %arg75: memref<256xf32>, %arg76: memref<256xf32>, %arg77: memref<512x256x3x3xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<512xf32>, %arg81: memref<512xf32>, %arg82: memref<512x512x3x3xf32>, %arg83: memref<512xf32>, %arg84: memref<512xf32>, %arg85: memref<512xf32>, %arg86: memref<512xf32>, %arg87: memref<512x256x1x1xf32>, %arg88: memref<512xf32>, %arg89: memref<512xf32>, %arg90: memref<512xf32>, %arg91: memref<512xf32>, %arg92: memref<512x512x3x3xf32>, %arg93: memref<512xf32>, %arg94: memref<512xf32>, %arg95: memref<512xf32>, %arg96: memref<512xf32>, %arg97: memref<512x512x3x3xf32>, %arg98: memref<512xf32>, %arg99: memref<512xf32>, %arg100: memref<512xf32>, %arg101: memref<512xf32>, %arg102: memref<1000x512xf32>, %arg103: memref<1000xf32>) -> (memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg0) : (memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16>
    %1 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @ConvOp_f16f16_f16(%0, %1, %alloc) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<64x3x7x7xf16>, memref<4x64x112x112xf16>
    %alloc_0 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc, %arg3, %arg4, %alloc_0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf16>
    %2 = call @Unknown3(%arg7) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %3 = call @Unknown4(%arg12) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %4 = call @Unknown5(%arg17) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %5 = call @Unknown6(%arg22) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %6 = call @Unknown7(%arg37) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %7 = call @Unknown8(%arg27) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %8 = call @Unknown9(%arg32) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %9 = call @Unknown10(%arg42) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %10 = call @Unknown11(%arg47) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %11 = call @Unknown12(%arg62) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %12 = call @Unknown13(%arg52) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %13 = call @Unknown14(%arg57) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %14 = call @Unknown15(%arg67) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %15 = call @Unknown16(%arg72) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %16 = call @Unknown17(%arg87) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %17 = call @Unknown18(%arg77) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %18 = call @Unknown19(%arg82) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %19 = call @Unknown20(%arg92) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %20 = call @Unknown21(%arg97) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %21 = call @Unknown22(%arg1) : (memref<4x1000xf32>) -> memref<4x1000xf16>
    %22 = call @Unknown23(%arg102) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    %alloc_1 = memref.alloc() : memref<4xf16>
    byre.compute @ReduceSumOp_f16_f16(%21, %alloc_1) {dimensions = dense<1> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<4x1000xf16>, memref<4xf16>
    %23:2 = call @Unknown24(%alloc_0) : (memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>)
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @PoolMaxOp_f16_f16(%23#0, %alloc_2) {base_dilations = dense<1> : tensor<4xi64>, memory_effects = [1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%alloc_2, %2, %alloc_3) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_4 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_3, %arg8, %arg9, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %24:2 = call @Unknown26(%alloc_4) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%24#0, %3, %alloc_5) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_6 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_5, %arg13, %arg14, %alloc_6) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %25:2 = call @Unknown28(%alloc_6, %alloc_2) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_7 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%25#0, %4, %alloc_7) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_8 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_7, %arg18, %arg19, %alloc_8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %26:2 = call @Unknown30(%alloc_8) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_9 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%26#0, %5, %alloc_9) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_10 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_9, %arg23, %arg24, %alloc_10) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %27:2 = call @Unknown32(%alloc_10, %25#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_11 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%27#0, %6, %alloc_11) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x1x1xf16>, memref<4x128x28x28xf16>
    %alloc_12 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_11, %arg38, %arg39, %alloc_12) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %alloc_13 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%27#0, %7, %alloc_13) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_14 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_13, %arg28, %arg29, %alloc_14) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %28:2 = call @Unknown35(%alloc_14) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_15 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%28#0, %8, %alloc_15) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_16 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_15, %arg33, %arg34, %alloc_16) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %29:2 = call @Unknown37(%alloc_16, %alloc_12) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_17 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%29#0, %9, %alloc_17) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_18 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_17, %arg43, %arg44, %alloc_18) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %30:2 = call @Unknown39(%alloc_18) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_19 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%30#0, %10, %alloc_19) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_20 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_19, %arg48, %arg49, %alloc_20) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %31:2 = call @Unknown41(%alloc_20, %29#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_21 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%31#0, %11, %alloc_21) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x1x1xf16>, memref<4x256x14x14xf16>
    %alloc_22 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_21, %arg63, %arg64, %alloc_22) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %alloc_23 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%31#0, %12, %alloc_23) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_24 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_23, %arg53, %arg54, %alloc_24) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %32:2 = call @Unknown44(%alloc_24) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_25 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%32#0, %13, %alloc_25) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_26 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_25, %arg58, %arg59, %alloc_26) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %33:2 = call @Unknown46(%alloc_26, %alloc_22) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_27 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%33#0, %14, %alloc_27) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_28 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_27, %arg68, %arg69, %alloc_28) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %34:2 = call @Unknown48(%alloc_28) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_29 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%34#0, %15, %alloc_29) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_30 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_29, %arg73, %arg74, %alloc_30) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %35:2 = call @Unknown50(%alloc_30, %33#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_31 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%35#0, %16, %alloc_31) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x1x1xf16>, memref<4x512x7x7xf16>
    %alloc_32 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_31, %arg88, %arg89, %alloc_32) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %alloc_33 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%35#0, %17, %alloc_33) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_34 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_33, %arg78, %arg79, %alloc_34) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %36:2 = call @Unknown53(%alloc_34) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_35 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%36#0, %18, %alloc_35) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_36 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_35, %arg83, %arg84, %alloc_36) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %37:2 = call @Unknown55(%alloc_36, %alloc_32) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_37 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%37#0, %19, %alloc_37) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_38 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_37, %arg93, %arg94, %alloc_38) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %38:2 = call @Unknown57(%alloc_38) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_39 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%38#0, %20, %alloc_39) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_40 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_39, %arg98, %arg99, %alloc_40) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %39:2 = call @Unknown59(%alloc_40, %37#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_41 = memref.alloc() : memref<4x512xf16>
    byre.compute @ReduceSumOp_f16_f16(%39#0, %alloc_41) {dimensions = dense<[3, 2]> : tensor<2xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<4x512xf16>
    %40 = call @Unknown60(%alloc_41) : (memref<4x512xf16>) -> memref<4x512xf16>
    %alloc_42 = memref.alloc() : memref<4x1000xf16>
    byre.compute @MatmulOp_f16f16_f16(%40, %22, %alloc_42) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<4x512xf16>, memref<1000x512xf16>, memref<4x1000xf16>
    %41 = call @Unknown61(%arg103, %alloc_42) : (memref<1000xf32>, memref<4x1000xf16>) -> memref<4x1000xf16>
    %alloc_43 = memref.alloc() : memref<4xf16>
    byre.compute @ReduceMaxOp_f16_f16(%41, %alloc_43) {dimensions = dense<1> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<4x1000xf16>, memref<4xf16>
    %42:2 = call @Unknown62(%alloc_43, %41) : (memref<4xf16>, memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>)
    %alloc_44 = memref.alloc() : memref<4xf16>
    byre.compute @ReduceSumOp_f16_f16(%42#1, %alloc_44) {dimensions = dense<1> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<4x1000xf16>, memref<4xf16>
    %43:3 = call @Unknown63(%alloc_44, %42#0, %alloc_1, %21, %arg1) : (memref<4xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf16>, memref<4x1000xf32>) -> (memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>)
    %alloc_45 = memref.alloc() : memref<4x512xf16>
    byre.compute @MatmulOp_f16f16_f16(%43#0, %22, %alloc_45) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<4x1000xf16>, memref<1000x512xf16>, memref<4x512xf16>
    %44 = call @Unknown64(%alloc_45, %39#1) : (memref<4x512xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %alloc_46 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_47 = memref.alloc() : memref<512xf32>
    %alloc_48 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_39, %arg98, %44, %alloc_46, %alloc_47, %alloc_48) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_49 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_46, %20, %alloc_49) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_50 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%38#0, %alloc_46, %alloc_50) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %45 = call @Unknown68(%38#1, %alloc_49) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %alloc_51 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_52 = memref.alloc() : memref<512xf32>
    %alloc_53 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_37, %arg93, %45, %alloc_51, %alloc_52, %alloc_53) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_54 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_51, %19, %alloc_54) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_55 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%37#0, %alloc_51, %alloc_55) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %46 = call @Unknown72(%44, %alloc_54, %37#1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %alloc_56 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_57 = memref.alloc() : memref<512xf32>
    %alloc_58 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_35, %arg83, %46, %alloc_56, %alloc_57, %alloc_58) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_59 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_56, %18, %alloc_59) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_60 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%36#0, %alloc_56, %alloc_60) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %47 = call @Unknown76(%36#1, %alloc_59) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %alloc_61 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_62 = memref.alloc() : memref<512xf32>
    %alloc_63 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_33, %arg78, %47, %alloc_61, %alloc_62, %alloc_63) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_64 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_61, %17, %alloc_64) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_65 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%35#0, %alloc_61, %alloc_65) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x3x3xf16>
    %alloc_66 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_67 = memref.alloc() : memref<512xf32>
    %alloc_68 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_31, %arg88, %46, %alloc_66, %alloc_67, %alloc_68) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_69 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_66, %16, %alloc_69) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x1x1xf16>, memref<4x256x14x14xf16>
    %alloc_70 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%35#0, %alloc_66, %alloc_70) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x1x1xf16>
    %48 = call @Unknown83(%alloc_69, %alloc_64, %35#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %alloc_71 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_72 = memref.alloc() : memref<256xf32>
    %alloc_73 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_29, %arg73, %48, %alloc_71, %alloc_72, %alloc_73) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_74 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_71, %15, %alloc_74) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_75 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%34#0, %alloc_71, %alloc_75) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %49 = call @Unknown87(%34#1, %alloc_74) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %alloc_76 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_77 = memref.alloc() : memref<256xf32>
    %alloc_78 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_27, %arg68, %49, %alloc_76, %alloc_77, %alloc_78) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_79 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_76, %14, %alloc_79) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_80 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%33#0, %alloc_76, %alloc_80) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %50 = call @Unknown91(%48, %alloc_79, %33#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %alloc_81 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_82 = memref.alloc() : memref<256xf32>
    %alloc_83 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_25, %arg58, %50, %alloc_81, %alloc_82, %alloc_83) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_84 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_81, %13, %alloc_84) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_85 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%32#0, %alloc_81, %alloc_85) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %51 = call @Unknown95(%32#1, %alloc_84) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %alloc_86 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_87 = memref.alloc() : memref<256xf32>
    %alloc_88 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_23, %arg53, %51, %alloc_86, %alloc_87, %alloc_88) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_89 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_86, %12, %alloc_89) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_90 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%31#0, %alloc_86, %alloc_90) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x3x3xf16>
    %alloc_91 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_92 = memref.alloc() : memref<256xf32>
    %alloc_93 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_21, %arg63, %50, %alloc_91, %alloc_92, %alloc_93) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_94 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_91, %11, %alloc_94) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x1x1xf16>, memref<4x128x28x28xf16>
    %alloc_95 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%31#0, %alloc_91, %alloc_95) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x1x1xf16>
    %52 = call @Unknown102(%alloc_94, %alloc_89, %31#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %alloc_96 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_97 = memref.alloc() : memref<128xf32>
    %alloc_98 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_19, %arg48, %52, %alloc_96, %alloc_97, %alloc_98) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_99 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_96, %10, %alloc_99) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_100 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%30#0, %alloc_96, %alloc_100) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %53 = call @Unknown106(%30#1, %alloc_99) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %alloc_101 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_102 = memref.alloc() : memref<128xf32>
    %alloc_103 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_17, %arg43, %53, %alloc_101, %alloc_102, %alloc_103) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_104 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_101, %9, %alloc_104) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_105 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%29#0, %alloc_101, %alloc_105) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %54 = call @Unknown110(%52, %alloc_104, %29#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %alloc_106 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_107 = memref.alloc() : memref<128xf32>
    %alloc_108 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_15, %arg33, %54, %alloc_106, %alloc_107, %alloc_108) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_109 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_106, %8, %alloc_109) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_110 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%28#0, %alloc_106, %alloc_110) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %55 = call @Unknown114(%28#1, %alloc_109) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %alloc_111 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_112 = memref.alloc() : memref<128xf32>
    %alloc_113 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_13, %arg28, %55, %alloc_111, %alloc_112, %alloc_113) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_114 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_111, %7, %alloc_114) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_115 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%27#0, %alloc_111, %alloc_115) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x3x3xf16>
    %alloc_116 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_117 = memref.alloc() : memref<128xf32>
    %alloc_118 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_11, %arg38, %54, %alloc_116, %alloc_117, %alloc_118) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_119 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_116, %6, %alloc_119) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x1x1xf16>, memref<4x64x56x56xf16>
    %alloc_120 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%27#0, %alloc_116, %alloc_120) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x1x1xf16>
    %56 = call @Unknown121(%alloc_119, %alloc_114, %27#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %alloc_121 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_122 = memref.alloc() : memref<64xf32>
    %alloc_123 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_9, %arg23, %56, %alloc_121, %alloc_122, %alloc_123) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_124 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_121, %5, %alloc_124) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_125 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%26#0, %alloc_121, %alloc_125) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %57 = call @Unknown125(%26#1, %alloc_124) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_126 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_127 = memref.alloc() : memref<64xf32>
    %alloc_128 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_7, %arg18, %57, %alloc_126, %alloc_127, %alloc_128) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_129 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_126, %4, %alloc_129) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_130 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%25#0, %alloc_126, %alloc_130) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %58 = call @Unknown129(%56, %alloc_129, %25#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %alloc_131 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_132 = memref.alloc() : memref<64xf32>
    %alloc_133 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_5, %arg13, %58, %alloc_131, %alloc_132, %alloc_133) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_134 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_131, %3, %alloc_134) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_135 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%24#0, %alloc_131, %alloc_135) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %59 = call @Unknown133(%24#1, %alloc_134) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_136 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_137 = memref.alloc() : memref<64xf32>
    %alloc_138 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_3, %arg8, %59, %alloc_136, %alloc_137, %alloc_138) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_139 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_136, %2, %alloc_139) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_140 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%alloc_2, %alloc_136, %alloc_140) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %60 = call @Unknown137(%58, %alloc_139) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_141 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @PoolMaxGradOp_f16f16_f16(%23#0, %60, %alloc_141) {memory_effects = [1 : i32, 1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>, memref<4x64x112x112xf16>
    %61 = call @Unknown138(%23#1, %alloc_141) : (memref<4x64x112x112xi1>, memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16>
    %alloc_142 = memref.alloc() : memref<4x64x112x112xf16>
    %alloc_143 = memref.alloc() : memref<64xf32>
    %alloc_144 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc, %arg3, %61, %alloc_142, %alloc_143, %alloc_144) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x112x112xf16>, memref<64xf32>, memref<4x64x112x112xf16>, memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    %alloc_145 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%0, %alloc_142, %alloc_145) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<4x64x112x112xf16>, memref<64x3x7x7xf16>
    %alloc_146 = memref.alloc() : memref<f32>
    byre.compute @ReduceSumOp_f32_f32(%43#1, %alloc_146) {dimensions = dense<[0, 1]> : tensor<2xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<4x1000xf32>, memref<f32>
    %62 = call @Unknown141(%alloc_146) : (memref<f32>) -> memref<f32>
    %63 = call @Unknown142(%alloc_145) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %64 = call @Unknown143(%alloc_140) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %65 = call @Unknown144(%alloc_135) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %66 = call @Unknown145(%alloc_130) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %67 = call @Unknown146(%alloc_125) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %68 = call @Unknown147(%alloc_115) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %69 = call @Unknown148(%alloc_110) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %70 = call @Unknown149(%alloc_120) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %71 = call @Unknown150(%alloc_105) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %72 = call @Unknown151(%alloc_100) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %73 = call @Unknown152(%alloc_90) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %74 = call @Unknown153(%alloc_85) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %75 = call @Unknown154(%alloc_95) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %76 = call @Unknown155(%alloc_80) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %77 = call @Unknown156(%alloc_75) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %78 = call @Unknown157(%alloc_65) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %79 = call @Unknown158(%alloc_60) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %80 = call @Unknown159(%alloc_70) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %81 = call @Unknown160(%alloc_55) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %82 = call @Unknown161(%alloc_50) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %alloc_147 = memref.alloc() : memref<1000x512xf16>
    byre.compute @MatmulOp_f16f16_f16(%40, %43#0, %alloc_147) {lhs_contracting_dimension = 0 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_transpose, rhs_contracting_dimension = 0 : i64} : memref<4x512xf16>, memref<4x1000xf16>, memref<1000x512xf16>
    %83 = call @Unknown163(%alloc_147) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %alloc_148 = memref.alloc() : memref<1000xf32>
    byre.compute @ReduceSumOp_f32_f32(%43#2, %alloc_148) {dimensions = dense<0> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<4x1000xf32>, memref<1000xf32>
    %84 = call @Unknown164(%alloc_148) : (memref<1000xf32>) -> memref<1000xf32>
    return %62, %63, %alloc_143, %alloc_144, %64, %alloc_137, %alloc_138, %65, %alloc_132, %alloc_133, %66, %alloc_127, %alloc_128, %67, %alloc_122, %alloc_123, %68, %alloc_112, %alloc_113, %69, %alloc_107, %alloc_108, %70, %alloc_117, %alloc_118, %71, %alloc_102, %alloc_103, %72, %alloc_97, %alloc_98, %73, %alloc_87, %alloc_88, %74, %alloc_82, %alloc_83, %75, %alloc_92, %alloc_93, %76, %alloc_77, %alloc_78, %77, %alloc_72, %alloc_73, %78, %alloc_62, %alloc_63, %79, %alloc_57, %alloc_58, %80, %alloc_67, %alloc_68, %81, %alloc_52, %alloc_53, %82, %alloc_47, %alloc_48, %83, %84 : memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>
  }
}