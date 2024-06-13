// RUN: byteir-opt %s -affine-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0 mod 64, 49)>
#map2 = affine_map<(d0) -> (d0 mod 64 + 1, 49)>
#map3 = affine_map<(d0, d1) -> (d0 - d1)>
#map4 = affine_map<(d0) -> (d0 * 2)>
#map5 = affine_map<(d0) -> (d0 * 2 + 1)>
module {
  func.func private @Unknown0(%arg0: memref<1x3x224x224xf32>) -> memref<1x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c224 = arith.constant 224 : index
    %alloc = memref.alloc() : memref<1x3x224x224xf16>
    scf.for %arg1 = %c0 to %c3 step %c1 {
      scf.for %arg2 = %c0 to %c224 step %c1 {
        scf.for %arg3 = %c0 to %c224 step %c1 {
          %subview = memref.subview %arg0[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x3x224x224xf32> to memref<f32, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x3x224x224xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f32, %out: f16):
            %0 = arith.truncf %in : f32 to f16
            linalg.yield %0 : f16
          }
        }
      }
    }
    return %alloc : memref<1x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf16>
    scf.for %arg1 = %c0 to %c64 step %c1 {
      scf.for %arg2 = %c0 to %c3 step %c1 {
        scf.for %arg3 = %c0 to %c7 step %c1 {
          scf.for %arg4 = %c0 to %c7 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<64x3x7x7xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<64x3x7x7xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<64x3x7x7xf16>
  }
  func.func private @Unknown3(%arg0: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c112 = arith.constant 112 : index
    %alloc = memref.alloc() : memref<1x64x112x112xf16>
    scf.for %arg1 = %c0 to %c64 step %c1 {
      scf.for %arg2 = %c0 to %c112 step %c1 {
        scf.for %arg3 = %c0 to %c112 step %c1 {
          %subview = memref.subview %arg0[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x64x112x112xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x64x112x112xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %out: f16):
            %0 = arith.maximumf %in, %cst : f16
            linalg.yield %0 : f16
          }
        }
      }
    }
    return %alloc : memref<1x64x112x112xf16>
  }
  func.func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    scf.for %arg1 = %c0 to %c64 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<64x64x3x3xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<64x64x3x3xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown6(%arg0: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c56 = arith.constant 56 : index
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    scf.for %arg1 = %c0 to %c64 step %c1 {
      scf.for %arg2 = %c0 to %c56 step %c1 {
        scf.for %arg3 = %c0 to %c56 step %c1 {
          %subview = memref.subview %arg0[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %out: f16):
            %0 = arith.maximumf %in, %cst : f16
            linalg.yield %0 : f16
          }
        }
      }
    }
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @Unknown9(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c56 = arith.constant 56 : index
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    scf.for %arg2 = %c0 to %c64 step %c1 {
      scf.for %arg3 = %c0 to %c56 step %c1 {
        scf.for %arg4 = %c0 to %c56 step %c1 {
          %subview = memref.subview %arg0[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
          %subview_1 = memref.subview %arg1[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %in_2: f16, %out: f16):
            %0 = arith.addf %in, %in_2 : f16
            %1 = arith.maximumf %0, %cst : f16
            linalg.yield %1 : f16
          }
        }
      }
    }
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @Unknown16(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf16>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x64x1x1xf32> to memref<f32, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x64x1x1xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f32, %out: f16):
          %0 = arith.truncf %in : f32 to f16
          linalg.yield %0 : f16
        }
      }
    }
    return %alloc : memref<128x64x1x1xf16>
  }
  func.func private @Unknown18(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf16>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x64x3x3xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x64x3x3xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<128x64x3x3xf16>
  }
  func.func private @Unknown20(%arg0: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      scf.for %arg2 = %c0 to %c28 step %c1 {
        scf.for %arg3 = %c0 to %c28 step %c1 {
          %subview = memref.subview %arg0[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %out: f16):
            %0 = arith.maximumf %in, %cst : f16
            linalg.yield %0 : f16
          }
        }
      }
    }
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @Unknown21(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x128x3x3xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x128x3x3xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown23(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    scf.for %arg2 = %c0 to %c128 step %c1 {
      scf.for %arg3 = %c0 to %c28 step %c1 {
        scf.for %arg4 = %c0 to %c28 step %c1 {
          %subview = memref.subview %arg0[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
          %subview_1 = memref.subview %arg1[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %in_2: f16, %out: f16):
            %0 = arith.addf %in, %in_2 : f16
            %1 = arith.maximumf %0, %cst : f16
            linalg.yield %1 : f16
          }
        }
      }
    }
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @Unknown30(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf16>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x128x1x1xf32> to memref<f32, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x128x1x1xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f32, %out: f16):
          %0 = arith.truncf %in : f32 to f16
          linalg.yield %0 : f16
        }
      }
    }
    return %alloc : memref<256x128x1x1xf16>
  }
  func.func private @Unknown32(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf16>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x128x3x3xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x128x3x3xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<256x128x3x3xf16>
  }
  func.func private @Unknown34(%arg0: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c14 = arith.constant 14 : index
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      scf.for %arg2 = %c0 to %c14 step %c1 {
        scf.for %arg3 = %c0 to %c14 step %c1 {
          %subview = memref.subview %arg0[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %out: f16):
            %0 = arith.maximumf %in, %cst : f16
            linalg.yield %0 : f16
          }
        }
      }
    }
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @Unknown35(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      scf.for %arg2 = %c0 to %c256 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x256x3x3xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x256x3x3xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown37(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c14 = arith.constant 14 : index
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      scf.for %arg3 = %c0 to %c14 step %c1 {
        scf.for %arg4 = %c0 to %c14 step %c1 {
          %subview = memref.subview %arg0[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
          %subview_1 = memref.subview %arg1[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %in_2: f16, %out: f16):
            %0 = arith.addf %in, %in_2 : f16
            %1 = arith.maximumf %0, %cst : f16
            linalg.yield %1 : f16
          }
        }
      }
    }
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @Unknown44(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf16>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      scf.for %arg2 = %c0 to %c256 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x256x1x1xf32> to memref<f32, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x256x1x1xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f32, %out: f16):
          %0 = arith.truncf %in : f32 to f16
          linalg.yield %0 : f16
        }
      }
    }
    return %alloc : memref<512x256x1x1xf16>
  }
  func.func private @Unknown46(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf16>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      scf.for %arg2 = %c0 to %c256 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x256x3x3xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x256x3x3xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<512x256x3x3xf16>
  }
  func.func private @Unknown48(%arg0: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      scf.for %arg2 = %c0 to %c7 step %c1 {
        scf.for %arg3 = %c0 to %c7 step %c1 {
          %subview = memref.subview %arg0[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg1, %arg2, %arg3] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %out: f16):
            %0 = arith.maximumf %in, %cst : f16
            linalg.yield %0 : f16
          }
        }
      }
    }
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @Unknown49(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      scf.for %arg2 = %c0 to %c512 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x512x3x3xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x512x3x3xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown51(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    scf.for %arg2 = %c0 to %c512 step %c1 {
      scf.for %arg3 = %c0 to %c7 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          %subview = memref.subview %arg0[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
          %subview_1 = memref.subview %arg1[0, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
          ^bb0(%in: f16, %in_2: f16, %out: f16):
            %0 = arith.addf %in, %in_2 : f16
            %1 = arith.maximumf %0, %cst : f16
            linalg.yield %1 : f16
          }
        }
      }
    }
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @Unknown58(%arg0: memref<1x512x7x7xf16>) -> memref<1x512xf16> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2, 3]] : memref<1x512x7x7xf16> into memref<512x49xf16>
    %alloc = memref.alloc() : memref<512xf16>
    scf.forall (%arg1) in (512) {
      %subview = memref.subview %collapse_shape[%arg1, 0] [1, 49] [1, 1] : memref<512x49xf16> to memref<49xf16, strided<[1], offset: ?>>
      %expand_shape_0 = memref.expand_shape %subview [[0, 1]] : memref<49xf16, strided<[1], offset: ?>> into memref<1x49xf16, strided<[49, 1], offset: ?>>
      %alloca = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = affine.min #map1(%arg2)
        %1 = affine.min #map2(%arg2)
        %2 = affine.apply #map3(%1, %0)
        %subview_6 = memref.subview %expand_shape_0[0, %0] [1, %2] [1, 1] : memref<1x49xf16, strided<[49, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_7 = memref.expand_shape %subview_6 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %3 = arith.cmpi ugt, %2, %c0 : index
        %4 = scf.if %3 -> (f16) {
          %6 = memref.load %expand_shape_7[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %6 : f16
        } else {
          scf.yield %cst : f16
        }
        %5 = arith.addf %4, %cst : f16
        memref.store %5, %alloca[%arg2] : memref<64xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_1 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (32) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca[%0] : memref<64xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca[%3] : memref<64xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_1[%arg2] : memref<32xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_2 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (16) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_1[%0] : memref<32xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_1[%3] : memref<32xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_2[%arg2] : memref<16xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_3 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (8) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_2[%0] : memref<16xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_2[%3] : memref<16xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_3[%arg2] : memref<8xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (4) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_3[%0] : memref<8xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_3[%3] : memref<8xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_4[%arg2] : memref<4xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (2) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_4[%0] : memref<4xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_4[%3] : memref<4xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_5[%arg2] : memref<2xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg2) in (1) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_5[%0] : memref<2xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_5[%3] : memref<2xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloc[%arg1] : memref<512xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    %expand_shape = memref.expand_shape %alloc [[0, 1]] : memref<512xf16> into memref<1x512xf16>
    return %expand_shape : memref<1x512xf16>
  }
  func.func private @Unknown59(%arg0: memref<1x512xf16>) -> memref<1x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<1x512xf16>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      %subview = memref.subview %arg0[0, %arg1] [1, 1] [1, 1] : memref<1x512xf16> to memref<f16, strided<[], offset: ?>>
      %subview_0 = memref.subview %alloc[0, %arg1] [1, 1] [1, 1] : memref<1x512xf16> to memref<f16, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
      ^bb0(%in: f16, %out: f16):
        %0 = arith.mulf %in, %cst : f16
        linalg.yield %0 : f16
      }
    }
    return %alloc : memref<1x512xf16>
  }
  func.func private @Unknown60(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<1000x512xf16>
    scf.for %arg1 = %c0 to %c1000 step %c1 {
      scf.for %arg2 = %c0 to %c512 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2] [1, 1] [1, 1] : memref<1000x512xf32> to memref<f32, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2] [1, 1] [1, 1] : memref<1000x512xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f32, %out: f16):
          %0 = arith.truncf %in : f32 to f16
          linalg.yield %0 : f16
        }
      }
    }
    return %alloc : memref<1000x512xf16>
  }
  func.func private @Unknown61(%arg0: memref<1000xf32>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() : memref<1x1000xf16>
    scf.for %arg2 = %c0 to %c1000 step %c1 {
      %subview = memref.subview %arg0[%arg2] [1] [1] : memref<1000xf32> to memref<f32, strided<[], offset: ?>>
      %subview_0 = memref.subview %alloc[0, %arg2] [1, 1] [1, 1] : memref<1x1000xf16> to memref<f16, strided<[], offset: ?>>
      %subview_1 = memref.subview %arg1[0, %arg2] [1, 1] [1, 1] : memref<1x1000xf16> to memref<f16, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f32, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %in_2: f16, %out: f16):
        %0 = arith.truncf %in : f32 to f16
        %1 = arith.addf %in_2, %0 : f16
        linalg.yield %1 : f16
      }
    }
    return %alloc : memref<1x1000xf16>
  }
  func.func private @Unknown62(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<64xf32>
    scf.for %arg2 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg1[%arg2] [1] [1] : memref<64xf32> to memref<f32, strided<[], offset: ?>>
      %subview_1 = memref.subview %alloc[%arg2] [1] [1] : memref<64xf32> to memref<f32, strided<[], offset: ?>>
      %subview_2 = memref.subview %arg0[%arg2] [1] [1] : memref<64xf32> to memref<f32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_2 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) outs(%subview_1 : memref<f32, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %0 = arith.mulf %in, %cst_0 : f32
        %1 = arith.mulf %in_3, %cst : f32
        %2 = arith.addf %1, %0 : f32
        linalg.yield %2 : f32
      }
    }
    return %alloc : memref<64xf32>
  }
  func.func private @Unknown72(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<128xf32>
    scf.for %arg2 = %c0 to %c128 step %c1 {
      %subview = memref.subview %arg1[%arg2] [1] [1] : memref<128xf32> to memref<f32, strided<[], offset: ?>>
      %subview_1 = memref.subview %alloc[%arg2] [1] [1] : memref<128xf32> to memref<f32, strided<[], offset: ?>>
      %subview_2 = memref.subview %arg0[%arg2] [1] [1] : memref<128xf32> to memref<f32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_2 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) outs(%subview_1 : memref<f32, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %0 = arith.mulf %in, %cst_0 : f32
        %1 = arith.mulf %in_3, %cst : f32
        %2 = arith.addf %1, %0 : f32
        linalg.yield %2 : f32
      }
    }
    return %alloc : memref<128xf32>
  }
  func.func private @Unknown82(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<256xf32>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      %subview = memref.subview %arg1[%arg2] [1] [1] : memref<256xf32> to memref<f32, strided<[], offset: ?>>
      %subview_1 = memref.subview %alloc[%arg2] [1] [1] : memref<256xf32> to memref<f32, strided<[], offset: ?>>
      %subview_2 = memref.subview %arg0[%arg2] [1] [1] : memref<256xf32> to memref<f32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_2 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) outs(%subview_1 : memref<f32, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %0 = arith.mulf %in, %cst_0 : f32
        %1 = arith.mulf %in_3, %cst : f32
        %2 = arith.addf %1, %0 : f32
        linalg.yield %2 : f32
      }
    }
    return %alloc : memref<256xf32>
  }
  func.func private @Unknown92(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<512xf32>
    scf.for %arg2 = %c0 to %c512 step %c1 {
      %subview = memref.subview %arg1[%arg2] [1] [1] : memref<512xf32> to memref<f32, strided<[], offset: ?>>
      %subview_1 = memref.subview %alloc[%arg2] [1] [1] : memref<512xf32> to memref<f32, strided<[], offset: ?>>
      %subview_2 = memref.subview %arg0[%arg2] [1] [1] : memref<512xf32> to memref<f32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_2 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) outs(%subview_1 : memref<f32, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %0 = arith.mulf %in, %cst_0 : f32
        %1 = arith.mulf %in_3, %cst : f32
        %2 = arith.addf %1, %0 : f32
        linalg.yield %2 : f32
      }
    }
    return %alloc : memref<512xf32>
  }
  func.func @main(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<1000xf32>, %arg4: memref<1000x512xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64x64x3x3xf32>, %arg10: memref<64x64x3x3xf32>, %arg11: memref<64xf32>, %arg12: memref<64xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64x64x3x3xf32>, %arg16: memref<64x64x3x3xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<128xf32>, %arg20: memref<128xf32>, %arg21: memref<128x64x3x3xf32>, %arg22: memref<128x128x3x3xf32>, %arg23: memref<128x64x1x1xf32>, %arg24: memref<128xf32>, %arg25: memref<128xf32>, %arg26: memref<128xf32>, %arg27: memref<128xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128x128x3x3xf32>, %arg31: memref<128x128x3x3xf32>, %arg32: memref<256xf32>, %arg33: memref<256xf32>, %arg34: memref<256xf32>, %arg35: memref<256xf32>, %arg36: memref<256x128x3x3xf32>, %arg37: memref<256x256x3x3xf32>, %arg38: memref<256x128x1x1xf32>, %arg39: memref<256xf32>, %arg40: memref<256xf32>, %arg41: memref<256xf32>, %arg42: memref<256xf32>, %arg43: memref<256xf32>, %arg44: memref<256xf32>, %arg45: memref<256x256x3x3xf32>, %arg46: memref<256x256x3x3xf32>, %arg47: memref<512xf32>, %arg48: memref<512xf32>, %arg49: memref<512xf32>, %arg50: memref<512xf32>, %arg51: memref<512x256x3x3xf32>, %arg52: memref<512x512x3x3xf32>, %arg53: memref<512x256x1x1xf32>, %arg54: memref<512xf32>, %arg55: memref<512xf32>, %arg56: memref<512xf32>, %arg57: memref<512xf32>, %arg58: memref<512xf32>, %arg59: memref<512xf32>, %arg60: memref<512x512x3x3xf32>, %arg61: memref<512x512x3x3xf32>, %arg62: memref<i64>, %arg63: memref<64xf32>, %arg64: memref<64xf32>, %arg65: memref<i64>, %arg66: memref<64xf32>, %arg67: memref<64xf32>, %arg68: memref<i64>, %arg69: memref<64xf32>, %arg70: memref<64xf32>, %arg71: memref<i64>, %arg72: memref<64xf32>, %arg73: memref<64xf32>, %arg74: memref<i64>, %arg75: memref<64xf32>, %arg76: memref<64xf32>, %arg77: memref<i64>, %arg78: memref<128xf32>, %arg79: memref<128xf32>, %arg80: memref<i64>, %arg81: memref<128xf32>, %arg82: memref<128xf32>, %arg83: memref<i64>, %arg84: memref<128xf32>, %arg85: memref<128xf32>, %arg86: memref<i64>, %arg87: memref<128xf32>, %arg88: memref<128xf32>, %arg89: memref<i64>, %arg90: memref<128xf32>, %arg91: memref<128xf32>, %arg92: memref<i64>, %arg93: memref<256xf32>, %arg94: memref<256xf32>, %arg95: memref<i64>, %arg96: memref<256xf32>, %arg97: memref<256xf32>, %arg98: memref<i64>, %arg99: memref<256xf32>, %arg100: memref<256xf32>, %arg101: memref<i64>, %arg102: memref<256xf32>, %arg103: memref<256xf32>, %arg104: memref<i64>, %arg105: memref<256xf32>, %arg106: memref<256xf32>, %arg107: memref<i64>, %arg108: memref<512xf32>, %arg109: memref<512xf32>, %arg110: memref<i64>, %arg111: memref<512xf32>, %arg112: memref<512xf32>, %arg113: memref<i64>, %arg114: memref<512xf32>, %arg115: memref<512xf32>, %arg116: memref<i64>, %arg117: memref<512xf32>, %arg118: memref<512xf32>, %arg119: memref<i64>, %arg120: memref<512xf32>, %arg121: memref<512xf32>, %arg122: memref<1x3x224x224xf32>) -> (memref<1x1000xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64x3x7x7xf16>, memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512xf16>, memref<512x1000xf16>) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg122) : (memref<1x3x224x224xf32>) -> memref<1x3x224x224xf16>
    %1 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    %alloc = memref.alloc() : memref<1x64x112x112xf16>
    byre.compute @ConvOp_f16f16_f16(%0, %1, %alloc) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x3x224x224xf16>, memref<64x3x7x7xf16>, memref<1x64x112x112xf16>
    %alloc_0 = memref.alloc() : memref<1x64x112x112xf16>
    %alloc_1 = memref.alloc() : memref<64xf32>
    %alloc_2 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc, %arg1, %arg0, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    %2 = call @Unknown3(%alloc_0) : (memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    %alloc_3 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @PoolMaxOp_f16_f16(%2, %alloc_3) {base_dilations = dense<1> : tensor<4xi64>, memory_effects = [1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<1x64x112x112xf16>, memref<1x64x56x56xf16>
    %3 = call @Unknown4(%arg9) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %alloc_4 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%alloc_3, %3, %alloc_4) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16>
    %alloc_6 = memref.alloc() : memref<64xf32>
    %alloc_7 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_4, %arg6, %arg5, %alloc_5, %alloc_6, %alloc_7) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %4 = call @Unknown6(%alloc_5) : (memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %5 = call @Unknown4(%arg10) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %alloc_8 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%4, %5, %alloc_8) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_9 = memref.alloc() : memref<1x64x56x56xf16>
    %alloc_10 = memref.alloc() : memref<64xf32>
    %alloc_11 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_8, %arg8, %arg7, %alloc_9, %alloc_10, %alloc_11) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %6 = call @Unknown9(%alloc_9, %alloc_3) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %7 = call @Unknown4(%arg15) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %alloc_12 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%6, %7, %alloc_12) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_13 = memref.alloc() : memref<1x64x56x56xf16>
    %alloc_14 = memref.alloc() : memref<64xf32>
    %alloc_15 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_12, %arg12, %arg11, %alloc_13, %alloc_14, %alloc_15) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %8 = call @Unknown6(%alloc_13) : (memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %9 = call @Unknown4(%arg16) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %alloc_16 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%8, %9, %alloc_16) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %alloc_17 = memref.alloc() : memref<1x64x56x56xf16>
    %alloc_18 = memref.alloc() : memref<64xf32>
    %alloc_19 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_16, %arg14, %arg13, %alloc_17, %alloc_18, %alloc_19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %10 = call @Unknown9(%alloc_17, %6) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %11 = call @Unknown16(%arg23) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %alloc_20 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%10, %11, %alloc_20) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>
    %alloc_21 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_22 = memref.alloc() : memref<128xf32>
    %alloc_23 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_20, %arg25, %arg24, %alloc_21, %alloc_22, %alloc_23) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %12 = call @Unknown18(%arg21) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %alloc_24 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%10, %12, %alloc_24) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>
    %alloc_25 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_26 = memref.alloc() : memref<128xf32>
    %alloc_27 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_24, %arg18, %arg17, %alloc_25, %alloc_26, %alloc_27) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %13 = call @Unknown20(%alloc_25) : (memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %14 = call @Unknown21(%arg22) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %alloc_28 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%13, %14, %alloc_28) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %alloc_29 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_30 = memref.alloc() : memref<128xf32>
    %alloc_31 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_28, %arg20, %arg19, %alloc_29, %alloc_30, %alloc_31) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %15 = call @Unknown23(%alloc_29, %alloc_21) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %16 = call @Unknown21(%arg30) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %alloc_32 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%15, %16, %alloc_32) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %alloc_33 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_34 = memref.alloc() : memref<128xf32>
    %alloc_35 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_32, %arg27, %arg26, %alloc_33, %alloc_34, %alloc_35) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %17 = call @Unknown20(%alloc_33) : (memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %18 = call @Unknown21(%arg31) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %alloc_36 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%17, %18, %alloc_36) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %alloc_37 = memref.alloc() : memref<1x128x28x28xf16>
    %alloc_38 = memref.alloc() : memref<128xf32>
    %alloc_39 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_36, %arg29, %arg28, %alloc_37, %alloc_38, %alloc_39) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %19 = call @Unknown23(%alloc_37, %15) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %20 = call @Unknown30(%arg38) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %alloc_40 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%19, %20, %alloc_40) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>
    %alloc_41 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_42 = memref.alloc() : memref<256xf32>
    %alloc_43 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_40, %arg40, %arg39, %alloc_41, %alloc_42, %alloc_43) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %21 = call @Unknown32(%arg36) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %alloc_44 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%19, %21, %alloc_44) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>
    %alloc_45 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_46 = memref.alloc() : memref<256xf32>
    %alloc_47 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_44, %arg33, %arg32, %alloc_45, %alloc_46, %alloc_47) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %22 = call @Unknown34(%alloc_45) : (memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %23 = call @Unknown35(%arg37) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %alloc_48 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%22, %23, %alloc_48) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %alloc_49 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_50 = memref.alloc() : memref<256xf32>
    %alloc_51 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_48, %arg35, %arg34, %alloc_49, %alloc_50, %alloc_51) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %24 = call @Unknown37(%alloc_49, %alloc_41) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %25 = call @Unknown35(%arg45) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %alloc_52 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%24, %25, %alloc_52) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %alloc_53 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_54 = memref.alloc() : memref<256xf32>
    %alloc_55 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_52, %arg42, %arg41, %alloc_53, %alloc_54, %alloc_55) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %26 = call @Unknown34(%alloc_53) : (memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %27 = call @Unknown35(%arg46) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %alloc_56 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%26, %27, %alloc_56) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %alloc_57 = memref.alloc() : memref<1x256x14x14xf16>
    %alloc_58 = memref.alloc() : memref<256xf32>
    %alloc_59 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_56, %arg44, %arg43, %alloc_57, %alloc_58, %alloc_59) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %28 = call @Unknown37(%alloc_57, %24) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %29 = call @Unknown44(%arg53) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %alloc_60 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%28, %29, %alloc_60) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>
    %alloc_61 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_62 = memref.alloc() : memref<512xf32>
    %alloc_63 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_60, %arg55, %arg54, %alloc_61, %alloc_62, %alloc_63) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %30 = call @Unknown46(%arg51) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %alloc_64 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%28, %30, %alloc_64) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>
    %alloc_65 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_66 = memref.alloc() : memref<512xf32>
    %alloc_67 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_64, %arg48, %arg47, %alloc_65, %alloc_66, %alloc_67) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %31 = call @Unknown48(%alloc_65) : (memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %32 = call @Unknown49(%arg52) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %alloc_68 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%31, %32, %alloc_68) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %alloc_69 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_70 = memref.alloc() : memref<512xf32>
    %alloc_71 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_68, %arg50, %arg49, %alloc_69, %alloc_70, %alloc_71) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %33 = call @Unknown51(%alloc_69, %alloc_61) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %34 = call @Unknown49(%arg60) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %alloc_72 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%33, %34, %alloc_72) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %alloc_73 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_74 = memref.alloc() : memref<512xf32>
    %alloc_75 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_72, %arg57, %arg56, %alloc_73, %alloc_74, %alloc_75) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %35 = call @Unknown48(%alloc_73) : (memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %36 = call @Unknown49(%arg61) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %alloc_76 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%35, %36, %alloc_76) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %alloc_77 = memref.alloc() : memref<1x512x7x7xf16>
    %alloc_78 = memref.alloc() : memref<512xf32>
    %alloc_79 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_76, %arg59, %arg58, %alloc_77, %alloc_78, %alloc_79) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %37 = call @Unknown51(%alloc_77, %33) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %38 = call @Unknown58(%37) : (memref<1x512x7x7xf16>) -> memref<1x512xf16>
    %39 = call @Unknown59(%38) : (memref<1x512xf16>) -> memref<1x512xf16>
    %40 = call @Unknown60(%arg4) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    %alloc_80 = memref.alloc() : memref<512x1000xf16>
    byre.compute @TransposeOp_f16_f16(%40, %alloc_80) {memory_effects = [1 : i32, 2 : i32], minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : memref<1000x512xf16>, memref<512x1000xf16>
    %alloc_81 = memref.alloc() : memref<1x1000xf16>
    byre.compute @MatmulOp_f16f16_f16(%39, %40, %alloc_81) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<1x512xf16>, memref<1000x512xf16>, memref<1x1000xf16>
    %41 = call @Unknown61(%arg3, %alloc_81) : (memref<1000xf32>, memref<1x1000xf16>) -> memref<1x1000xf16>
    %42 = call @Unknown62(%alloc_1, %arg63) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %43 = call @Unknown62(%alloc_2, %arg64) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %44 = call @Unknown62(%alloc_6, %arg66) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %45 = call @Unknown62(%alloc_7, %arg67) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %46 = call @Unknown62(%alloc_10, %arg69) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %47 = call @Unknown62(%alloc_11, %arg70) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %48 = call @Unknown62(%alloc_14, %arg72) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %49 = call @Unknown62(%alloc_15, %arg73) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %50 = call @Unknown62(%alloc_18, %arg75) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %51 = call @Unknown62(%alloc_19, %arg76) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %52 = call @Unknown72(%alloc_26, %arg78) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %53 = call @Unknown72(%alloc_27, %arg79) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %54 = call @Unknown72(%alloc_30, %arg81) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %55 = call @Unknown72(%alloc_31, %arg82) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %56 = call @Unknown72(%alloc_22, %arg84) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %57 = call @Unknown72(%alloc_23, %arg85) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %58 = call @Unknown72(%alloc_34, %arg87) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %59 = call @Unknown72(%alloc_35, %arg88) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %60 = call @Unknown72(%alloc_38, %arg90) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %61 = call @Unknown72(%alloc_39, %arg91) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %62 = call @Unknown82(%alloc_46, %arg93) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %63 = call @Unknown82(%alloc_47, %arg94) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %64 = call @Unknown82(%alloc_50, %arg96) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %65 = call @Unknown82(%alloc_51, %arg97) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %66 = call @Unknown82(%alloc_42, %arg99) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %67 = call @Unknown82(%alloc_43, %arg100) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %68 = call @Unknown82(%alloc_54, %arg102) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %69 = call @Unknown82(%alloc_55, %arg103) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %70 = call @Unknown82(%alloc_58, %arg105) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %71 = call @Unknown82(%alloc_59, %arg106) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %72 = call @Unknown92(%alloc_66, %arg108) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %73 = call @Unknown92(%alloc_67, %arg109) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %74 = call @Unknown92(%alloc_70, %arg111) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %75 = call @Unknown92(%alloc_71, %arg112) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %76 = call @Unknown92(%alloc_62, %arg114) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %77 = call @Unknown92(%alloc_63, %arg115) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %78 = call @Unknown92(%alloc_74, %arg117) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %79 = call @Unknown92(%alloc_75, %arg118) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %80 = call @Unknown92(%alloc_78, %arg120) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %81 = call @Unknown92(%alloc_79, %arg121) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    return %41, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %1, %0, %alloc, %2, %alloc_3, %3, %alloc_4, %4, %5, %alloc_8, %6, %7, %alloc_12, %8, %9, %alloc_16, %10, %12, %alloc_24, %13, %14, %alloc_28, %11, %alloc_20, %15, %16, %alloc_32, %17, %18, %alloc_36, %19, %21, %alloc_44, %22, %23, %alloc_48, %20, %alloc_40, %24, %25, %alloc_52, %26, %27, %alloc_56, %28, %30, %alloc_64, %31, %32, %alloc_68, %29, %alloc_60, %33, %34, %alloc_72, %35, %36, %alloc_76, %37, %39, %alloc_80 : memref<1x1000xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64x3x7x7xf16>, memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512xf16>, memref<512x1000xf16>
  }
}