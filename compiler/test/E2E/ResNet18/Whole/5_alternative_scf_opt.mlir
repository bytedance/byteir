// RUN: byteir-opt %s -scf-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 512) * 1024, 1000)>
#map2 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 512) * 1024 + 2, 1000)>
#map3 = affine_map<(d0, d1) -> (d0 - d1)>
#map4 = affine_map<(d0) -> (d0 * 2)>
#map5 = affine_map<(d0) -> (d0 * 2 + 1)>
#map6 = affine_map<(d0) -> (d0 mod 64, 49)>
#map7 = affine_map<(d0) -> (d0 mod 64 + 1, 49)>
#map8 = affine_map<(d0) -> (d0 mod 128, 125)>
#map9 = affine_map<(d0) -> (d0 mod 128 + 1, 125)>
#map10 = affine_map<(d0)[s0] -> (d0 * 32 + s0)>
#map11 = affine_map<(d0) -> (d0 * -32 + 1000, 32)>
#map12 = affine_map<(d0) -> (d0 * 32)>
#map13 = affine_map<(d0, d1) -> (d1 * -32 + 1000, 32, d0)>
#map14 = affine_map<(d0, d1) -> (d1 * -32 + 1000, 32, d0 + 1)>
module @IrToMhlo.2452 {
  func.func private @Unknown0(%arg0: memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c224 = arith.constant 224 : index
    %alloc = memref.alloc() : memref<4x3x224x224xf16>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c3 step %c1 {
        scf.for %arg3 = %c0 to %c224 step %c1 {
          scf.for %arg4 = %c0 to %c224 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x3x224x224xf32> to memref<f32, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x3x224x224xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f16):
              %0 = arith.truncf %in : f32 to f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x3x224x224xf16>
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
  func.func private @Unknown3(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown7(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown12(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown17(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown22(%arg0: memref<4x1000xf32>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -2.500000e-01 : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c1000 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2] [1, 1] [1, 1] : memref<4x1000xf32> to memref<f32, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f32, %out: f16):
          %0 = arith.mulf %in, %cst : f32
          %1 = arith.truncf %0 : f32 to f16
          linalg.yield %1 : f16
        }
      }
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown23(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown24(%arg0: memref<1000xf32>) -> memref<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1000xf16>
    scf.for %arg1 = %c0 to %c1000 step %c1 {
      %subview = memref.subview %arg0[%arg1] [1] [1] : memref<1000xf32> to memref<f32, strided<[], offset: ?>>
      %subview_0 = memref.subview %alloc[%arg1] [1] [1] : memref<1000xf16> to memref<f16, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %out: f16):
        %0 = arith.truncf %in : f32 to f16
        linalg.yield %0 : f16
      }
    }
    return %alloc : memref<1000xf16>
  }
  func.func private @Unknown25(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4xf16>
    scf.forall (%arg1) in (4) {
      %subview = memref.subview %arg0[%arg1, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (512) {
        %0 = affine.min #map1(%arg2)
        %1 = affine.min #map2(%arg2)
        %2 = affine.apply #map3(%1, %0)
        %subview_8 = memref.subview %expand_shape[0, %0] [1, %2] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_9 = memref.expand_shape %subview_8 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %3 = arith.cmpi ugt, %2, %c0 : index
        %4 = scf.if %3 -> (f16) {
          %9 = memref.load %expand_shape_9[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %9 : f16
        } else {
          scf.yield %cst : f16
        }
        %5 = arith.addf %4, %cst : f16
        %6 = arith.cmpi ugt, %2, %c1 : index
        %7 = scf.if %6 -> (f16) {
          %9 = memref.load %expand_shape_9[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %9 : f16
        } else {
          scf.yield %cst : f16
        }
        %8 = arith.addf %5, %7 : f16
        memref.store %8, %alloca[%arg2] : memref<512xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_0 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (256) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca[%0] : memref<512xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca[%3] : memref<512xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_0[%arg2] : memref<256xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_1 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (128) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_0[%0] : memref<256xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_0[%3] : memref<256xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_1[%arg2] : memref<128xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_2 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_1[%0] : memref<128xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_1[%3] : memref<128xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_2[%arg2] : memref<64xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_3 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (32) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_2[%0] : memref<64xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_2[%3] : memref<64xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_3[%arg2] : memref<32xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (16) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_3[%0] : memref<32xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_3[%3] : memref<32xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_4[%arg2] : memref<16xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (8) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_4[%0] : memref<16xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_4[%3] : memref<16xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_5[%arg2] : memref<8xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_6 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (4) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_5[%0] : memref<8xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_5[%3] : memref<8xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_6[%arg2] : memref<4xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (2) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_6[%0] : memref<4xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_6[%3] : memref<4xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_7[%arg2] : memref<2xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg2) in (1) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_7[%0] : memref<2xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_7[%3] : memref<2xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloc[%arg1] : memref<4xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown26(%arg0: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c112 = arith.constant 112 : index
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    %alloc_0 = memref.alloc() : memref<4x64x112x112xi1>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        scf.for %arg3 = %c0 to %c112 step %c1 {
          scf.for %arg4 = %c0 to %c112 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x112x112xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x112x112xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x112x112xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f16, %out_3: i1):
              %0 = arith.maximumf %in, %cst : f16
              %1 = arith.cmpf ogt, %0, %cst : f16
              linalg.yield %0, %1 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
  }
  func.func private @Unknown28(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c56 = arith.constant 56 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        scf.for %arg3 = %c0 to %c56 step %c1 {
          scf.for %arg4 = %c0 to %c56 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f16, %out_3: i1):
              %0 = arith.maximumf %in, %cst : f16
              %1 = arith.cmpf ogt, %0, %cst : f16
              linalg.yield %0, %1 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown30(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c56 = arith.constant 56 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c56 step %c1 {
          scf.for %arg5 = %c0 to %c56 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_3 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_3 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_4: f16, %out: f16, %out_5: i1):
              %0 = arith.addf %in, %in_4 : f16
              %1 = arith.maximumf %0, %cst : f16
              %2 = arith.cmpf ogt, %1, %cst : f16
              linalg.yield %1, %2 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown37(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c1 {
        scf.for %arg3 = %c0 to %c28 step %c1 {
          scf.for %arg4 = %c0 to %c28 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f16, %out_3: i1):
              %0 = arith.maximumf %in, %cst : f16
              %1 = arith.cmpf ogt, %0, %cst : f16
              linalg.yield %0, %1 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown39(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c128 step %c1 {
        scf.for %arg4 = %c0 to %c28 step %c1 {
          scf.for %arg5 = %c0 to %c28 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            %subview_3 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_3 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_4: f16, %out: f16, %out_5: i1):
              %0 = arith.addf %in, %in_4 : f16
              %1 = arith.maximumf %0, %cst : f16
              %2 = arith.cmpf ogt, %1, %cst : f16
              linalg.yield %1, %2 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown46(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c14 = arith.constant 14 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c256 step %c1 {
        scf.for %arg3 = %c0 to %c14 step %c1 {
          scf.for %arg4 = %c0 to %c14 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f16, %out_3: i1):
              %0 = arith.maximumf %in, %cst : f16
              %1 = arith.cmpf ogt, %0, %cst : f16
              linalg.yield %0, %1 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown48(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c14 = arith.constant 14 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c256 step %c1 {
        scf.for %arg4 = %c0 to %c14 step %c1 {
          scf.for %arg5 = %c0 to %c14 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            %subview_3 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_3 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_4: f16, %out: f16, %out_5: i1):
              %0 = arith.addf %in, %in_4 : f16
              %1 = arith.maximumf %0, %cst : f16
              %2 = arith.cmpf ogt, %1, %cst : f16
              linalg.yield %1, %2 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown55(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c512 step %c1 {
        scf.for %arg3 = %c0 to %c7 step %c1 {
          scf.for %arg4 = %c0 to %c7 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f16, %out_3: i1):
              %0 = arith.maximumf %in, %cst : f16
              %1 = arith.cmpf ogt, %0, %cst : f16
              linalg.yield %0, %1 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown57(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c512 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc_0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xi1> to memref<i1, strided<[], offset: ?>>
            %subview_2 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_3 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_3 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_4: f16, %out: f16, %out_5: i1):
              %0 = arith.addf %in, %in_4 : f16
              %1 = arith.maximumf %0, %cst : f16
              %2 = arith.cmpf ogt, %1, %cst : f16
              linalg.yield %1, %2 : f16, i1
            }
          }
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown62(%arg0: memref<4x512x7x7xf16>) -> memref<4x512xf16> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2, 3]] : memref<4x512x7x7xf16> into memref<2048x49xf16>
    %alloc = memref.alloc() : memref<2048xf16>
    scf.forall (%arg1) in (2048) {
      %subview = memref.subview %collapse_shape[%arg1, 0] [1, 49] [1, 1] : memref<2048x49xf16> to memref<49xf16, strided<[1], offset: ?>>
      %expand_shape_0 = memref.expand_shape %subview [[0, 1]] : memref<49xf16, strided<[1], offset: ?>> into memref<1x49xf16, strided<[49, 1], offset: ?>>
      %alloca = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = affine.min #map6(%arg2)
        %1 = affine.min #map7(%arg2)
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
        memref.store %5, %alloc[%arg1] : memref<2048xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    %expand_shape = memref.expand_shape %alloc [[0, 1]] : memref<2048xf16> into memref<4x512xf16>
    return %expand_shape : memref<4x512xf16>
  }
  func.func private @Unknown63(%arg0: memref<4x512xf16>) -> memref<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512xf16>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c512 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2] [1, 1] [1, 1] : memref<4x512xf16> to memref<f16, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2] [1, 1] [1, 1] : memref<4x512xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f16, %out: f16):
          %0 = arith.mulf %in, %cst : f16
          linalg.yield %0 : f16
        }
      }
    }
    return %alloc : memref<4x512xf16>
  }
  func.func private @Unknown64(%arg0: memref<1000xf16>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c1000 step %c1 {
        %subview = memref.subview %arg0[%arg3] [1] [1] : memref<1000xf16> to memref<f16, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg2, %arg3] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        %subview_1 = memref.subview %arg1[%arg2, %arg3] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %0 = arith.addf %in_2, %in : f16
          linalg.yield %0 : f16
        }
      }
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown65(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4xf16>
    scf.forall (%arg1) in (4) {
      %subview = memref.subview %arg0[%arg1, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (512) {
        %0 = affine.min #map1(%arg2)
        %1 = affine.min #map2(%arg2)
        %2 = affine.apply #map3(%1, %0)
        %subview_8 = memref.subview %expand_shape[0, %0] [1, %2] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_9 = memref.expand_shape %subview_8 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %3 = arith.cmpi ugt, %2, %c0 : index
        %4 = scf.if %3 -> (f16) {
          %8 = memref.load %expand_shape_9[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %8 : f16
        } else {
          scf.yield %cst : f16
        }
        %5 = arith.cmpi ugt, %2, %c1 : index
        %6 = scf.if %5 -> (f16) {
          %8 = memref.load %expand_shape_9[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %8 : f16
        } else {
          scf.yield %cst : f16
        }
        %7 = arith.maximumf %4, %6 : f16
        memref.store %7, %alloca[%arg2] : memref<512xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_0 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (256) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca[%0] : memref<512xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca[%2] : memref<512xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_0[%arg2] : memref<256xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_1 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (128) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_0[%0] : memref<256xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca_0[%2] : memref<256xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_1[%arg2] : memref<128xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_2 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_1[%0] : memref<128xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca_1[%2] : memref<128xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_2[%arg2] : memref<64xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_3 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (32) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_2[%0] : memref<64xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca_2[%2] : memref<64xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_3[%arg2] : memref<32xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (16) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_3[%0] : memref<32xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca_3[%2] : memref<32xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_4[%arg2] : memref<16xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (8) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_4[%0] : memref<16xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca_4[%2] : memref<16xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_5[%arg2] : memref<8xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_6 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (4) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_5[%0] : memref<8xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca_5[%2] : memref<8xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_6[%arg2] : memref<4xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (2) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_6[%0] : memref<4xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca_6[%2] : memref<4xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_7[%arg2] : memref<2xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg2) in (1) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_7[%0] : memref<2xf16, #gpu.address_space<workgroup>>
        %2 = affine.apply #map5(%arg2)
        %3 = memref.load %alloca_7[%2] : memref<2xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloc[%arg1] : memref<4xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown66(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c1000 step %c1 {
        %subview = memref.subview %arg0[%arg2] [1] [1] : memref<4xf16> to memref<f16, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg2, %arg3] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        %subview_1 = memref.subview %arg1[%arg2, %arg3] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %0 = arith.subf %in_2, %in : f16
          linalg.yield %0 : f16
        }
      }
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown67(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4xf16>
    scf.forall (%arg1) in (4) {
      %subview = memref.subview %arg0[%arg1, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (512) {
        %0 = affine.min #map1(%arg2)
        %1 = affine.min #map2(%arg2)
        %2 = affine.apply #map3(%1, %0)
        %subview_8 = memref.subview %expand_shape[0, %0] [1, %2] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_9 = memref.expand_shape %subview_8 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %3 = arith.cmpi ugt, %2, %c0 : index
        %4 = scf.if %3 -> (f16) {
          %11 = memref.load %expand_shape_9[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %11 : f16
        } else {
          scf.yield %cst : f16
        }
        %5 = math.exp %4 : f16
        %6 = arith.addf %5, %cst : f16
        %7 = arith.cmpi ugt, %2, %c1 : index
        %8 = scf.if %7 -> (f16) {
          %11 = memref.load %expand_shape_9[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %11 : f16
        } else {
          scf.yield %cst : f16
        }
        %9 = math.exp %8 : f16
        %10 = arith.addf %6, %9 : f16
        memref.store %10, %alloca[%arg2] : memref<512xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_0 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (256) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca[%0] : memref<512xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca[%3] : memref<512xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_0[%arg2] : memref<256xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_1 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (128) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_0[%0] : memref<256xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_0[%3] : memref<256xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_1[%arg2] : memref<128xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_2 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_1[%0] : memref<128xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_1[%3] : memref<128xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_2[%arg2] : memref<64xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_3 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (32) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_2[%0] : memref<64xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_2[%3] : memref<64xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_3[%arg2] : memref<32xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (16) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_3[%0] : memref<32xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_3[%3] : memref<32xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_4[%arg2] : memref<16xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (8) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_4[%0] : memref<16xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_4[%3] : memref<16xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_5[%arg2] : memref<8xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_6 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (4) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_5[%0] : memref<8xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_5[%3] : memref<8xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_6[%arg2] : memref<4xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (2) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_6[%0] : memref<4xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_6[%3] : memref<4xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_7[%arg2] : memref<2xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg2) in (1) {
        %0 = affine.apply #map4(%arg2)
        %1 = memref.load %alloca_7[%0] : memref<2xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = affine.apply #map5(%arg2)
        %4 = memref.load %alloca_7[%3] : memref<2xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloc[%arg1] : memref<4xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown68(%arg0: memref<4xf16>) -> memref<4xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<4xf16>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      %subview = memref.subview %arg0[%arg1] [1] [1] : memref<4xf16> to memref<f16, strided<[], offset: ?>>
      %subview_0 = memref.subview %alloc[%arg1] [1] [1] : memref<4xf16> to memref<f16, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
      ^bb0(%in: f16, %out: f16):
        %0 = math.log %in : f16
        linalg.yield %0 : f16
      }
    }
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown69(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    %alloc_0 = memref.alloc() : memref<4x1000xf16>
    scf.for %arg4 = %c0 to %c4 step %c1 {
      scf.for %arg5 = %c0 to %c1000 step %c1 {
        %subview = memref.subview %arg2[%arg4] [1] [1] : memref<4xf16> to memref<f16, strided<[], offset: ?>>
        %subview_1 = memref.subview %alloc_0[%arg4, %arg5] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        %subview_2 = memref.subview %alloc[%arg4, %arg5] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        %subview_3 = memref.subview %arg0[%arg4] [1] [1] : memref<4xf16> to memref<f16, strided<[], offset: ?>>
        %subview_4 = memref.subview %arg1[%arg4, %arg5] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        %subview_5 = memref.subview %arg3[%arg4, %arg5] [1, 1] [1, 1] : memref<4x1000xf16> to memref<f16, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map, #map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_3, %subview_4, %subview_5 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_2, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) {
        ^bb0(%in: f16, %in_6: f16, %in_7: f16, %in_8: f16, %out: f16, %out_9: f16):
          %0 = arith.subf %in_7, %in_6 : f16
          %1 = math.exp %0 : f16
          %2 = arith.mulf %1, %in : f16
          %3 = arith.subf %in_8, %2 : f16
          linalg.yield %0, %3 : f16, f16
        }
      }
    }
    return %alloc, %alloc_0 : memref<4x1000xf16>, memref<4x1000xf16>
  }
  func.func private @Unknown70(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.900000e+01 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c512 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3] [1, 1] [1, 1] : memref<4x512xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_2 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xi1> to memref<i1, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_2 : memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) outs(%subview_1 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_3: i1, %out: f16):
              %0 = arith.divf %in, %cst : f16
              %1 = arith.select %in_3, %0, %cst_0 : f16
              linalg.yield %1 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown74(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c512 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xi1> to memref<i1, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<i1, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: i1, %in_2: f16, %out: f16):
              %0 = arith.select %in, %in_2, %cst : f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown78(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c512 step %c1 {
        scf.for %arg5 = %c0 to %c7 step %c1 {
          scf.for %arg6 = %c0 to %c7 step %c1 {
            %subview = memref.subview %arg0[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_2 = memref.subview %arg2[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x512x7x7xi1> to memref<i1, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_1, %subview_2 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_3: f16, %in_4: i1, %out: f16):
              %0 = arith.addf %in, %in_3 : f16
              %1 = arith.select %in_4, %0, %cst : f16
              linalg.yield %1 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown89(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c14 = arith.constant 14 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c256 step %c1 {
        scf.for %arg5 = %c0 to %c14 step %c1 {
          scf.for %arg6 = %c0 to %c14 step %c1 {
            %subview = memref.subview %arg0[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            %subview_2 = memref.subview %arg2[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xi1> to memref<i1, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_1, %subview_2 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_3: f16, %in_4: i1, %out: f16):
              %0 = arith.addf %in, %in_3 : f16
              %1 = arith.select %in_4, %0, %cst : f16
              linalg.yield %1 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown93(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c14 = arith.constant 14 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c256 step %c1 {
        scf.for %arg4 = %c0 to %c14 step %c1 {
          scf.for %arg5 = %c0 to %c14 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xi1> to memref<i1, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x256x14x14xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<i1, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: i1, %in_2: f16, %out: f16):
              %0 = arith.select %in, %in_2, %cst : f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown108(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        scf.for %arg5 = %c0 to %c28 step %c1 {
          scf.for %arg6 = %c0 to %c28 step %c1 {
            %subview = memref.subview %arg0[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            %subview_2 = memref.subview %arg2[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xi1> to memref<i1, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_1, %subview_2 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_3: f16, %in_4: i1, %out: f16):
              %0 = arith.addf %in, %in_3 : f16
              %1 = arith.select %in_4, %0, %cst : f16
              linalg.yield %1 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown112(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c128 step %c1 {
        scf.for %arg4 = %c0 to %c28 step %c1 {
          scf.for %arg5 = %c0 to %c28 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xi1> to memref<i1, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x128x28x28xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<i1, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: i1, %in_2: f16, %out: f16):
              %0 = arith.select %in, %in_2, %cst : f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown127(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c56 = arith.constant 56 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c64 step %c1 {
        scf.for %arg5 = %c0 to %c56 step %c1 {
          scf.for %arg6 = %c0 to %c56 step %c1 {
            %subview = memref.subview %arg0[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_2 = memref.subview %arg2[%arg3, %arg4, %arg5, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xi1> to memref<i1, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%subview, %subview_1, %subview_2 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>, memref<i1, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_3: f16, %in_4: i1, %out: f16):
              %0 = arith.addf %in, %in_3 : f16
              %1 = arith.select %in_4, %0, %cst : f16
              linalg.yield %1 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown131(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c56 = arith.constant 56 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c56 step %c1 {
          scf.for %arg5 = %c0 to %c56 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xi1> to memref<i1, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<i1, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: i1, %in_2: f16, %out: f16):
              %0 = arith.select %in, %in_2, %cst : f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown143(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c56 = arith.constant 56 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c56 step %c1 {
          scf.for %arg5 = %c0 to %c56 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x56x56xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f16, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %in_2: f16, %out: f16):
              %0 = arith.addf %in, %in_2 : f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown144(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c112 = arith.constant 112 : index
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c112 step %c1 {
          scf.for %arg5 = %c0 to %c112 step %c1 {
            %subview = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x112x112xi1> to memref<i1, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x112x112xf16> to memref<f16, strided<[], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x64x112x112xf16> to memref<f16, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<i1, strided<[], offset: ?>>, memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f16, strided<[], offset: ?>>) {
            ^bb0(%in: i1, %in_2: f16, %out: f16):
              %0 = arith.select %in, %in_2, %cst : f16
              linalg.yield %0 : f16
            }
          }
        }
      }
    }
    return %alloc : memref<4x64x112x112xf16>
  }
  func.func private @Unknown147(%arg0: memref<4x1000xf16>, %arg1: memref<4x1000xf32>) -> memref<f32> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<f32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<4x1000xf16> into memref<4000xf16>
    %collapse_shape_1 = memref.collapse_shape %arg1 [[0, 1]] : memref<4x1000xf32> into memref<4000xf32>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] : memref<4000xf16> into memref<32x125xf16>
    %expand_shape_2 = memref.expand_shape %collapse_shape_1 [[0, 1]] : memref<4000xf32> into memref<32x125xf32>
    %alloc_3 = memref.alloc() : memref<32xf32>
    scf.forall (%arg2) in (32) {
      %subview = memref.subview %expand_shape[%arg2, 0] [1, 125] [1, 1] : memref<32x125xf16> to memref<125xf16, strided<[1], offset: ?>>
      %expand_shape_4 = memref.expand_shape %subview [[0, 1]] : memref<125xf16, strided<[1], offset: ?>> into memref<1x125xf16, strided<[125, 1], offset: ?>>
      %subview_5 = memref.subview %expand_shape_2[%arg2, 0] [1, 125] [1, 1] : memref<32x125xf32> to memref<125xf32, strided<[1], offset: ?>>
      %expand_shape_6 = memref.expand_shape %subview_5 [[0, 1]] : memref<125xf32, strided<[1], offset: ?>> into memref<1x125xf32, strided<[125, 1], offset: ?>>
      %alloca = memref.alloca() : memref<128xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (128) {
        %0 = affine.min #map8(%arg3)
        %1 = affine.min #map9(%arg3)
        %2 = affine.apply #map3(%1, %0)
        %subview_13 = memref.subview %expand_shape_4[0, %0] [1, %2] [1, 1] : memref<1x125xf16, strided<[125, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_14 = memref.expand_shape %subview_13 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %subview_15 = memref.subview %expand_shape_6[0, %0] [1, %2] [1, 1] : memref<1x125xf32, strided<[125, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %expand_shape_16 = memref.expand_shape %subview_15 [[0, 1]] : memref<?xf32, strided<[1], offset: ?>> into memref<1x?xf32, strided<[?, 1], offset: ?>>
        %3 = arith.cmpi ugt, %2, %c0 : index
        %4:2 = scf.if %3 -> (f16, f32) {
          %8 = memref.load %expand_shape_14[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          %9 = memref.load %expand_shape_16[%c0, %c0] : memref<1x?xf32, strided<[?, 1], offset: ?>>
          scf.yield %8, %9 : f16, f32
        } else {
          scf.yield %cst_0, %cst : f16, f32
        }
        %5 = arith.extf %4#0 : f16 to f32
        %6 = arith.mulf %5, %4#1 : f32
        %7 = arith.addf %6, %cst : f32
        memref.store %7, %alloca[%arg3] : memref<128xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<64xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (64) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca[%0] : memref<128xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca[%3] : memref<128xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_7[%arg3] : memref<64xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_8 = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (32) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_7[%0] : memref<64xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_7[%3] : memref<64xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_8[%arg3] : memref<32xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_9 = memref.alloca() : memref<16xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (16) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_8[%0] : memref<32xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_8[%3] : memref<32xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_9[%arg3] : memref<16xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_10 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (8) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_9[%0] : memref<16xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_9[%3] : memref<16xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_10[%arg3] : memref<8xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_11 = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (4) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_10[%0] : memref<8xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_10[%3] : memref<8xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_11[%arg3] : memref<4xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_12 = memref.alloca() : memref<2xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (2) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_11[%0] : memref<4xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_11[%3] : memref<4xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_12[%arg3] : memref<2xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg3) in (1) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_12[%0] : memref<2xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_12[%3] : memref<2xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloc_3[%arg2] : memref<32xf32>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    scf.forall (%arg2) in (1) {
      %alloca = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (32) {
        %0 = affine.apply #map10(%arg2)[%arg3]
        %1 = memref.load %alloc_3[%0] : memref<32xf32>
        %2 = arith.addf %1, %cst : f32
        memref.store %2, %alloca[%arg3] : memref<32xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<16xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (16) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca[%0] : memref<32xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca[%3] : memref<32xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_4[%arg3] : memref<16xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (8) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_4[%0] : memref<16xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_4[%3] : memref<16xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_5[%arg3] : memref<8xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_6 = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (4) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_5[%0] : memref<8xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_5[%3] : memref<8xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_6[%arg3] : memref<4xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<2xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (2) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_6[%0] : memref<4xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_6[%3] : memref<4xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_7[%arg3] : memref<2xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg3) in (1) {
        %0 = affine.apply #map4(%arg3)
        %1 = memref.load %alloca_7[%0] : memref<2xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f32
        %3 = affine.apply #map5(%arg3)
        %4 = memref.load %alloca_7[%3] : memref<2xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloc[] : memref<f32>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<f32>
  }
  func.func private @Unknown148(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.000000e+00 : f32
    %alloc = memref.alloc() : memref<f32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : memref<f32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.negf %in : f32
      %1 = arith.divf %0, %cst : f32
      linalg.yield %1 : f32
    }
    return %alloc : memref<f32>
  }
  func.func private @Unknown149(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf32>
    scf.for %arg1 = %c0 to %c64 step %c1 {
      scf.for %arg2 = %c0 to %c3 step %c1 {
        scf.for %arg3 = %c0 to %c7 step %c1 {
          scf.for %arg4 = %c0 to %c7 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<64x3x7x7xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<64x3x7x7xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %0 = arith.extf %in : f16 to f32
              linalg.yield %0 : f32
            }
          }
        }
      }
    }
    return %alloc : memref<64x3x7x7xf32>
  }
  func.func private @Unknown150(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    scf.for %arg1 = %c0 to %c64 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<64x64x3x3xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<64x64x3x3xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %0 = arith.extf %in : f16 to f32
              linalg.yield %0 : f32
            }
          }
        }
      }
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown154(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x64x3x3xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x64x3x3xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %0 = arith.extf %in : f16 to f32
              linalg.yield %0 : f32
            }
          }
        }
      }
    }
    return %alloc : memref<128x64x3x3xf32>
  }
  func.func private @Unknown155(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x128x3x3xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x128x3x3xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %0 = arith.extf %in : f16 to f32
              linalg.yield %0 : f32
            }
          }
        }
      }
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown156(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x64x1x1xf16> to memref<f16, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<128x64x1x1xf32> to memref<f32, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
        ^bb0(%in: f16, %out: f32):
          %0 = arith.extf %in : f16 to f32
          linalg.yield %0 : f32
        }
      }
    }
    return %alloc : memref<128x64x1x1xf32>
  }
  func.func private @Unknown159(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x128x3x3xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x128x3x3xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %0 = arith.extf %in : f16 to f32
              linalg.yield %0 : f32
            }
          }
        }
      }
    }
    return %alloc : memref<256x128x3x3xf32>
  }
  func.func private @Unknown160(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      scf.for %arg2 = %c0 to %c256 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x256x3x3xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x256x3x3xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %0 = arith.extf %in : f16 to f32
              linalg.yield %0 : f32
            }
          }
        }
      }
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown161(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x128x1x1xf16> to memref<f16, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<256x128x1x1xf32> to memref<f32, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
        ^bb0(%in: f16, %out: f32):
          %0 = arith.extf %in : f16 to f32
          linalg.yield %0 : f32
        }
      }
    }
    return %alloc : memref<256x128x1x1xf32>
  }
  func.func private @Unknown164(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf32>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      scf.for %arg2 = %c0 to %c256 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x256x3x3xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x256x3x3xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %0 = arith.extf %in : f16 to f32
              linalg.yield %0 : f32
            }
          }
        }
      }
    }
    return %alloc : memref<512x256x3x3xf32>
  }
  func.func private @Unknown165(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      scf.for %arg2 = %c0 to %c512 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          scf.for %arg4 = %c0 to %c3 step %c1 {
            %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x512x3x3xf16> to memref<f16, strided<[], offset: ?>>
            %subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3, %arg4] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x512x3x3xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %0 = arith.extf %in : f16 to f32
              linalg.yield %0 : f32
            }
          }
        }
      }
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown166(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf32>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      scf.for %arg2 = %c0 to %c256 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x256x1x1xf16> to memref<f16, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<512x256x1x1xf32> to memref<f32, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
        ^bb0(%in: f16, %out: f32):
          %0 = arith.extf %in : f16 to f32
          linalg.yield %0 : f32
        }
      }
    }
    return %alloc : memref<512x256x1x1xf32>
  }
  func.func private @Unknown170(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<1000x512xf32>
    scf.for %arg1 = %c0 to %c1000 step %c1 {
      scf.for %arg2 = %c0 to %c512 step %c1 {
        %subview = memref.subview %arg0[%arg1, %arg2] [1, 1] [1, 1] : memref<1000x512xf16> to memref<f16, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg1, %arg2] [1, 1] [1, 1] : memref<1000x512xf32> to memref<f32, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f16, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
        ^bb0(%in: f16, %out: f32):
          %0 = arith.extf %in : f16 to f32
          linalg.yield %0 : f32
        }
      }
    }
    return %alloc : memref<1000x512xf32>
  }
  func.func private @Unknown171(%arg0: memref<4x1000xf16>) -> memref<1000xf32> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1000xf32>
    scf.forall (%arg1) in (32) {
      %0 = affine.min #map11(%arg1)
      %1 = affine.apply #map12(%arg1)
      %alloca = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      %alloca_1 = memref.alloca() : memref<2x32xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg2, %arg3) in (2, 32) {
        %2 = affine.min #map13(%arg3, %arg1)
        %3 = affine.min #map14(%arg3, %arg1)
        %4 = affine.apply #map3(%3, %2)
        %5 = arith.cmpi ugt, %4, %c0 : index
        %6 = scf.if %5 -> (f16) {
          %12 = affine.apply #map4(%arg2)
          %13 = affine.apply #map10(%arg1)[%2]
          %14 = memref.load %arg0[%12, %13] : memref<4x1000xf16>
          scf.yield %14 : f16
        } else {
          scf.yield %cst_0 : f16
        }
        %7 = arith.extf %6 : f16 to f32
        %8 = arith.addf %7, %cst : f32
        %9 = scf.if %5 -> (f16) {
          %12 = affine.apply #map5(%arg2)
          %13 = affine.apply #map10(%arg1)[%2]
          %14 = memref.load %arg0[%12, %13] : memref<4x1000xf16>
          scf.yield %14 : f16
        } else {
          scf.yield %cst_0 : f16
        }
        %10 = arith.extf %9 : f16 to f32
        %11 = arith.addf %8, %10 : f32
        memref.store %11, %alloca_1[%arg2, %arg3] : memref<2x32xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.forall (%arg2) in (32) {
        %2 = memref.load %alloca_1[%c0, %arg2] : memref<2x32xf32, #gpu.address_space<workgroup>>
        %3 = arith.addf %2, %cst : f32
        %4 = memref.load %alloca_1[%c1, %arg2] : memref<2x32xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %3 : f32
        memref.store %5, %alloca[%arg2] : memref<32xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %subview = memref.subview %alloca[0] [%0] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<?xf32, strided<[1]>, #gpu.address_space<workgroup>>
      %subview_2 = memref.subview %alloc[%1] [%0] [1] : memref<1000xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview, %subview_2 : memref<?xf32, strided<[1]>, #gpu.address_space<workgroup>> to memref<?xf32, strided<[1], offset: ?>>
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<1000xf32>
  }
  func.func private @Unknown172(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1000xf32>
    scf.for %arg1 = %c0 to %c1000 step %c1 {
      %subview = memref.subview %arg0[%arg1] [1] [1] : memref<1000xf32> to memref<f32, strided<[], offset: ?>>
      %subview_0 = memref.subview %alloc[%arg1] [1] [1] : memref<1000xf32> to memref<f32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%subview : memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.truncf %in : f32 to f16
        %1 = arith.extf %0 : f16 to f32
        linalg.yield %1 : f32
      }
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
    %3 = call @Unknown3(%arg12) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %4 = call @Unknown3(%arg17) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %5 = call @Unknown3(%arg22) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %6 = call @Unknown7(%arg37) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %7 = call @Unknown8(%arg27) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %8 = call @Unknown9(%arg32) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %9 = call @Unknown9(%arg42) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %10 = call @Unknown9(%arg47) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %11 = call @Unknown12(%arg62) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %12 = call @Unknown13(%arg52) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %13 = call @Unknown14(%arg57) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %14 = call @Unknown14(%arg67) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %15 = call @Unknown14(%arg72) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %16 = call @Unknown17(%arg87) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %17 = call @Unknown18(%arg77) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %18 = call @Unknown19(%arg82) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %19 = call @Unknown19(%arg92) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %20 = call @Unknown19(%arg97) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %21 = call @Unknown22(%arg1) : (memref<4x1000xf32>) -> memref<4x1000xf16>
    %22 = call @Unknown23(%arg102) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    %23 = call @Unknown24(%arg103) : (memref<1000xf32>) -> memref<1000xf16>
    %24 = call @Unknown25(%21) : (memref<4x1000xf16>) -> memref<4xf16>
    %25:2 = call @Unknown26(%alloc_0) : (memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>)
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @PoolMaxOp_f16_f16(%25#0, %alloc_1) {base_dilations = dense<1> : tensor<4xi64>, memory_effects = [1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%alloc_1, %2, %alloc_2) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_2, %arg8, %arg9, %alloc_3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %26:2 = call @Unknown28(%alloc_3) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_4 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%26#0, %3, %alloc_4) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_4, %arg13, %arg14, %alloc_5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %27:2 = call @Unknown30(%alloc_5, %alloc_1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_6 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%27#0, %4, %alloc_6) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_7 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_6, %arg18, %arg19, %alloc_7) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %28:2 = call @Unknown28(%alloc_7) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_8 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%28#0, %5, %alloc_8) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_9 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_8, %arg23, %arg24, %alloc_9) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %29:2 = call @Unknown30(%alloc_9, %27#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_10 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%29#0, %6, %alloc_10) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x1x1xf16>, memref<4x128x28x28xf16>
    %alloc_11 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_10, %arg38, %arg39, %alloc_11) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %alloc_12 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%29#0, %7, %alloc_12) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_13 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_12, %arg28, %arg29, %alloc_13) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %30:2 = call @Unknown37(%alloc_13) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_14 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%30#0, %8, %alloc_14) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_15 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_14, %arg33, %arg34, %alloc_15) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %31:2 = call @Unknown39(%alloc_15, %alloc_11) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_16 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%31#0, %9, %alloc_16) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_17 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_16, %arg43, %arg44, %alloc_17) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %32:2 = call @Unknown37(%alloc_17) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_18 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%32#0, %10, %alloc_18) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_19 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_18, %arg48, %arg49, %alloc_19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %33:2 = call @Unknown39(%alloc_19, %31#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_20 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%33#0, %11, %alloc_20) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x1x1xf16>, memref<4x256x14x14xf16>
    %alloc_21 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_20, %arg63, %arg64, %alloc_21) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %alloc_22 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%33#0, %12, %alloc_22) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_23 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_22, %arg53, %arg54, %alloc_23) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %34:2 = call @Unknown46(%alloc_23) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_24 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%34#0, %13, %alloc_24) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_25 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_24, %arg58, %arg59, %alloc_25) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %35:2 = call @Unknown48(%alloc_25, %alloc_21) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_26 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%35#0, %14, %alloc_26) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_27 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_26, %arg68, %arg69, %alloc_27) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %36:2 = call @Unknown46(%alloc_27) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_28 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%36#0, %15, %alloc_28) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_29 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_28, %arg73, %arg74, %alloc_29) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %37:2 = call @Unknown48(%alloc_29, %35#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_30 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%37#0, %16, %alloc_30) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x1x1xf16>, memref<4x512x7x7xf16>
    %alloc_31 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_30, %arg88, %arg89, %alloc_31) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %alloc_32 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%37#0, %17, %alloc_32) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_33 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_32, %arg78, %arg79, %alloc_33) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %38:2 = call @Unknown55(%alloc_33) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_34 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%38#0, %18, %alloc_34) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_35 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_34, %arg83, %arg84, %alloc_35) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %39:2 = call @Unknown57(%alloc_35, %alloc_31) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_36 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%39#0, %19, %alloc_36) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_37 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_36, %arg93, %arg94, %alloc_37) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %40:2 = call @Unknown55(%alloc_37) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_38 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%40#0, %20, %alloc_38) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_39 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_38, %arg98, %arg99, %alloc_39) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %41:2 = call @Unknown57(%alloc_39, %39#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %42 = call @Unknown62(%41#0) : (memref<4x512x7x7xf16>) -> memref<4x512xf16>
    %43 = call @Unknown63(%42) : (memref<4x512xf16>) -> memref<4x512xf16>
    %alloc_40 = memref.alloc() : memref<4x1000xf16>
    byre.compute @MatmulOp_f16f16_f16(%43, %22, %alloc_40) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<4x512xf16>, memref<1000x512xf16>, memref<4x1000xf16>
    %44 = call @Unknown64(%23, %alloc_40) : (memref<1000xf16>, memref<4x1000xf16>) -> memref<4x1000xf16>
    %45 = call @Unknown65(%44) : (memref<4x1000xf16>) -> memref<4xf16>
    %46 = call @Unknown66(%45, %44) : (memref<4xf16>, memref<4x1000xf16>) -> memref<4x1000xf16>
    %47 = call @Unknown67(%46) : (memref<4x1000xf16>) -> memref<4xf16>
    %48 = call @Unknown68(%47) : (memref<4xf16>) -> memref<4xf16>
    %49:2 = call @Unknown69(%48, %46, %24, %21) : (memref<4xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>)
    %alloc_41 = memref.alloc() : memref<4x512xf16>
    byre.compute @MatmulOp_f16f16_f16(%49#1, %22, %alloc_41) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<4x1000xf16>, memref<1000x512xf16>, memref<4x512xf16>
    %50 = call @Unknown70(%alloc_41, %41#1) : (memref<4x512xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %alloc_42 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_43 = memref.alloc() : memref<512xf32>
    %alloc_44 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_38, %arg98, %50, %alloc_42, %alloc_43, %alloc_44) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_45 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_42, %20, %alloc_45) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_46 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%40#0, %alloc_42, %alloc_46) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %51 = call @Unknown74(%40#1, %alloc_45) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %alloc_47 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_48 = memref.alloc() : memref<512xf32>
    %alloc_49 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_36, %arg93, %51, %alloc_47, %alloc_48, %alloc_49) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_50 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_47, %19, %alloc_50) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_51 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%39#0, %alloc_47, %alloc_51) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %52 = call @Unknown78(%50, %alloc_50, %39#1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %alloc_52 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_53 = memref.alloc() : memref<512xf32>
    %alloc_54 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_34, %arg83, %52, %alloc_52, %alloc_53, %alloc_54) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_55 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_52, %18, %alloc_55) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_56 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%38#0, %alloc_52, %alloc_56) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %53 = call @Unknown74(%38#1, %alloc_55) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %alloc_57 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_58 = memref.alloc() : memref<512xf32>
    %alloc_59 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_32, %arg78, %53, %alloc_57, %alloc_58, %alloc_59) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_60 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_57, %17, %alloc_60) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_61 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%37#0, %alloc_57, %alloc_61) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x3x3xf16>
    %alloc_62 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_63 = memref.alloc() : memref<512xf32>
    %alloc_64 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_30, %arg88, %52, %alloc_62, %alloc_63, %alloc_64) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_65 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_62, %16, %alloc_65) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x1x1xf16>, memref<4x256x14x14xf16>
    %alloc_66 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%37#0, %alloc_62, %alloc_66) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x1x1xf16>
    %54 = call @Unknown89(%alloc_65, %alloc_60, %37#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %alloc_67 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_68 = memref.alloc() : memref<256xf32>
    %alloc_69 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_28, %arg73, %54, %alloc_67, %alloc_68, %alloc_69) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_70 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_67, %15, %alloc_70) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_71 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%36#0, %alloc_67, %alloc_71) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %55 = call @Unknown93(%36#1, %alloc_70) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %alloc_72 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_73 = memref.alloc() : memref<256xf32>
    %alloc_74 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_26, %arg68, %55, %alloc_72, %alloc_73, %alloc_74) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_75 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_72, %14, %alloc_75) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_76 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%35#0, %alloc_72, %alloc_76) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %56 = call @Unknown89(%54, %alloc_75, %35#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %alloc_77 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_78 = memref.alloc() : memref<256xf32>
    %alloc_79 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_24, %arg58, %56, %alloc_77, %alloc_78, %alloc_79) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_80 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_77, %13, %alloc_80) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_81 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%34#0, %alloc_77, %alloc_81) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %57 = call @Unknown93(%34#1, %alloc_80) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %alloc_82 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_83 = memref.alloc() : memref<256xf32>
    %alloc_84 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_22, %arg53, %57, %alloc_82, %alloc_83, %alloc_84) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_85 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_82, %12, %alloc_85) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_86 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%33#0, %alloc_82, %alloc_86) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x3x3xf16>
    %alloc_87 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_88 = memref.alloc() : memref<256xf32>
    %alloc_89 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_20, %arg63, %56, %alloc_87, %alloc_88, %alloc_89) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_90 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_87, %11, %alloc_90) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x1x1xf16>, memref<4x128x28x28xf16>
    %alloc_91 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%33#0, %alloc_87, %alloc_91) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x1x1xf16>
    %58 = call @Unknown108(%alloc_90, %alloc_85, %33#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %alloc_92 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_93 = memref.alloc() : memref<128xf32>
    %alloc_94 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_18, %arg48, %58, %alloc_92, %alloc_93, %alloc_94) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_95 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_92, %10, %alloc_95) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_96 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%32#0, %alloc_92, %alloc_96) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %59 = call @Unknown112(%32#1, %alloc_95) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %alloc_97 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_98 = memref.alloc() : memref<128xf32>
    %alloc_99 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_16, %arg43, %59, %alloc_97, %alloc_98, %alloc_99) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_100 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_97, %9, %alloc_100) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_101 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%31#0, %alloc_97, %alloc_101) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %60 = call @Unknown108(%58, %alloc_100, %31#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %alloc_102 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_103 = memref.alloc() : memref<128xf32>
    %alloc_104 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_14, %arg33, %60, %alloc_102, %alloc_103, %alloc_104) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_105 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_102, %8, %alloc_105) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_106 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%30#0, %alloc_102, %alloc_106) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %61 = call @Unknown112(%30#1, %alloc_105) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %alloc_107 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_108 = memref.alloc() : memref<128xf32>
    %alloc_109 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_12, %arg28, %61, %alloc_107, %alloc_108, %alloc_109) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_110 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_107, %7, %alloc_110) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_111 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%29#0, %alloc_107, %alloc_111) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x3x3xf16>
    %alloc_112 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_113 = memref.alloc() : memref<128xf32>
    %alloc_114 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_10, %arg38, %60, %alloc_112, %alloc_113, %alloc_114) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_115 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_112, %6, %alloc_115) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x1x1xf16>, memref<4x64x56x56xf16>
    %alloc_116 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%29#0, %alloc_112, %alloc_116) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x1x1xf16>
    %62 = call @Unknown127(%alloc_115, %alloc_110, %29#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %alloc_117 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_118 = memref.alloc() : memref<64xf32>
    %alloc_119 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_8, %arg23, %62, %alloc_117, %alloc_118, %alloc_119) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_120 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_117, %5, %alloc_120) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_121 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%28#0, %alloc_117, %alloc_121) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %63 = call @Unknown131(%28#1, %alloc_120) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_122 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_123 = memref.alloc() : memref<64xf32>
    %alloc_124 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_6, %arg18, %63, %alloc_122, %alloc_123, %alloc_124) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_125 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_122, %4, %alloc_125) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_126 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%27#0, %alloc_122, %alloc_126) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %64 = call @Unknown127(%62, %alloc_125, %27#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %alloc_127 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_128 = memref.alloc() : memref<64xf32>
    %alloc_129 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_4, %arg13, %64, %alloc_127, %alloc_128, %alloc_129) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_130 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_127, %3, %alloc_130) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_131 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%26#0, %alloc_127, %alloc_131) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %65 = call @Unknown131(%26#1, %alloc_130) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_132 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_133 = memref.alloc() : memref<64xf32>
    %alloc_134 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_2, %arg8, %65, %alloc_132, %alloc_133, %alloc_134) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_135 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_132, %2, %alloc_135) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_136 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%alloc_1, %alloc_132, %alloc_136) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %66 = call @Unknown143(%64, %alloc_135) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_137 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @PoolMaxGradOp_f16f16_f16(%25#0, %66, %alloc_137) {memory_effects = [1 : i32, 1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>, memref<4x64x112x112xf16>
    %67 = call @Unknown144(%25#1, %alloc_137) : (memref<4x64x112x112xi1>, memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16>
    %alloc_138 = memref.alloc() : memref<4x64x112x112xf16>
    %alloc_139 = memref.alloc() : memref<64xf32>
    %alloc_140 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc, %arg3, %67, %alloc_138, %alloc_139, %alloc_140) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x112x112xf16>, memref<64xf32>, memref<4x64x112x112xf16>, memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    %alloc_141 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%0, %alloc_138, %alloc_141) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<4x64x112x112xf16>, memref<64x3x7x7xf16>
    %68 = call @Unknown147(%49#0, %arg1) : (memref<4x1000xf16>, memref<4x1000xf32>) -> memref<f32>
    %69 = call @Unknown148(%68) : (memref<f32>) -> memref<f32>
    %70 = call @Unknown149(%alloc_141) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %71 = call @Unknown150(%alloc_136) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %72 = call @Unknown150(%alloc_131) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %73 = call @Unknown150(%alloc_126) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %74 = call @Unknown150(%alloc_121) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %75 = call @Unknown154(%alloc_111) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %76 = call @Unknown155(%alloc_106) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %77 = call @Unknown156(%alloc_116) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %78 = call @Unknown155(%alloc_101) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %79 = call @Unknown155(%alloc_96) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %80 = call @Unknown159(%alloc_86) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %81 = call @Unknown160(%alloc_81) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %82 = call @Unknown161(%alloc_91) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %83 = call @Unknown160(%alloc_76) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %84 = call @Unknown160(%alloc_71) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %85 = call @Unknown164(%alloc_61) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %86 = call @Unknown165(%alloc_56) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %87 = call @Unknown166(%alloc_66) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %88 = call @Unknown165(%alloc_51) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %89 = call @Unknown165(%alloc_46) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %alloc_142 = memref.alloc() : memref<1000x512xf16>
    byre.compute @MatmulOp_f16f16_f16(%43, %49#1, %alloc_142) {lhs_contracting_dimension = 0 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_transpose, rhs_contracting_dimension = 0 : i64} : memref<4x512xf16>, memref<4x1000xf16>, memref<1000x512xf16>
    %90 = call @Unknown170(%alloc_142) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %91 = call @Unknown171(%49#1) : (memref<4x1000xf16>) -> memref<1000xf32>
    %92 = call @Unknown172(%91) : (memref<1000xf32>) -> memref<1000xf32>
    return %69, %70, %alloc_139, %alloc_140, %71, %alloc_133, %alloc_134, %72, %alloc_128, %alloc_129, %73, %alloc_123, %alloc_124, %74, %alloc_118, %alloc_119, %75, %alloc_108, %alloc_109, %76, %alloc_103, %alloc_104, %77, %alloc_113, %alloc_114, %78, %alloc_98, %alloc_99, %79, %alloc_93, %alloc_94, %80, %alloc_83, %alloc_84, %81, %alloc_78, %alloc_79, %82, %alloc_88, %alloc_89, %83, %alloc_73, %alloc_74, %84, %alloc_68, %alloc_69, %85, %alloc_58, %alloc_59, %86, %alloc_53, %alloc_54, %87, %alloc_63, %alloc_64, %88, %alloc_48, %alloc_49, %89, %alloc_43, %alloc_44, %90, %92 : memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>
  }
}