// RUN: byteir-opt %s -byteir-bufferize-opt | FileCheck %s

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
#map10 = affine_map<(d0) -> (d0 * 32)>
#map11 = affine_map<(d0) -> (d0 * -32 + 1000, 32)>
#map12 = affine_map<(d0, d1) -> (d1 * -32 + 1000, 32, d0)>
#map13 = affine_map<(d0, d1) -> (d1 * -32 + 1000, 32, d0 + 1)>
#map14 = affine_map<(d0)[s0] -> (d0 * 32 + s0)>
module @IrToMhlo.2452 {
  func.func private @Unknown0(%arg0: tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %c224 = arith.constant 224 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4x3x224x224xf16>
    %1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0) -> (tensor<4x3x224x224xf16>) {
      %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4x3x224x224xf16>) {
        %3 = scf.for %arg5 = %c0 to %c224 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4x3x224x224xf16>) {
          %4 = scf.for %arg7 = %c0 to %c224 step %c1 iter_args(%arg8 = %arg6) -> (tensor<4x3x224x224xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x3x224x224xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x3x224x224xf16>
            scf.yield %inserted_slice : tensor<4x3x224x224xf16>
          }
          scf.yield %4 : tensor<4x3x224x224xf16>
        }
        scf.yield %3 : tensor<4x3x224x224xf16>
      }
      scf.yield %2 : tensor<4x3x224x224xf16>
    }
    return %1 : tensor<4x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<64x3x7x7xf16>
    %1 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %0) -> (tensor<64x3x7x7xf16>) {
      %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2) -> (tensor<64x3x7x7xf16>) {
        %3 = scf.for %arg5 = %c0 to %c7 step %c1 iter_args(%arg6 = %arg4) -> (tensor<64x3x7x7xf16>) {
          %4 = scf.for %arg7 = %c0 to %c7 step %c1 iter_args(%arg8 = %arg6) -> (tensor<64x3x7x7xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<64x3x7x7xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<64x3x7x7xf16>
            scf.yield %inserted_slice : tensor<64x3x7x7xf16>
          }
          scf.yield %4 : tensor<64x3x7x7xf16>
        }
        scf.yield %3 : tensor<64x3x7x7xf16>
      }
      scf.yield %2 : tensor<64x3x7x7xf16>
    }
    return %1 : tensor<64x3x7x7xf16>
  }
  func.func private @Unknown3(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<64x64x3x3xf16>
    %1 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %0) -> (tensor<64x64x3x3xf16>) {
      %2 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<64x64x3x3xf16>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<64x64x3x3xf16>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<64x64x3x3xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<64x64x3x3xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<64x64x3x3xf16>
            scf.yield %inserted_slice : tensor<64x64x3x3xf16>
          }
          scf.yield %4 : tensor<64x64x3x3xf16>
        }
        scf.yield %3 : tensor<64x64x3x3xf16>
      }
      scf.yield %2 : tensor<64x64x3x3xf16>
    }
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown7(%arg0: tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<128x64x1x1xf16>
    %1 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %0) -> (tensor<128x64x1x1xf16>) {
      %2 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<128x64x1x1xf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<128x64x1x1xf32> to tensor<f32>
        %3 = tensor.empty() : tensor<f16>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%3 : tensor<f16>) {
        ^bb0(%in: f32, %out: f16):
          %5 = arith.truncf %in : f32 to f16
          linalg.yield %5 : f16
        } -> tensor<f16>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<128x64x1x1xf16>
        scf.yield %inserted_slice : tensor<128x64x1x1xf16>
      }
      scf.yield %2 : tensor<128x64x1x1xf16>
    }
    return %1 : tensor<128x64x1x1xf16>
  }
  func.func private @Unknown8(%arg0: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<128x64x3x3xf16>
    %1 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %0) -> (tensor<128x64x3x3xf16>) {
      %2 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<128x64x3x3xf16>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<128x64x3x3xf16>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<128x64x3x3xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<128x64x3x3xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<128x64x3x3xf16>
            scf.yield %inserted_slice : tensor<128x64x3x3xf16>
          }
          scf.yield %4 : tensor<128x64x3x3xf16>
        }
        scf.yield %3 : tensor<128x64x3x3xf16>
      }
      scf.yield %2 : tensor<128x64x3x3xf16>
    }
    return %1 : tensor<128x64x3x3xf16>
  }
  func.func private @Unknown9(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<128x128x3x3xf16>
    %1 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %0) -> (tensor<128x128x3x3xf16>) {
      %2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %arg2) -> (tensor<128x128x3x3xf16>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<128x128x3x3xf16>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<128x128x3x3xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<128x128x3x3xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<128x128x3x3xf16>
            scf.yield %inserted_slice : tensor<128x128x3x3xf16>
          }
          scf.yield %4 : tensor<128x128x3x3xf16>
        }
        scf.yield %3 : tensor<128x128x3x3xf16>
      }
      scf.yield %2 : tensor<128x128x3x3xf16>
    }
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown12(%arg0: tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<256x128x1x1xf16>
    %1 = scf.for %arg1 = %c0 to %c256 step %c1 iter_args(%arg2 = %0) -> (tensor<256x128x1x1xf16>) {
      %2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %arg2) -> (tensor<256x128x1x1xf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<256x128x1x1xf32> to tensor<f32>
        %3 = tensor.empty() : tensor<f16>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%3 : tensor<f16>) {
        ^bb0(%in: f32, %out: f16):
          %5 = arith.truncf %in : f32 to f16
          linalg.yield %5 : f16
        } -> tensor<f16>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<256x128x1x1xf16>
        scf.yield %inserted_slice : tensor<256x128x1x1xf16>
      }
      scf.yield %2 : tensor<256x128x1x1xf16>
    }
    return %1 : tensor<256x128x1x1xf16>
  }
  func.func private @Unknown13(%arg0: tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<256x128x3x3xf16>
    %1 = scf.for %arg1 = %c0 to %c256 step %c1 iter_args(%arg2 = %0) -> (tensor<256x128x3x3xf16>) {
      %2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %arg2) -> (tensor<256x128x3x3xf16>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<256x128x3x3xf16>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<256x128x3x3xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<256x128x3x3xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<256x128x3x3xf16>
            scf.yield %inserted_slice : tensor<256x128x3x3xf16>
          }
          scf.yield %4 : tensor<256x128x3x3xf16>
        }
        scf.yield %3 : tensor<256x128x3x3xf16>
      }
      scf.yield %2 : tensor<256x128x3x3xf16>
    }
    return %1 : tensor<256x128x3x3xf16>
  }
  func.func private @Unknown14(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<256x256x3x3xf16>
    %1 = scf.for %arg1 = %c0 to %c256 step %c1 iter_args(%arg2 = %0) -> (tensor<256x256x3x3xf16>) {
      %2 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %arg2) -> (tensor<256x256x3x3xf16>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<256x256x3x3xf16>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<256x256x3x3xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<256x256x3x3xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<256x256x3x3xf16>
            scf.yield %inserted_slice : tensor<256x256x3x3xf16>
          }
          scf.yield %4 : tensor<256x256x3x3xf16>
        }
        scf.yield %3 : tensor<256x256x3x3xf16>
      }
      scf.yield %2 : tensor<256x256x3x3xf16>
    }
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown17(%arg0: tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<512x256x1x1xf16>
    %1 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %0) -> (tensor<512x256x1x1xf16>) {
      %2 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %arg2) -> (tensor<512x256x1x1xf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<512x256x1x1xf32> to tensor<f32>
        %3 = tensor.empty() : tensor<f16>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%3 : tensor<f16>) {
        ^bb0(%in: f32, %out: f16):
          %5 = arith.truncf %in : f32 to f16
          linalg.yield %5 : f16
        } -> tensor<f16>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<512x256x1x1xf16>
        scf.yield %inserted_slice : tensor<512x256x1x1xf16>
      }
      scf.yield %2 : tensor<512x256x1x1xf16>
    }
    return %1 : tensor<512x256x1x1xf16>
  }
  func.func private @Unknown18(%arg0: tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<512x256x3x3xf16>
    %1 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %0) -> (tensor<512x256x3x3xf16>) {
      %2 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %arg2) -> (tensor<512x256x3x3xf16>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<512x256x3x3xf16>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<512x256x3x3xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<512x256x3x3xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<512x256x3x3xf16>
            scf.yield %inserted_slice : tensor<512x256x3x3xf16>
          }
          scf.yield %4 : tensor<512x256x3x3xf16>
        }
        scf.yield %3 : tensor<512x256x3x3xf16>
      }
      scf.yield %2 : tensor<512x256x3x3xf16>
    }
    return %1 : tensor<512x256x3x3xf16>
  }
  func.func private @Unknown19(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<512x512x3x3xf16>
    %1 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %0) -> (tensor<512x512x3x3xf16>) {
      %2 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %arg2) -> (tensor<512x512x3x3xf16>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<512x512x3x3xf16>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<512x512x3x3xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<512x512x3x3xf32> to tensor<f32>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f32, %out: f16):
              %7 = arith.truncf %in : f32 to f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<512x512x3x3xf16>
            scf.yield %inserted_slice : tensor<512x512x3x3xf16>
          }
          scf.yield %4 : tensor<512x512x3x3xf16>
        }
        scf.yield %3 : tensor<512x512x3x3xf16>
      }
      scf.yield %2 : tensor<512x512x3x3xf16>
    }
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown22(%arg0: tensor<4x1000xf32>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant -2.500000e-01 : f32
    %0 = tensor.empty() : tensor<4x1000xf16>
    %1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0) -> (tensor<4x1000xf16>) {
      %2 = scf.for %arg3 = %c0 to %c1000 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4x1000xf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3] [1, 1] [1, 1] : tensor<4x1000xf32> to tensor<f32>
        %3 = tensor.empty() : tensor<f16>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%3 : tensor<f16>) {
        ^bb0(%in: f32, %out: f16):
          %5 = arith.mulf %in, %cst : f32
          %6 = arith.truncf %5 : f32 to f16
          linalg.yield %6 : f16
        } -> tensor<f16>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<f16> into tensor<4x1000xf16>
        scf.yield %inserted_slice : tensor<4x1000xf16>
      }
      scf.yield %2 : tensor<4x1000xf16>
    }
    return %1 : tensor<4x1000xf16>
  }
  func.func private @Unknown23(%arg0: tensor<1000x512xf32>) -> tensor<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1000x512xf16>
    %1 = scf.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %0) -> (tensor<1000x512xf16>) {
      %2 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1000x512xf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3] [1, 1] [1, 1] : tensor<1000x512xf32> to tensor<f32>
        %3 = tensor.empty() : tensor<f16>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%3 : tensor<f16>) {
        ^bb0(%in: f32, %out: f16):
          %5 = arith.truncf %in : f32 to f16
          linalg.yield %5 : f16
        } -> tensor<f16>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<f16> into tensor<1000x512xf16>
        scf.yield %inserted_slice : tensor<1000x512xf16>
      }
      scf.yield %2 : tensor<1000x512xf16>
    }
    return %1 : tensor<1000x512xf16>
  }
  func.func private @Unknown24(%arg0: tensor<1000xf32>) -> tensor<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1000xf16>
    %1 = scf.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %0) -> (tensor<1000xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1] [1] [1] : tensor<1000xf32> to tensor<f32>
      %2 = tensor.empty() : tensor<f16>
      %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%2 : tensor<f16>) {
      ^bb0(%in: f32, %out: f16):
        %4 = arith.truncf %in : f32 to f16
        linalg.yield %4 : f16
      } -> tensor<f16>
      %inserted_slice = tensor.insert_slice %3 into %arg2[%arg1] [1] [1] : tensor<f16> into tensor<1000xf16>
      scf.yield %inserted_slice : tensor<1000xf16>
    }
    return %1 : tensor<1000xf16>
  }
  func.func private @Unknown25(%arg0: tensor<4x1000xf16>) -> tensor<4xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4xf16>
    %1 = scf.forall (%arg1) in (4) shared_outs(%arg2 = %0) -> (tensor<4xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [1, 1000] [1, 1] : tensor<4x1000xf16> to tensor<1000xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] : tensor<1000xf16> into tensor<1x1000xf16>
      %extracted_slice_0 = tensor.extract_slice %arg2[%arg1] [1] [1] : tensor<4xf16> to tensor<f16>
      %2 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<512xf16>
      %3 = scf.forall (%arg3) in (512) shared_outs(%arg4 = %2) -> (tensor<512xf16>) {
        %21 = affine.min #map1(%arg3)
        %22 = affine.min #map2(%arg3)
        %23 = affine.apply #map3(%22, %21)
        %extracted_slice_9 = tensor.extract_slice %expanded[0, %21] [1, %23] [1, 1] : tensor<1x1000xf16> to tensor<?xf16>
        %expanded_10 = tensor.expand_shape %extracted_slice_9 [[0, 1]] : tensor<?xf16> into tensor<1x?xf16>
        %dim = tensor.dim %extracted_slice_9, %c0 : tensor<?xf16>
        %24 = arith.cmpi ugt, %dim, %c0 : index
        %25 = scf.if %24 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %26 = arith.addf %25, %cst : f16
        %dim_11 = tensor.dim %extracted_slice_9, %c0 : tensor<?xf16>
        %27 = arith.cmpi ugt, %dim_11, %c1 : index
        %28 = scf.if %27 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c1] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %29 = arith.addf %26, %28 : f16
        %extracted_slice_12 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<512xf16> to tensor<f16>
        %inserted = tensor.insert %29 into %extracted_slice_12[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<512xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_1 = tensor.expand_shape %3 [[0, 1]] : tensor<512xf16> into tensor<256x2xf16>
      %4 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<256xf16>
      %5 = scf.forall (%arg3) in (256) shared_outs(%arg4 = %4) -> (tensor<256xf16>) {
        %extracted = tensor.extract %expanded_1[%arg3, %c0] : tensor<256x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_1[%arg3, %c1] : tensor<256x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<256xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<256xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_2 = tensor.expand_shape %5 [[0, 1]] : tensor<256xf16> into tensor<128x2xf16>
      %6 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128xf16>
      %7 = scf.forall (%arg3) in (128) shared_outs(%arg4 = %6) -> (tensor<128xf16>) {
        %extracted = tensor.extract %expanded_2[%arg3, %c0] : tensor<128x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_2[%arg3, %c1] : tensor<128x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<128xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<128xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_3 = tensor.expand_shape %7 [[0, 1]] : tensor<128xf16> into tensor<64x2xf16>
      %8 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<64xf16>
      %9 = scf.forall (%arg3) in (64) shared_outs(%arg4 = %8) -> (tensor<64xf16>) {
        %extracted = tensor.extract %expanded_3[%arg3, %c0] : tensor<64x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_3[%arg3, %c1] : tensor<64x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<64xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<64xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_4 = tensor.expand_shape %9 [[0, 1]] : tensor<64xf16> into tensor<32x2xf16>
      %10 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf16>
      %11 = scf.forall (%arg3) in (32) shared_outs(%arg4 = %10) -> (tensor<32xf16>) {
        %extracted = tensor.extract %expanded_4[%arg3, %c0] : tensor<32x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_4[%arg3, %c1] : tensor<32x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<32xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<32xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_5 = tensor.expand_shape %11 [[0, 1]] : tensor<32xf16> into tensor<16x2xf16>
      %12 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16xf16>
      %13 = scf.forall (%arg3) in (16) shared_outs(%arg4 = %12) -> (tensor<16xf16>) {
        %extracted = tensor.extract %expanded_5[%arg3, %c0] : tensor<16x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_5[%arg3, %c1] : tensor<16x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<16xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<16xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_6 = tensor.expand_shape %13 [[0, 1]] : tensor<16xf16> into tensor<8x2xf16>
      %14 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<8xf16>
      %15 = scf.forall (%arg3) in (8) shared_outs(%arg4 = %14) -> (tensor<8xf16>) {
        %extracted = tensor.extract %expanded_6[%arg3, %c0] : tensor<8x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_6[%arg3, %c1] : tensor<8x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<8xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<8xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_7 = tensor.expand_shape %15 [[0, 1]] : tensor<8xf16> into tensor<4x2xf16>
      %16 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4xf16>
      %17 = scf.forall (%arg3) in (4) shared_outs(%arg4 = %16) -> (tensor<4xf16>) {
        %extracted = tensor.extract %expanded_7[%arg3, %c0] : tensor<4x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_7[%arg3, %c1] : tensor<4x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<4xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<4xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_8 = tensor.expand_shape %17 [[0, 1]] : tensor<4xf16> into tensor<2x2xf16>
      %18 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2xf16>
      %19 = scf.forall (%arg3) in (2) shared_outs(%arg4 = %18) -> (tensor<2xf16>) {
        %extracted = tensor.extract %expanded_8[%arg3, %c0] : tensor<2x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_8[%arg3, %c1] : tensor<2x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<2xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<2xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %20 = scf.forall (%arg3) in (1) shared_outs(%arg4 = %extracted_slice_0) -> (tensor<f16>) {
        %21 = affine.apply #map4(%arg3)
        %extracted = tensor.extract %19[%21] : tensor<2xf16>
        %22 = arith.addf %extracted, %cst : f16
        %23 = affine.apply #map5(%arg3)
        %extracted_9 = tensor.extract %19[%23] : tensor<2xf16>
        %24 = arith.addf %extracted_9, %22 : f16
        %inserted = tensor.insert %24 into %arg4[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[] [] [] : tensor<f16> into tensor<f16>
        }
      } {mapping = [#gpu.thread<x>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %20 into %arg2[%arg1] [1] [1] : tensor<f16> into tensor<4xf16>
      }
    } {mapping = [#gpu.block<x>]}
    return %1 : tensor<4xf16>
  }
  func.func private @Unknown26(%arg0: tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %c112 = arith.constant 112 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x112x112xf16>
    %1 = tensor.empty() : tensor<4x64x112x112xi1>
    %2:2 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0, %arg3 = %1) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) {
      %3:2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) {
        %4:2 = scf.for %arg7 = %c0 to %c112 step %c1 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) {
          %5:2 = scf.for %arg10 = %c0 to %c112 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x112x112xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %out: f16, %out_1: i1):
              %9 = arith.maximumf %in, %cst : f16
              %10 = arith.cmpf ogt, %9, %cst : f16
              linalg.yield %9, %10 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg11[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x64x112x112xf16>
            %inserted_slice_0 = tensor.insert_slice %8#1 into %arg12[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x64x112x112xi1>
            scf.yield %inserted_slice, %inserted_slice_0 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>
    }
    return %2#0, %2#1 : tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>
  }
  func.func private @Unknown28(%arg0: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = tensor.empty() : tensor<4x64x56x56xi1>
    %2:2 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0, %arg3 = %1) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
      %3:2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
        %4:2 = scf.for %arg7 = %c0 to %c56 step %c1 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
          %5:2 = scf.for %arg10 = %c0 to %c56 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %out: f16, %out_1: i1):
              %9 = arith.maximumf %in, %cst : f16
              %10 = arith.cmpf ogt, %9, %cst : f16
              linalg.yield %9, %10 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg11[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x64x56x56xf16>
            %inserted_slice_0 = tensor.insert_slice %8#1 into %arg12[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x64x56x56xi1>
            scf.yield %inserted_slice, %inserted_slice_0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
    }
    return %2#0, %2#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @Unknown30(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = tensor.empty() : tensor<4x64x56x56xi1>
    %2:2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
      %3:2 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
        %4:2 = scf.for %arg8 = %c0 to %c56 step %c1 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
          %5:2 = scf.for %arg11 = %c0 to %c56 step %c1 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %in_2: f16, %out: f16, %out_3: i1):
              %9 = arith.addf %in, %in_2 : f16
              %10 = arith.maximumf %9, %cst : f16
              %11 = arith.cmpf ogt, %10, %cst : f16
              linalg.yield %10, %11 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg12[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x64x56x56xf16>
            %inserted_slice_1 = tensor.insert_slice %8#1 into %arg13[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x64x56x56xi1>
            scf.yield %inserted_slice, %inserted_slice_1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
    }
    return %2#0, %2#1 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>
  }
  func.func private @Unknown37(%arg0: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = tensor.empty() : tensor<4x128x28x28xi1>
    %2:2 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0, %arg3 = %1) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
      %3:2 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
        %4:2 = scf.for %arg7 = %c0 to %c28 step %c1 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
          %5:2 = scf.for %arg10 = %c0 to %c28 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x128x28x28xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %out: f16, %out_1: i1):
              %9 = arith.maximumf %in, %cst : f16
              %10 = arith.cmpf ogt, %9, %cst : f16
              linalg.yield %9, %10 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg11[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x128x28x28xf16>
            %inserted_slice_0 = tensor.insert_slice %8#1 into %arg12[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x128x28x28xi1>
            scf.yield %inserted_slice, %inserted_slice_0 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
    }
    return %2#0, %2#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @Unknown39(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = tensor.empty() : tensor<4x128x28x28xi1>
    %2:2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
      %3:2 = scf.for %arg5 = %c0 to %c128 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
        %4:2 = scf.for %arg8 = %c0 to %c28 step %c1 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
          %5:2 = scf.for %arg11 = %c0 to %c28 step %c1 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x128x28x28xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x128x28x28xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %in_2: f16, %out: f16, %out_3: i1):
              %9 = arith.addf %in, %in_2 : f16
              %10 = arith.maximumf %9, %cst : f16
              %11 = arith.cmpf ogt, %10, %cst : f16
              linalg.yield %10, %11 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg12[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x128x28x28xf16>
            %inserted_slice_1 = tensor.insert_slice %8#1 into %arg13[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x128x28x28xi1>
            scf.yield %inserted_slice, %inserted_slice_1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
    }
    return %2#0, %2#1 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>
  }
  func.func private @Unknown46(%arg0: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = tensor.empty() : tensor<4x256x14x14xi1>
    %2:2 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0, %arg3 = %1) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
      %3:2 = scf.for %arg4 = %c0 to %c256 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
        %4:2 = scf.for %arg7 = %c0 to %c14 step %c1 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
          %5:2 = scf.for %arg10 = %c0 to %c14 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x256x14x14xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %out: f16, %out_1: i1):
              %9 = arith.maximumf %in, %cst : f16
              %10 = arith.cmpf ogt, %9, %cst : f16
              linalg.yield %9, %10 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg11[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x256x14x14xf16>
            %inserted_slice_0 = tensor.insert_slice %8#1 into %arg12[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x256x14x14xi1>
            scf.yield %inserted_slice, %inserted_slice_0 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
    }
    return %2#0, %2#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @Unknown48(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = tensor.empty() : tensor<4x256x14x14xi1>
    %2:2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
      %3:2 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
        %4:2 = scf.for %arg8 = %c0 to %c14 step %c1 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
          %5:2 = scf.for %arg11 = %c0 to %c14 step %c1 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x256x14x14xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x256x14x14xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %in_2: f16, %out: f16, %out_3: i1):
              %9 = arith.addf %in, %in_2 : f16
              %10 = arith.maximumf %9, %cst : f16
              %11 = arith.cmpf ogt, %10, %cst : f16
              linalg.yield %10, %11 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg12[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x256x14x14xf16>
            %inserted_slice_1 = tensor.insert_slice %8#1 into %arg13[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x256x14x14xi1>
            scf.yield %inserted_slice, %inserted_slice_1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
    }
    return %2#0, %2#1 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>
  }
  func.func private @Unknown55(%arg0: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = tensor.empty() : tensor<4x512x7x7xi1>
    %2:2 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0, %arg3 = %1) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
      %3:2 = scf.for %arg4 = %c0 to %c512 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
        %4:2 = scf.for %arg7 = %c0 to %c7 step %c1 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
          %5:2 = scf.for %arg10 = %c0 to %c7 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %out: f16, %out_1: i1):
              %9 = arith.maximumf %in, %cst : f16
              %10 = arith.cmpf ogt, %9, %cst : f16
              linalg.yield %9, %10 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg11[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x512x7x7xf16>
            %inserted_slice_0 = tensor.insert_slice %8#1 into %arg12[%arg1, %arg4, %arg7, %arg10] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x512x7x7xi1>
            scf.yield %inserted_slice, %inserted_slice_0 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
    }
    return %2#0, %2#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @Unknown57(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = tensor.empty() : tensor<4x512x7x7xi1>
    %2:2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
      %3:2 = scf.for %arg5 = %c0 to %c512 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
        %4:2 = scf.for %arg8 = %c0 to %c7 step %c1 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
          %5:2 = scf.for %arg11 = %c0 to %c7 step %c1 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xf16> to tensor<f16>
            %6 = tensor.empty() : tensor<f16>
            %7 = tensor.empty() : tensor<i1>
            %8:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%6, %7 : tensor<f16>, tensor<i1>) {
            ^bb0(%in: f16, %in_2: f16, %out: f16, %out_3: i1):
              %9 = arith.addf %in, %in_2 : f16
              %10 = arith.maximumf %9, %cst : f16
              %11 = arith.cmpf ogt, %10, %cst : f16
              linalg.yield %10, %11 : f16, i1
            } -> (tensor<f16>, tensor<i1>)
            %inserted_slice = tensor.insert_slice %8#0 into %arg12[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x512x7x7xf16>
            %inserted_slice_1 = tensor.insert_slice %8#1 into %arg13[%arg2, %arg5, %arg8, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<i1> into tensor<4x512x7x7xi1>
            scf.yield %inserted_slice, %inserted_slice_1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
          }
          scf.yield %5#0, %5#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
        }
        scf.yield %4#0, %4#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
      }
      scf.yield %3#0, %3#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
    }
    return %2#0, %2#1 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>
  }
  func.func private @Unknown62(%arg0: tensor<4x512x7x7xf16>) -> tensor<4x512xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<4x512x7x7xf16> into tensor<2048x49xf16>
    %0 = tensor.empty() : tensor<2048xf16>
    %1 = scf.forall (%arg1) in (2048) shared_outs(%arg2 = %0) -> (tensor<2048xf16>) {
      %extracted_slice = tensor.extract_slice %collapsed[%arg1, 0] [1, 49] [1, 1] : tensor<2048x49xf16> to tensor<49xf16>
      %expanded_0 = tensor.expand_shape %extracted_slice [[0, 1]] : tensor<49xf16> into tensor<1x49xf16>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg1] [1] [1] : tensor<2048xf16> to tensor<f16>
      %2 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<64xf16>
      %3 = scf.forall (%arg3) in (64) shared_outs(%arg4 = %2) -> (tensor<64xf16>) {
        %15 = affine.min #map6(%arg3)
        %16 = affine.min #map7(%arg3)
        %17 = affine.apply #map3(%16, %15)
        %extracted_slice_7 = tensor.extract_slice %expanded_0[0, %15] [1, %17] [1, 1] : tensor<1x49xf16> to tensor<?xf16>
        %expanded_8 = tensor.expand_shape %extracted_slice_7 [[0, 1]] : tensor<?xf16> into tensor<1x?xf16>
        %dim = tensor.dim %extracted_slice_7, %c0 : tensor<?xf16>
        %18 = arith.cmpi ugt, %dim, %c0 : index
        %19 = scf.if %18 -> (f16) {
          %extracted = tensor.extract %expanded_8[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %20 = arith.addf %19, %cst : f16
        %extracted_slice_9 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<64xf16> to tensor<f16>
        %inserted = tensor.insert %20 into %extracted_slice_9[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<64xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_2 = tensor.expand_shape %3 [[0, 1]] : tensor<64xf16> into tensor<32x2xf16>
      %4 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf16>
      %5 = scf.forall (%arg3) in (32) shared_outs(%arg4 = %4) -> (tensor<32xf16>) {
        %extracted = tensor.extract %expanded_2[%arg3, %c0] : tensor<32x2xf16>
        %15 = arith.addf %extracted, %cst : f16
        %extracted_7 = tensor.extract %expanded_2[%arg3, %c1] : tensor<32x2xf16>
        %16 = arith.addf %extracted_7, %15 : f16
        %extracted_slice_8 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<32xf16> to tensor<f16>
        %inserted = tensor.insert %16 into %extracted_slice_8[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<32xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_3 = tensor.expand_shape %5 [[0, 1]] : tensor<32xf16> into tensor<16x2xf16>
      %6 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16xf16>
      %7 = scf.forall (%arg3) in (16) shared_outs(%arg4 = %6) -> (tensor<16xf16>) {
        %extracted = tensor.extract %expanded_3[%arg3, %c0] : tensor<16x2xf16>
        %15 = arith.addf %extracted, %cst : f16
        %extracted_7 = tensor.extract %expanded_3[%arg3, %c1] : tensor<16x2xf16>
        %16 = arith.addf %extracted_7, %15 : f16
        %extracted_slice_8 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<16xf16> to tensor<f16>
        %inserted = tensor.insert %16 into %extracted_slice_8[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<16xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_4 = tensor.expand_shape %7 [[0, 1]] : tensor<16xf16> into tensor<8x2xf16>
      %8 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<8xf16>
      %9 = scf.forall (%arg3) in (8) shared_outs(%arg4 = %8) -> (tensor<8xf16>) {
        %extracted = tensor.extract %expanded_4[%arg3, %c0] : tensor<8x2xf16>
        %15 = arith.addf %extracted, %cst : f16
        %extracted_7 = tensor.extract %expanded_4[%arg3, %c1] : tensor<8x2xf16>
        %16 = arith.addf %extracted_7, %15 : f16
        %extracted_slice_8 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<8xf16> to tensor<f16>
        %inserted = tensor.insert %16 into %extracted_slice_8[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<8xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_5 = tensor.expand_shape %9 [[0, 1]] : tensor<8xf16> into tensor<4x2xf16>
      %10 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4xf16>
      %11 = scf.forall (%arg3) in (4) shared_outs(%arg4 = %10) -> (tensor<4xf16>) {
        %extracted = tensor.extract %expanded_5[%arg3, %c0] : tensor<4x2xf16>
        %15 = arith.addf %extracted, %cst : f16
        %extracted_7 = tensor.extract %expanded_5[%arg3, %c1] : tensor<4x2xf16>
        %16 = arith.addf %extracted_7, %15 : f16
        %extracted_slice_8 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<4xf16> to tensor<f16>
        %inserted = tensor.insert %16 into %extracted_slice_8[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<4xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_6 = tensor.expand_shape %11 [[0, 1]] : tensor<4xf16> into tensor<2x2xf16>
      %12 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2xf16>
      %13 = scf.forall (%arg3) in (2) shared_outs(%arg4 = %12) -> (tensor<2xf16>) {
        %extracted = tensor.extract %expanded_6[%arg3, %c0] : tensor<2x2xf16>
        %15 = arith.addf %extracted, %cst : f16
        %extracted_7 = tensor.extract %expanded_6[%arg3, %c1] : tensor<2x2xf16>
        %16 = arith.addf %extracted_7, %15 : f16
        %extracted_slice_8 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<2xf16> to tensor<f16>
        %inserted = tensor.insert %16 into %extracted_slice_8[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<2xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %14 = scf.forall (%arg3) in (1) shared_outs(%arg4 = %extracted_slice_1) -> (tensor<f16>) {
        %15 = affine.apply #map4(%arg3)
        %extracted = tensor.extract %13[%15] : tensor<2xf16>
        %16 = arith.addf %extracted, %cst : f16
        %17 = affine.apply #map5(%arg3)
        %extracted_7 = tensor.extract %13[%17] : tensor<2xf16>
        %18 = arith.addf %extracted_7, %16 : f16
        %inserted = tensor.insert %18 into %arg4[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[] [] [] : tensor<f16> into tensor<f16>
        }
      } {mapping = [#gpu.thread<x>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg2[%arg1] [1] [1] : tensor<f16> into tensor<2048xf16>
      }
    } {mapping = [#gpu.block<x>]}
    %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<2048xf16> into tensor<4x512xf16>
    return %expanded : tensor<4x512xf16>
  }
  func.func private @Unknown63(%arg0: tensor<4x512xf16>) -> tensor<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.040100e-02 : f16
    %0 = tensor.empty() : tensor<4x512xf16>
    %1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0) -> (tensor<4x512xf16>) {
      %2 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4x512xf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3] [1, 1] [1, 1] : tensor<4x512xf16> to tensor<f16>
        %3 = tensor.empty() : tensor<f16>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%3 : tensor<f16>) {
        ^bb0(%in: f16, %out: f16):
          %5 = arith.mulf %in, %cst : f16
          linalg.yield %5 : f16
        } -> tensor<f16>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<f16> into tensor<4x512xf16>
        scf.yield %inserted_slice : tensor<4x512xf16>
      }
      scf.yield %2 : tensor<4x512xf16>
    }
    return %1 : tensor<4x512xf16>
  }
  func.func private @Unknown64(%arg0: tensor<1000xf16>, %arg1: tensor<4x1000xf16>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4x1000xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x1000xf16>) {
      %2 = scf.for %arg4 = %c0 to %c1000 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x1000xf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg4] [1] [1] : tensor<1000xf16> to tensor<f16>
        %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4] [1, 1] [1, 1] : tensor<4x1000xf16> to tensor<f16>
        %3 = tensor.empty() : tensor<f16>
        %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%3 : tensor<f16>) {
        ^bb0(%in: f16, %in_1: f16, %out: f16):
          %5 = arith.addf %in_1, %in : f16
          linalg.yield %5 : f16
        } -> tensor<f16>
        %inserted_slice = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [1, 1] [1, 1] : tensor<f16> into tensor<4x1000xf16>
        scf.yield %inserted_slice : tensor<4x1000xf16>
      }
      scf.yield %2 : tensor<4x1000xf16>
    }
    return %1 : tensor<4x1000xf16>
  }
  func.func private @Unknown65(%arg0: tensor<4x1000xf16>) -> tensor<4xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4xf16>
    %1 = scf.forall (%arg1) in (4) shared_outs(%arg2 = %0) -> (tensor<4xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [1, 1000] [1, 1] : tensor<4x1000xf16> to tensor<1000xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] : tensor<1000xf16> into tensor<1x1000xf16>
      %extracted_slice_0 = tensor.extract_slice %arg2[%arg1] [1] [1] : tensor<4xf16> to tensor<f16>
      %2 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<512xf16>
      %3 = scf.forall (%arg3) in (512) shared_outs(%arg4 = %2) -> (tensor<512xf16>) {
        %21 = affine.min #map1(%arg3)
        %22 = affine.min #map2(%arg3)
        %23 = affine.apply #map3(%22, %21)
        %extracted_slice_9 = tensor.extract_slice %expanded[0, %21] [1, %23] [1, 1] : tensor<1x1000xf16> to tensor<?xf16>
        %expanded_10 = tensor.expand_shape %extracted_slice_9 [[0, 1]] : tensor<?xf16> into tensor<1x?xf16>
        %dim = tensor.dim %extracted_slice_9, %c0 : tensor<?xf16>
        %24 = arith.cmpi ugt, %dim, %c0 : index
        %25 = scf.if %24 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %dim_11 = tensor.dim %extracted_slice_9, %c0 : tensor<?xf16>
        %26 = arith.cmpi ugt, %dim_11, %c1 : index
        %27 = scf.if %26 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c1] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %28 = arith.maximumf %25, %27 : f16
        %extracted_slice_12 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<512xf16> to tensor<f16>
        %inserted = tensor.insert %28 into %extracted_slice_12[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<512xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_1 = tensor.expand_shape %3 [[0, 1]] : tensor<512xf16> into tensor<256x2xf16>
      %4 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<256xf16>
      %5 = scf.forall (%arg3) in (256) shared_outs(%arg4 = %4) -> (tensor<256xf16>) {
        %extracted = tensor.extract %expanded_1[%arg3, %c0] : tensor<256x2xf16>
        %extracted_9 = tensor.extract %expanded_1[%arg3, %c1] : tensor<256x2xf16>
        %21 = arith.maximumf %extracted_9, %extracted : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<256xf16> to tensor<f16>
        %inserted = tensor.insert %21 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<256xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_2 = tensor.expand_shape %5 [[0, 1]] : tensor<256xf16> into tensor<128x2xf16>
      %6 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128xf16>
      %7 = scf.forall (%arg3) in (128) shared_outs(%arg4 = %6) -> (tensor<128xf16>) {
        %extracted = tensor.extract %expanded_2[%arg3, %c0] : tensor<128x2xf16>
        %extracted_9 = tensor.extract %expanded_2[%arg3, %c1] : tensor<128x2xf16>
        %21 = arith.maximumf %extracted_9, %extracted : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<128xf16> to tensor<f16>
        %inserted = tensor.insert %21 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<128xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_3 = tensor.expand_shape %7 [[0, 1]] : tensor<128xf16> into tensor<64x2xf16>
      %8 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<64xf16>
      %9 = scf.forall (%arg3) in (64) shared_outs(%arg4 = %8) -> (tensor<64xf16>) {
        %extracted = tensor.extract %expanded_3[%arg3, %c0] : tensor<64x2xf16>
        %extracted_9 = tensor.extract %expanded_3[%arg3, %c1] : tensor<64x2xf16>
        %21 = arith.maximumf %extracted_9, %extracted : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<64xf16> to tensor<f16>
        %inserted = tensor.insert %21 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<64xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_4 = tensor.expand_shape %9 [[0, 1]] : tensor<64xf16> into tensor<32x2xf16>
      %10 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf16>
      %11 = scf.forall (%arg3) in (32) shared_outs(%arg4 = %10) -> (tensor<32xf16>) {
        %extracted = tensor.extract %expanded_4[%arg3, %c0] : tensor<32x2xf16>
        %extracted_9 = tensor.extract %expanded_4[%arg3, %c1] : tensor<32x2xf16>
        %21 = arith.maximumf %extracted_9, %extracted : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<32xf16> to tensor<f16>
        %inserted = tensor.insert %21 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<32xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_5 = tensor.expand_shape %11 [[0, 1]] : tensor<32xf16> into tensor<16x2xf16>
      %12 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16xf16>
      %13 = scf.forall (%arg3) in (16) shared_outs(%arg4 = %12) -> (tensor<16xf16>) {
        %extracted = tensor.extract %expanded_5[%arg3, %c0] : tensor<16x2xf16>
        %extracted_9 = tensor.extract %expanded_5[%arg3, %c1] : tensor<16x2xf16>
        %21 = arith.maximumf %extracted_9, %extracted : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<16xf16> to tensor<f16>
        %inserted = tensor.insert %21 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<16xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_6 = tensor.expand_shape %13 [[0, 1]] : tensor<16xf16> into tensor<8x2xf16>
      %14 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<8xf16>
      %15 = scf.forall (%arg3) in (8) shared_outs(%arg4 = %14) -> (tensor<8xf16>) {
        %extracted = tensor.extract %expanded_6[%arg3, %c0] : tensor<8x2xf16>
        %extracted_9 = tensor.extract %expanded_6[%arg3, %c1] : tensor<8x2xf16>
        %21 = arith.maximumf %extracted_9, %extracted : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<8xf16> to tensor<f16>
        %inserted = tensor.insert %21 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<8xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_7 = tensor.expand_shape %15 [[0, 1]] : tensor<8xf16> into tensor<4x2xf16>
      %16 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4xf16>
      %17 = scf.forall (%arg3) in (4) shared_outs(%arg4 = %16) -> (tensor<4xf16>) {
        %extracted = tensor.extract %expanded_7[%arg3, %c0] : tensor<4x2xf16>
        %extracted_9 = tensor.extract %expanded_7[%arg3, %c1] : tensor<4x2xf16>
        %21 = arith.maximumf %extracted_9, %extracted : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<4xf16> to tensor<f16>
        %inserted = tensor.insert %21 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<4xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_8 = tensor.expand_shape %17 [[0, 1]] : tensor<4xf16> into tensor<2x2xf16>
      %18 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2xf16>
      %19 = scf.forall (%arg3) in (2) shared_outs(%arg4 = %18) -> (tensor<2xf16>) {
        %extracted = tensor.extract %expanded_8[%arg3, %c0] : tensor<2x2xf16>
        %extracted_9 = tensor.extract %expanded_8[%arg3, %c1] : tensor<2x2xf16>
        %21 = arith.maximumf %extracted_9, %extracted : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<2xf16> to tensor<f16>
        %inserted = tensor.insert %21 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<2xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %20 = scf.forall (%arg3) in (1) shared_outs(%arg4 = %extracted_slice_0) -> (tensor<f16>) {
        %21 = affine.apply #map4(%arg3)
        %extracted = tensor.extract %19[%21] : tensor<2xf16>
        %22 = affine.apply #map5(%arg3)
        %extracted_9 = tensor.extract %19[%22] : tensor<2xf16>
        %23 = arith.maximumf %extracted_9, %extracted : f16
        %inserted = tensor.insert %23 into %arg4[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[] [] [] : tensor<f16> into tensor<f16>
        }
      } {mapping = [#gpu.thread<x>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %20 into %arg2[%arg1] [1] [1] : tensor<f16> into tensor<4xf16>
      }
    } {mapping = [#gpu.block<x>]}
    return %1 : tensor<4xf16>
  }
  func.func private @Unknown66(%arg0: tensor<4xf16>, %arg1: tensor<4x1000xf16>) -> tensor<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4x1000xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x1000xf16>) {
      %2 = scf.for %arg4 = %c0 to %c1000 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x1000xf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg2] [1] [1] : tensor<4xf16> to tensor<f16>
        %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4] [1, 1] [1, 1] : tensor<4x1000xf16> to tensor<f16>
        %3 = tensor.empty() : tensor<f16>
        %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%3 : tensor<f16>) {
        ^bb0(%in: f16, %in_1: f16, %out: f16):
          %5 = arith.subf %in_1, %in : f16
          linalg.yield %5 : f16
        } -> tensor<f16>
        %inserted_slice = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [1, 1] [1, 1] : tensor<f16> into tensor<4x1000xf16>
        scf.yield %inserted_slice : tensor<4x1000xf16>
      }
      scf.yield %2 : tensor<4x1000xf16>
    }
    return %1 : tensor<4x1000xf16>
  }
  func.func private @Unknown67(%arg0: tensor<4x1000xf16>) -> tensor<4xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4xf16>
    %1 = scf.forall (%arg1) in (4) shared_outs(%arg2 = %0) -> (tensor<4xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [1, 1000] [1, 1] : tensor<4x1000xf16> to tensor<1000xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] : tensor<1000xf16> into tensor<1x1000xf16>
      %extracted_slice_0 = tensor.extract_slice %arg2[%arg1] [1] [1] : tensor<4xf16> to tensor<f16>
      %2 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<512xf16>
      %3 = scf.forall (%arg3) in (512) shared_outs(%arg4 = %2) -> (tensor<512xf16>) {
        %21 = affine.min #map1(%arg3)
        %22 = affine.min #map2(%arg3)
        %23 = affine.apply #map3(%22, %21)
        %extracted_slice_9 = tensor.extract_slice %expanded[0, %21] [1, %23] [1, 1] : tensor<1x1000xf16> to tensor<?xf16>
        %expanded_10 = tensor.expand_shape %extracted_slice_9 [[0, 1]] : tensor<?xf16> into tensor<1x?xf16>
        %dim = tensor.dim %extracted_slice_9, %c0 : tensor<?xf16>
        %24 = arith.cmpi ugt, %dim, %c0 : index
        %25 = scf.if %24 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %26 = math.exp %25 : f16
        %27 = arith.addf %26, %cst : f16
        %dim_11 = tensor.dim %extracted_slice_9, %c0 : tensor<?xf16>
        %28 = arith.cmpi ugt, %dim_11, %c1 : index
        %29 = scf.if %28 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c1] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %30 = math.exp %29 : f16
        %31 = arith.addf %27, %30 : f16
        %extracted_slice_12 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<512xf16> to tensor<f16>
        %inserted = tensor.insert %31 into %extracted_slice_12[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<512xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_1 = tensor.expand_shape %3 [[0, 1]] : tensor<512xf16> into tensor<256x2xf16>
      %4 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<256xf16>
      %5 = scf.forall (%arg3) in (256) shared_outs(%arg4 = %4) -> (tensor<256xf16>) {
        %extracted = tensor.extract %expanded_1[%arg3, %c0] : tensor<256x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_1[%arg3, %c1] : tensor<256x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<256xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<256xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_2 = tensor.expand_shape %5 [[0, 1]] : tensor<256xf16> into tensor<128x2xf16>
      %6 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128xf16>
      %7 = scf.forall (%arg3) in (128) shared_outs(%arg4 = %6) -> (tensor<128xf16>) {
        %extracted = tensor.extract %expanded_2[%arg3, %c0] : tensor<128x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_2[%arg3, %c1] : tensor<128x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<128xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<128xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_3 = tensor.expand_shape %7 [[0, 1]] : tensor<128xf16> into tensor<64x2xf16>
      %8 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<64xf16>
      %9 = scf.forall (%arg3) in (64) shared_outs(%arg4 = %8) -> (tensor<64xf16>) {
        %extracted = tensor.extract %expanded_3[%arg3, %c0] : tensor<64x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_3[%arg3, %c1] : tensor<64x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<64xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<64xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_4 = tensor.expand_shape %9 [[0, 1]] : tensor<64xf16> into tensor<32x2xf16>
      %10 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf16>
      %11 = scf.forall (%arg3) in (32) shared_outs(%arg4 = %10) -> (tensor<32xf16>) {
        %extracted = tensor.extract %expanded_4[%arg3, %c0] : tensor<32x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_4[%arg3, %c1] : tensor<32x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<32xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<32xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_5 = tensor.expand_shape %11 [[0, 1]] : tensor<32xf16> into tensor<16x2xf16>
      %12 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16xf16>
      %13 = scf.forall (%arg3) in (16) shared_outs(%arg4 = %12) -> (tensor<16xf16>) {
        %extracted = tensor.extract %expanded_5[%arg3, %c0] : tensor<16x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_5[%arg3, %c1] : tensor<16x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<16xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<16xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_6 = tensor.expand_shape %13 [[0, 1]] : tensor<16xf16> into tensor<8x2xf16>
      %14 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<8xf16>
      %15 = scf.forall (%arg3) in (8) shared_outs(%arg4 = %14) -> (tensor<8xf16>) {
        %extracted = tensor.extract %expanded_6[%arg3, %c0] : tensor<8x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_6[%arg3, %c1] : tensor<8x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<8xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<8xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_7 = tensor.expand_shape %15 [[0, 1]] : tensor<8xf16> into tensor<4x2xf16>
      %16 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4xf16>
      %17 = scf.forall (%arg3) in (4) shared_outs(%arg4 = %16) -> (tensor<4xf16>) {
        %extracted = tensor.extract %expanded_7[%arg3, %c0] : tensor<4x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_7[%arg3, %c1] : tensor<4x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<4xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<4xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_8 = tensor.expand_shape %17 [[0, 1]] : tensor<4xf16> into tensor<2x2xf16>
      %18 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2xf16>
      %19 = scf.forall (%arg3) in (2) shared_outs(%arg4 = %18) -> (tensor<2xf16>) {
        %extracted = tensor.extract %expanded_8[%arg3, %c0] : tensor<2x2xf16>
        %21 = arith.addf %extracted, %cst : f16
        %extracted_9 = tensor.extract %expanded_8[%arg3, %c1] : tensor<2x2xf16>
        %22 = arith.addf %extracted_9, %21 : f16
        %extracted_slice_10 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<2xf16> to tensor<f16>
        %inserted = tensor.insert %22 into %extracted_slice_10[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f16> into tensor<2xf16>
        }
      } {mapping = [#gpu.thread<x>]}
      %20 = scf.forall (%arg3) in (1) shared_outs(%arg4 = %extracted_slice_0) -> (tensor<f16>) {
        %21 = affine.apply #map4(%arg3)
        %extracted = tensor.extract %19[%21] : tensor<2xf16>
        %22 = arith.addf %extracted, %cst : f16
        %23 = affine.apply #map5(%arg3)
        %extracted_9 = tensor.extract %19[%23] : tensor<2xf16>
        %24 = arith.addf %extracted_9, %22 : f16
        %inserted = tensor.insert %24 into %arg4[] : tensor<f16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[] [] [] : tensor<f16> into tensor<f16>
        }
      } {mapping = [#gpu.thread<x>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %20 into %arg2[%arg1] [1] [1] : tensor<f16> into tensor<4xf16>
      }
    } {mapping = [#gpu.block<x>]}
    return %1 : tensor<4xf16>
  }
  func.func private @Unknown68(%arg0: tensor<4xf16>) -> tensor<4xf16> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4xf16>
    %1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0) -> (tensor<4xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1] [1] [1] : tensor<4xf16> to tensor<f16>
      %2 = tensor.empty() : tensor<f16>
      %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%2 : tensor<f16>) {
      ^bb0(%in: f16, %out: f16):
        %4 = math.log %in : f16
        linalg.yield %4 : f16
      } -> tensor<f16>
      %inserted_slice = tensor.insert_slice %3 into %arg2[%arg1] [1] [1] : tensor<f16> into tensor<4xf16>
      scf.yield %inserted_slice : tensor<4xf16>
    }
    return %1 : tensor<4xf16>
  }
  func.func private @Unknown69(%arg0: tensor<4xf16>, %arg1: tensor<4x1000xf16>, %arg2: tensor<4xf16>, %arg3: tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4x1000xf16>
    %1:2 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %0, %arg6 = %0) -> (tensor<4x1000xf16>, tensor<4x1000xf16>) {
      %2:2 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<4x1000xf16>, tensor<4x1000xf16>) {
        %extracted_slice = tensor.extract_slice %arg2[%arg4] [1] [1] : tensor<4xf16> to tensor<f16>
        %extracted_slice_0 = tensor.extract_slice %arg0[%arg4] [1] [1] : tensor<4xf16> to tensor<f16>
        %extracted_slice_1 = tensor.extract_slice %arg1[%arg4, %arg7] [1, 1] [1, 1] : tensor<4x1000xf16> to tensor<f16>
        %extracted_slice_2 = tensor.extract_slice %arg3[%arg4, %arg7] [1, 1] [1, 1] : tensor<4x1000xf16> to tensor<f16>
        %3 = tensor.empty() : tensor<f16>
        %4:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2 : tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16>) outs(%3, %3 : tensor<f16>, tensor<f16>) {
        ^bb0(%in: f16, %in_4: f16, %in_5: f16, %in_6: f16, %out: f16, %out_7: f16):
          %5 = arith.subf %in_5, %in_4 : f16
          %6 = math.exp %5 : f16
          %7 = arith.mulf %6, %in : f16
          %8 = arith.subf %in_6, %7 : f16
          linalg.yield %5, %8 : f16, f16
        } -> (tensor<f16>, tensor<f16>)
        %inserted_slice = tensor.insert_slice %4#0 into %arg8[%arg4, %arg7] [1, 1] [1, 1] : tensor<f16> into tensor<4x1000xf16>
        %inserted_slice_3 = tensor.insert_slice %4#1 into %arg9[%arg4, %arg7] [1, 1] [1, 1] : tensor<f16> into tensor<4x1000xf16>
        scf.yield %inserted_slice, %inserted_slice_3 : tensor<4x1000xf16>, tensor<4x1000xf16>
      }
      scf.yield %2#0, %2#1 : tensor<4x1000xf16>, tensor<4x1000xf16>
    }
    return %1#0, %1#1 : tensor<4x1000xf16>, tensor<4x1000xf16>
  }
  func.func private @Unknown70(%arg0: tensor<4x512xf16>, %arg1: tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 4.900000e+01 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x512x7x7xf16>) {
      %2 = scf.for %arg4 = %c0 to %c512 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x512x7x7xf16>) {
        %3 = scf.for %arg6 = %c0 to %c7 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x512x7x7xf16>) {
          %4 = scf.for %arg8 = %c0 to %c7 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4x512x7x7xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4] [1, 1] [1, 1] : tensor<4x512xf16> to tensor<f16>
            %extracted_slice_1 = tensor.extract_slice %arg1[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xi1> to tensor<i1>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_1 : tensor<f16>, tensor<i1>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f16, %in_2: i1, %out: f16):
              %7 = arith.divf %in, %cst_0 : f16
              %8 = arith.select %in_2, %7, %cst : f16
              linalg.yield %8 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg9[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x512x7x7xf16>
            scf.yield %inserted_slice : tensor<4x512x7x7xf16>
          }
          scf.yield %4 : tensor<4x512x7x7xf16>
        }
        scf.yield %3 : tensor<4x512x7x7xf16>
      }
      scf.yield %2 : tensor<4x512x7x7xf16>
    }
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown74(%arg0: tensor<4x512x7x7xi1>, %arg1: tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x512x7x7xf16>) {
      %2 = scf.for %arg4 = %c0 to %c512 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x512x7x7xf16>) {
        %3 = scf.for %arg6 = %c0 to %c7 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x512x7x7xf16>) {
          %4 = scf.for %arg8 = %c0 to %c7 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4x512x7x7xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xi1> to tensor<i1>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<i1>, tensor<f16>) outs(%5 : tensor<f16>) {
            ^bb0(%in: i1, %in_1: f16, %out: f16):
              %7 = arith.select %in, %in_1, %cst : f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg9[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x512x7x7xf16>
            scf.yield %inserted_slice : tensor<4x512x7x7xf16>
          }
          scf.yield %4 : tensor<4x512x7x7xf16>
        }
        scf.yield %3 : tensor<4x512x7x7xf16>
      }
      scf.yield %2 : tensor<4x512x7x7xf16>
    }
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown78(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>, %arg2: tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x512x7x7xf16>
    %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %0) -> (tensor<4x512x7x7xf16>) {
      %2 = scf.for %arg5 = %c0 to %c512 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4x512x7x7xf16>) {
        %3 = scf.for %arg7 = %c0 to %c7 step %c1 iter_args(%arg8 = %arg6) -> (tensor<4x512x7x7xf16>) {
          %4 = scf.for %arg9 = %c0 to %c7 step %c1 iter_args(%arg10 = %arg8) -> (tensor<4x512x7x7xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xf16> to tensor<f16>
            %extracted_slice_1 = tensor.extract_slice %arg2[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x512x7x7xi1> to tensor<i1>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<f16>, tensor<f16>, tensor<i1>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f16, %in_2: f16, %in_3: i1, %out: f16):
              %7 = arith.addf %in, %in_2 : f16
              %8 = arith.select %in_3, %7, %cst : f16
              linalg.yield %8 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg10[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x512x7x7xf16>
            scf.yield %inserted_slice : tensor<4x512x7x7xf16>
          }
          scf.yield %4 : tensor<4x512x7x7xf16>
        }
        scf.yield %3 : tensor<4x512x7x7xf16>
      }
      scf.yield %2 : tensor<4x512x7x7xf16>
    }
    return %1 : tensor<4x512x7x7xf16>
  }
  func.func private @Unknown89(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>, %arg2: tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %0) -> (tensor<4x256x14x14xf16>) {
      %2 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4x256x14x14xf16>) {
        %3 = scf.for %arg7 = %c0 to %c14 step %c1 iter_args(%arg8 = %arg6) -> (tensor<4x256x14x14xf16>) {
          %4 = scf.for %arg9 = %c0 to %c14 step %c1 iter_args(%arg10 = %arg8) -> (tensor<4x256x14x14xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x256x14x14xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x256x14x14xf16> to tensor<f16>
            %extracted_slice_1 = tensor.extract_slice %arg2[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x256x14x14xi1> to tensor<i1>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<f16>, tensor<f16>, tensor<i1>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f16, %in_2: f16, %in_3: i1, %out: f16):
              %7 = arith.addf %in, %in_2 : f16
              %8 = arith.select %in_3, %7, %cst : f16
              linalg.yield %8 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg10[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x256x14x14xf16>
            scf.yield %inserted_slice : tensor<4x256x14x14xf16>
          }
          scf.yield %4 : tensor<4x256x14x14xf16>
        }
        scf.yield %3 : tensor<4x256x14x14xf16>
      }
      scf.yield %2 : tensor<4x256x14x14xf16>
    }
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown93(%arg0: tensor<4x256x14x14xi1>, %arg1: tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x256x14x14xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x256x14x14xf16>) {
      %2 = scf.for %arg4 = %c0 to %c256 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x256x14x14xf16>) {
        %3 = scf.for %arg6 = %c0 to %c14 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x256x14x14xf16>) {
          %4 = scf.for %arg8 = %c0 to %c14 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4x256x14x14xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x256x14x14xi1> to tensor<i1>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x256x14x14xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<i1>, tensor<f16>) outs(%5 : tensor<f16>) {
            ^bb0(%in: i1, %in_1: f16, %out: f16):
              %7 = arith.select %in, %in_1, %cst : f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg9[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x256x14x14xf16>
            scf.yield %inserted_slice : tensor<4x256x14x14xf16>
          }
          scf.yield %4 : tensor<4x256x14x14xf16>
        }
        scf.yield %3 : tensor<4x256x14x14xf16>
      }
      scf.yield %2 : tensor<4x256x14x14xf16>
    }
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @Unknown108(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>, %arg2: tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %0) -> (tensor<4x128x28x28xf16>) {
      %2 = scf.for %arg5 = %c0 to %c128 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4x128x28x28xf16>) {
        %3 = scf.for %arg7 = %c0 to %c28 step %c1 iter_args(%arg8 = %arg6) -> (tensor<4x128x28x28xf16>) {
          %4 = scf.for %arg9 = %c0 to %c28 step %c1 iter_args(%arg10 = %arg8) -> (tensor<4x128x28x28xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x128x28x28xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x128x28x28xf16> to tensor<f16>
            %extracted_slice_1 = tensor.extract_slice %arg2[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x128x28x28xi1> to tensor<i1>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<f16>, tensor<f16>, tensor<i1>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f16, %in_2: f16, %in_3: i1, %out: f16):
              %7 = arith.addf %in, %in_2 : f16
              %8 = arith.select %in_3, %7, %cst : f16
              linalg.yield %8 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg10[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x128x28x28xf16>
            scf.yield %inserted_slice : tensor<4x128x28x28xf16>
          }
          scf.yield %4 : tensor<4x128x28x28xf16>
        }
        scf.yield %3 : tensor<4x128x28x28xf16>
      }
      scf.yield %2 : tensor<4x128x28x28xf16>
    }
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown112(%arg0: tensor<4x128x28x28xi1>, %arg1: tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x128x28x28xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x128x28x28xf16>) {
      %2 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x128x28x28xf16>) {
        %3 = scf.for %arg6 = %c0 to %c28 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x128x28x28xf16>) {
          %4 = scf.for %arg8 = %c0 to %c28 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4x128x28x28xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x128x28x28xi1> to tensor<i1>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x128x28x28xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<i1>, tensor<f16>) outs(%5 : tensor<f16>) {
            ^bb0(%in: i1, %in_1: f16, %out: f16):
              %7 = arith.select %in, %in_1, %cst : f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg9[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x128x28x28xf16>
            scf.yield %inserted_slice : tensor<4x128x28x28xf16>
          }
          scf.yield %4 : tensor<4x128x28x28xf16>
        }
        scf.yield %3 : tensor<4x128x28x28xf16>
      }
      scf.yield %2 : tensor<4x128x28x28xf16>
    }
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @Unknown127(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>, %arg2: tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %0) -> (tensor<4x64x56x56xf16>) {
      %2 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4x64x56x56xf16>) {
        %3 = scf.for %arg7 = %c0 to %c56 step %c1 iter_args(%arg8 = %arg6) -> (tensor<4x64x56x56xf16>) {
          %4 = scf.for %arg9 = %c0 to %c56 step %c1 iter_args(%arg10 = %arg8) -> (tensor<4x64x56x56xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xf16> to tensor<f16>
            %extracted_slice_1 = tensor.extract_slice %arg2[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xi1> to tensor<i1>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<f16>, tensor<f16>, tensor<i1>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f16, %in_2: f16, %in_3: i1, %out: f16):
              %7 = arith.addf %in, %in_2 : f16
              %8 = arith.select %in_3, %7, %cst : f16
              linalg.yield %8 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg10[%arg3, %arg5, %arg7, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x64x56x56xf16>
            scf.yield %inserted_slice : tensor<4x64x56x56xf16>
          }
          scf.yield %4 : tensor<4x64x56x56xf16>
        }
        scf.yield %3 : tensor<4x64x56x56xf16>
      }
      scf.yield %2 : tensor<4x64x56x56xf16>
    }
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown131(%arg0: tensor<4x64x56x56xi1>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x64x56x56xf16>) {
      %2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x64x56x56xf16>) {
        %3 = scf.for %arg6 = %c0 to %c56 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x64x56x56xf16>) {
          %4 = scf.for %arg8 = %c0 to %c56 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4x64x56x56xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xi1> to tensor<i1>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<i1>, tensor<f16>) outs(%5 : tensor<f16>) {
            ^bb0(%in: i1, %in_1: f16, %out: f16):
              %7 = arith.select %in, %in_1, %cst : f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg9[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x64x56x56xf16>
            scf.yield %inserted_slice : tensor<4x64x56x56xf16>
          }
          scf.yield %4 : tensor<4x64x56x56xf16>
        }
        scf.yield %3 : tensor<4x64x56x56xf16>
      }
      scf.yield %2 : tensor<4x64x56x56xf16>
    }
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown143(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4x64x56x56xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x64x56x56xf16>) {
      %2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x64x56x56xf16>) {
        %3 = scf.for %arg6 = %c0 to %c56 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x64x56x56xf16>) {
          %4 = scf.for %arg8 = %c0 to %c56 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4x64x56x56xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xf16> to tensor<f16>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x56x56xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%5 : tensor<f16>) {
            ^bb0(%in: f16, %in_1: f16, %out: f16):
              %7 = arith.addf %in, %in_1 : f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg9[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x64x56x56xf16>
            scf.yield %inserted_slice : tensor<4x64x56x56xf16>
          }
          scf.yield %4 : tensor<4x64x56x56xf16>
        }
        scf.yield %3 : tensor<4x64x56x56xf16>
      }
      scf.yield %2 : tensor<4x64x56x56xf16>
    }
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @Unknown144(%arg0: tensor<4x64x112x112xi1>, %arg1: tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %c112 = arith.constant 112 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4x64x112x112xf16>
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x64x112x112xf16>) {
      %2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x64x112x112xf16>) {
        %3 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x64x112x112xf16>) {
          %4 = scf.for %arg8 = %c0 to %c112 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4x64x112x112xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x112x112xi1> to tensor<i1>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<4x64x112x112xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f16>
            %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<i1>, tensor<f16>) outs(%5 : tensor<f16>) {
            ^bb0(%in: i1, %in_1: f16, %out: f16):
              %7 = arith.select %in, %in_1, %cst : f16
              linalg.yield %7 : f16
            } -> tensor<f16>
            %inserted_slice = tensor.insert_slice %6 into %arg9[%arg2, %arg4, %arg6, %arg8] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<4x64x112x112xf16>
            scf.yield %inserted_slice : tensor<4x64x112x112xf16>
          }
          scf.yield %4 : tensor<4x64x112x112xf16>
        }
        scf.yield %3 : tensor<4x64x112x112xf16>
      }
      scf.yield %2 : tensor<4x64x112x112xf16>
    }
    return %1 : tensor<4x64x112x112xf16>
  }
  func.func private @Unknown147(%arg0: tensor<4x1000xf16>, %arg1: tensor<4x1000xf32>) -> tensor<f32> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<4x1000xf16> into tensor<4000xf16>
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<4x1000xf32> into tensor<4000xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] : tensor<4000xf16> into tensor<32x125xf16>
    %expanded_2 = tensor.expand_shape %collapsed_1 [[0, 1]] : tensor<4000xf32> into tensor<32x125xf32>
    %1 = tensor.empty() : tensor<32xf32>
    %2 = scf.forall (%arg2) in (32) shared_outs(%arg3 = %1) -> (tensor<32xf32>) {
      %extracted_slice = tensor.extract_slice %expanded[%arg2, 0] [1, 125] [1, 1] : tensor<32x125xf16> to tensor<125xf16>
      %expanded_3 = tensor.expand_shape %extracted_slice [[0, 1]] : tensor<125xf16> into tensor<1x125xf16>
      %extracted_slice_4 = tensor.extract_slice %expanded_2[%arg2, 0] [1, 125] [1, 1] : tensor<32x125xf32> to tensor<125xf32>
      %expanded_5 = tensor.expand_shape %extracted_slice_4 [[0, 1]] : tensor<125xf32> into tensor<1x125xf32>
      %extracted_slice_6 = tensor.extract_slice %arg3[%arg2] [1] [1] : tensor<32xf32> to tensor<f32>
      %4 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128xf32>
      %5 = scf.forall (%arg4) in (128) shared_outs(%arg5 = %4) -> (tensor<128xf32>) {
        %19 = affine.min #map8(%arg4)
        %20 = affine.min #map9(%arg4)
        %21 = affine.apply #map3(%20, %19)
        %extracted_slice_13 = tensor.extract_slice %expanded_3[0, %19] [1, %21] [1, 1] : tensor<1x125xf16> to tensor<?xf16>
        %expanded_14 = tensor.expand_shape %extracted_slice_13 [[0, 1]] : tensor<?xf16> into tensor<1x?xf16>
        %extracted_slice_15 = tensor.extract_slice %expanded_5[0, %19] [1, %21] [1, 1] : tensor<1x125xf32> to tensor<?xf32>
        %expanded_16 = tensor.expand_shape %extracted_slice_15 [[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
        %dim = tensor.dim %extracted_slice_13, %c0 : tensor<?xf16>
        %22 = arith.cmpi ugt, %dim, %c0 : index
        %23 = scf.if %22 -> (f16) {
          %extracted = tensor.extract %expanded_14[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %dim_17 = tensor.dim %extracted_slice_15, %c0 : tensor<?xf32>
        %24 = arith.cmpi ugt, %dim_17, %c0 : index
        %25 = scf.if %24 -> (f32) {
          %extracted = tensor.extract %expanded_16[%c0, %c0] : tensor<1x?xf32>
          scf.yield %extracted : f32
        } else {
          scf.yield %cst_0 : f32
        }
        %26 = arith.extf %23 : f16 to f32
        %27 = arith.mulf %26, %25 : f32
        %28 = arith.addf %27, %cst_0 : f32
        %extracted_slice_18 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<128xf32> to tensor<f32>
        %inserted = tensor.insert %28 into %extracted_slice_18[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<128xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_7 = tensor.expand_shape %5 [[0, 1]] : tensor<128xf32> into tensor<64x2xf32>
      %6 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<64xf32>
      %7 = scf.forall (%arg4) in (64) shared_outs(%arg5 = %6) -> (tensor<64xf32>) {
        %extracted = tensor.extract %expanded_7[%arg4, %c0] : tensor<64x2xf32>
        %19 = arith.addf %extracted, %cst_0 : f32
        %extracted_13 = tensor.extract %expanded_7[%arg4, %c1] : tensor<64x2xf32>
        %20 = arith.addf %extracted_13, %19 : f32
        %extracted_slice_14 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<64xf32> to tensor<f32>
        %inserted = tensor.insert %20 into %extracted_slice_14[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<64xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_8 = tensor.expand_shape %7 [[0, 1]] : tensor<64xf32> into tensor<32x2xf32>
      %8 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf32>
      %9 = scf.forall (%arg4) in (32) shared_outs(%arg5 = %8) -> (tensor<32xf32>) {
        %extracted = tensor.extract %expanded_8[%arg4, %c0] : tensor<32x2xf32>
        %19 = arith.addf %extracted, %cst_0 : f32
        %extracted_13 = tensor.extract %expanded_8[%arg4, %c1] : tensor<32x2xf32>
        %20 = arith.addf %extracted_13, %19 : f32
        %extracted_slice_14 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<32xf32> to tensor<f32>
        %inserted = tensor.insert %20 into %extracted_slice_14[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<32xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_9 = tensor.expand_shape %9 [[0, 1]] : tensor<32xf32> into tensor<16x2xf32>
      %10 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16xf32>
      %11 = scf.forall (%arg4) in (16) shared_outs(%arg5 = %10) -> (tensor<16xf32>) {
        %extracted = tensor.extract %expanded_9[%arg4, %c0] : tensor<16x2xf32>
        %19 = arith.addf %extracted, %cst_0 : f32
        %extracted_13 = tensor.extract %expanded_9[%arg4, %c1] : tensor<16x2xf32>
        %20 = arith.addf %extracted_13, %19 : f32
        %extracted_slice_14 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<16xf32> to tensor<f32>
        %inserted = tensor.insert %20 into %extracted_slice_14[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<16xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_10 = tensor.expand_shape %11 [[0, 1]] : tensor<16xf32> into tensor<8x2xf32>
      %12 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<8xf32>
      %13 = scf.forall (%arg4) in (8) shared_outs(%arg5 = %12) -> (tensor<8xf32>) {
        %extracted = tensor.extract %expanded_10[%arg4, %c0] : tensor<8x2xf32>
        %19 = arith.addf %extracted, %cst_0 : f32
        %extracted_13 = tensor.extract %expanded_10[%arg4, %c1] : tensor<8x2xf32>
        %20 = arith.addf %extracted_13, %19 : f32
        %extracted_slice_14 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<8xf32> to tensor<f32>
        %inserted = tensor.insert %20 into %extracted_slice_14[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<8xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_11 = tensor.expand_shape %13 [[0, 1]] : tensor<8xf32> into tensor<4x2xf32>
      %14 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4xf32>
      %15 = scf.forall (%arg4) in (4) shared_outs(%arg5 = %14) -> (tensor<4xf32>) {
        %extracted = tensor.extract %expanded_11[%arg4, %c0] : tensor<4x2xf32>
        %19 = arith.addf %extracted, %cst_0 : f32
        %extracted_13 = tensor.extract %expanded_11[%arg4, %c1] : tensor<4x2xf32>
        %20 = arith.addf %extracted_13, %19 : f32
        %extracted_slice_14 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<4xf32> to tensor<f32>
        %inserted = tensor.insert %20 into %extracted_slice_14[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<4xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_12 = tensor.expand_shape %15 [[0, 1]] : tensor<4xf32> into tensor<2x2xf32>
      %16 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2xf32>
      %17 = scf.forall (%arg4) in (2) shared_outs(%arg5 = %16) -> (tensor<2xf32>) {
        %extracted = tensor.extract %expanded_12[%arg4, %c0] : tensor<2x2xf32>
        %19 = arith.addf %extracted, %cst_0 : f32
        %extracted_13 = tensor.extract %expanded_12[%arg4, %c1] : tensor<2x2xf32>
        %20 = arith.addf %extracted_13, %19 : f32
        %extracted_slice_14 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<2xf32> to tensor<f32>
        %inserted = tensor.insert %20 into %extracted_slice_14[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<2xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %18 = scf.forall (%arg4) in (1) shared_outs(%arg5 = %extracted_slice_6) -> (tensor<f32>) {
        %19 = affine.apply #map4(%arg4)
        %extracted = tensor.extract %17[%19] : tensor<2xf32>
        %20 = arith.addf %extracted, %cst_0 : f32
        %21 = affine.apply #map5(%arg4)
        %extracted_13 = tensor.extract %17[%21] : tensor<2xf32>
        %22 = arith.addf %extracted_13, %20 : f32
        %inserted = tensor.insert %22 into %arg5[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[] [] [] : tensor<f32> into tensor<f32>
        }
      } {mapping = [#gpu.thread<x>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %18 into %arg3[%arg2] [1] [1] : tensor<f32> into tensor<32xf32>
      }
    } {mapping = [#gpu.block<x>]}
    %3 = scf.forall (%arg2) in (1) shared_outs(%arg3 = %0) -> (tensor<f32>) {
      %4 = affine.apply #map10(%arg2)
      %extracted_slice = tensor.extract_slice %2[%4] [32] [1] : tensor<32xf32> to tensor<32xf32>
      %expanded_3 = tensor.expand_shape %extracted_slice [[0, 1]] : tensor<32xf32> into tensor<32x1xf32>
      %5 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf32>
      %6 = scf.forall (%arg4) in (32) shared_outs(%arg5 = %5) -> (tensor<32xf32>) {
        %extracted = tensor.extract %expanded_3[%arg4, %c0] : tensor<32x1xf32>
        %16 = arith.addf %extracted, %cst_0 : f32
        %extracted_slice_8 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<32xf32> to tensor<f32>
        %inserted = tensor.insert %16 into %extracted_slice_8[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<32xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_4 = tensor.expand_shape %6 [[0, 1]] : tensor<32xf32> into tensor<16x2xf32>
      %7 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16xf32>
      %8 = scf.forall (%arg4) in (16) shared_outs(%arg5 = %7) -> (tensor<16xf32>) {
        %extracted = tensor.extract %expanded_4[%arg4, %c0] : tensor<16x2xf32>
        %16 = arith.addf %extracted, %cst_0 : f32
        %extracted_8 = tensor.extract %expanded_4[%arg4, %c1] : tensor<16x2xf32>
        %17 = arith.addf %extracted_8, %16 : f32
        %extracted_slice_9 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<16xf32> to tensor<f32>
        %inserted = tensor.insert %17 into %extracted_slice_9[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<16xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_5 = tensor.expand_shape %8 [[0, 1]] : tensor<16xf32> into tensor<8x2xf32>
      %9 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<8xf32>
      %10 = scf.forall (%arg4) in (8) shared_outs(%arg5 = %9) -> (tensor<8xf32>) {
        %extracted = tensor.extract %expanded_5[%arg4, %c0] : tensor<8x2xf32>
        %16 = arith.addf %extracted, %cst_0 : f32
        %extracted_8 = tensor.extract %expanded_5[%arg4, %c1] : tensor<8x2xf32>
        %17 = arith.addf %extracted_8, %16 : f32
        %extracted_slice_9 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<8xf32> to tensor<f32>
        %inserted = tensor.insert %17 into %extracted_slice_9[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<8xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_6 = tensor.expand_shape %10 [[0, 1]] : tensor<8xf32> into tensor<4x2xf32>
      %11 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4xf32>
      %12 = scf.forall (%arg4) in (4) shared_outs(%arg5 = %11) -> (tensor<4xf32>) {
        %extracted = tensor.extract %expanded_6[%arg4, %c0] : tensor<4x2xf32>
        %16 = arith.addf %extracted, %cst_0 : f32
        %extracted_8 = tensor.extract %expanded_6[%arg4, %c1] : tensor<4x2xf32>
        %17 = arith.addf %extracted_8, %16 : f32
        %extracted_slice_9 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<4xf32> to tensor<f32>
        %inserted = tensor.insert %17 into %extracted_slice_9[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<4xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %expanded_7 = tensor.expand_shape %12 [[0, 1]] : tensor<4xf32> into tensor<2x2xf32>
      %13 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2xf32>
      %14 = scf.forall (%arg4) in (2) shared_outs(%arg5 = %13) -> (tensor<2xf32>) {
        %extracted = tensor.extract %expanded_7[%arg4, %c0] : tensor<2x2xf32>
        %16 = arith.addf %extracted, %cst_0 : f32
        %extracted_8 = tensor.extract %expanded_7[%arg4, %c1] : tensor<2x2xf32>
        %17 = arith.addf %extracted_8, %16 : f32
        %extracted_slice_9 = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<2xf32> to tensor<f32>
        %inserted = tensor.insert %17 into %extracted_slice_9[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg4] [1] [1] : tensor<f32> into tensor<2xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %15 = scf.forall (%arg4) in (1) shared_outs(%arg5 = %arg3) -> (tensor<f32>) {
        %16 = affine.apply #map4(%arg4)
        %extracted = tensor.extract %14[%16] : tensor<2xf32>
        %17 = arith.addf %extracted, %cst_0 : f32
        %18 = affine.apply #map5(%arg4)
        %extracted_8 = tensor.extract %14[%18] : tensor<2xf32>
        %19 = arith.addf %extracted_8, %17 : f32
        %inserted = tensor.insert %19 into %arg5[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[] [] [] : tensor<f32> into tensor<f32>
        }
      } {mapping = [#gpu.thread<x>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %15 into %arg3[] [] [] : tensor<f32> into tensor<f32>
      }
    } {mapping = [#gpu.block<x>]}
    return %3 : tensor<f32>
  }
  func.func private @Unknown148(%arg0: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.negf %in : f32
      %3 = arith.divf %2, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @Unknown149(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<64x3x7x7xf32>
    %1 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %0) -> (tensor<64x3x7x7xf32>) {
      %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2) -> (tensor<64x3x7x7xf32>) {
        %3 = scf.for %arg5 = %c0 to %c7 step %c1 iter_args(%arg6 = %arg4) -> (tensor<64x3x7x7xf32>) {
          %4 = scf.for %arg7 = %c0 to %c7 step %c1 iter_args(%arg8 = %arg6) -> (tensor<64x3x7x7xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<64x3x7x7xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f32>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%5 : tensor<f32>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            } -> tensor<f32>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<64x3x7x7xf32>
            scf.yield %inserted_slice : tensor<64x3x7x7xf32>
          }
          scf.yield %4 : tensor<64x3x7x7xf32>
        }
        scf.yield %3 : tensor<64x3x7x7xf32>
      }
      scf.yield %2 : tensor<64x3x7x7xf32>
    }
    return %1 : tensor<64x3x7x7xf32>
  }
  func.func private @Unknown150(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %0) -> (tensor<64x64x3x3xf32>) {
      %2 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<64x64x3x3xf32>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<64x64x3x3xf32>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<64x64x3x3xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<64x64x3x3xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f32>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%5 : tensor<f32>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            } -> tensor<f32>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<64x64x3x3xf32>
            scf.yield %inserted_slice : tensor<64x64x3x3xf32>
          }
          scf.yield %4 : tensor<64x64x3x3xf32>
        }
        scf.yield %3 : tensor<64x64x3x3xf32>
      }
      scf.yield %2 : tensor<64x64x3x3xf32>
    }
    return %1 : tensor<64x64x3x3xf32>
  }
  func.func private @Unknown154(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<128x64x3x3xf32>
    %1 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %0) -> (tensor<128x64x3x3xf32>) {
      %2 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<128x64x3x3xf32>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<128x64x3x3xf32>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<128x64x3x3xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<128x64x3x3xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f32>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%5 : tensor<f32>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            } -> tensor<f32>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<128x64x3x3xf32>
            scf.yield %inserted_slice : tensor<128x64x3x3xf32>
          }
          scf.yield %4 : tensor<128x64x3x3xf32>
        }
        scf.yield %3 : tensor<128x64x3x3xf32>
      }
      scf.yield %2 : tensor<128x64x3x3xf32>
    }
    return %1 : tensor<128x64x3x3xf32>
  }
  func.func private @Unknown155(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<128x128x3x3xf32>
    %1 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %0) -> (tensor<128x128x3x3xf32>) {
      %2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %arg2) -> (tensor<128x128x3x3xf32>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<128x128x3x3xf32>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<128x128x3x3xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<128x128x3x3xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f32>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%5 : tensor<f32>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            } -> tensor<f32>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<128x128x3x3xf32>
            scf.yield %inserted_slice : tensor<128x128x3x3xf32>
          }
          scf.yield %4 : tensor<128x128x3x3xf32>
        }
        scf.yield %3 : tensor<128x128x3x3xf32>
      }
      scf.yield %2 : tensor<128x128x3x3xf32>
    }
    return %1 : tensor<128x128x3x3xf32>
  }
  func.func private @Unknown156(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<128x64x1x1xf32>
    %1 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %0) -> (tensor<128x64x1x1xf32>) {
      %2 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<128x64x1x1xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<128x64x1x1xf16> to tensor<f16>
        %3 = tensor.empty() : tensor<f32>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%3 : tensor<f32>) {
        ^bb0(%in: f16, %out: f32):
          %5 = arith.extf %in : f16 to f32
          linalg.yield %5 : f32
        } -> tensor<f32>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<128x64x1x1xf32>
        scf.yield %inserted_slice : tensor<128x64x1x1xf32>
      }
      scf.yield %2 : tensor<128x64x1x1xf32>
    }
    return %1 : tensor<128x64x1x1xf32>
  }
  func.func private @Unknown159(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<256x128x3x3xf32>
    %1 = scf.for %arg1 = %c0 to %c256 step %c1 iter_args(%arg2 = %0) -> (tensor<256x128x3x3xf32>) {
      %2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %arg2) -> (tensor<256x128x3x3xf32>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<256x128x3x3xf32>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<256x128x3x3xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<256x128x3x3xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f32>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%5 : tensor<f32>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            } -> tensor<f32>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<256x128x3x3xf32>
            scf.yield %inserted_slice : tensor<256x128x3x3xf32>
          }
          scf.yield %4 : tensor<256x128x3x3xf32>
        }
        scf.yield %3 : tensor<256x128x3x3xf32>
      }
      scf.yield %2 : tensor<256x128x3x3xf32>
    }
    return %1 : tensor<256x128x3x3xf32>
  }
  func.func private @Unknown160(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<256x256x3x3xf32>
    %1 = scf.for %arg1 = %c0 to %c256 step %c1 iter_args(%arg2 = %0) -> (tensor<256x256x3x3xf32>) {
      %2 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %arg2) -> (tensor<256x256x3x3xf32>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<256x256x3x3xf32>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<256x256x3x3xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<256x256x3x3xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f32>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%5 : tensor<f32>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            } -> tensor<f32>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<256x256x3x3xf32>
            scf.yield %inserted_slice : tensor<256x256x3x3xf32>
          }
          scf.yield %4 : tensor<256x256x3x3xf32>
        }
        scf.yield %3 : tensor<256x256x3x3xf32>
      }
      scf.yield %2 : tensor<256x256x3x3xf32>
    }
    return %1 : tensor<256x256x3x3xf32>
  }
  func.func private @Unknown161(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<256x128x1x1xf32>
    %1 = scf.for %arg1 = %c0 to %c256 step %c1 iter_args(%arg2 = %0) -> (tensor<256x128x1x1xf32>) {
      %2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %arg2) -> (tensor<256x128x1x1xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<256x128x1x1xf16> to tensor<f16>
        %3 = tensor.empty() : tensor<f32>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%3 : tensor<f32>) {
        ^bb0(%in: f16, %out: f32):
          %5 = arith.extf %in : f16 to f32
          linalg.yield %5 : f32
        } -> tensor<f32>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<256x128x1x1xf32>
        scf.yield %inserted_slice : tensor<256x128x1x1xf32>
      }
      scf.yield %2 : tensor<256x128x1x1xf32>
    }
    return %1 : tensor<256x128x1x1xf32>
  }
  func.func private @Unknown164(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<512x256x3x3xf32>
    %1 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %0) -> (tensor<512x256x3x3xf32>) {
      %2 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %arg2) -> (tensor<512x256x3x3xf32>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<512x256x3x3xf32>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<512x256x3x3xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<512x256x3x3xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f32>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%5 : tensor<f32>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            } -> tensor<f32>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<512x256x3x3xf32>
            scf.yield %inserted_slice : tensor<512x256x3x3xf32>
          }
          scf.yield %4 : tensor<512x256x3x3xf32>
        }
        scf.yield %3 : tensor<512x256x3x3xf32>
      }
      scf.yield %2 : tensor<512x256x3x3xf32>
    }
    return %1 : tensor<512x256x3x3xf32>
  }
  func.func private @Unknown165(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<512x512x3x3xf32>
    %1 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %0) -> (tensor<512x512x3x3xf32>) {
      %2 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %arg2) -> (tensor<512x512x3x3xf32>) {
        %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<512x512x3x3xf32>) {
          %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<512x512x3x3xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<512x512x3x3xf16> to tensor<f16>
            %5 = tensor.empty() : tensor<f32>
            %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%5 : tensor<f32>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            } -> tensor<f32>
            %inserted_slice = tensor.insert_slice %6 into %arg8[%arg1, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<512x512x3x3xf32>
            scf.yield %inserted_slice : tensor<512x512x3x3xf32>
          }
          scf.yield %4 : tensor<512x512x3x3xf32>
        }
        scf.yield %3 : tensor<512x512x3x3xf32>
      }
      scf.yield %2 : tensor<512x512x3x3xf32>
    }
    return %1 : tensor<512x512x3x3xf32>
  }
  func.func private @Unknown166(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<512x256x1x1xf32>
    %1 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %0) -> (tensor<512x256x1x1xf32>) {
      %2 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %arg2) -> (tensor<512x256x1x1xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<512x256x1x1xf16> to tensor<f16>
        %3 = tensor.empty() : tensor<f32>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%3 : tensor<f32>) {
        ^bb0(%in: f16, %out: f32):
          %5 = arith.extf %in : f16 to f32
          linalg.yield %5 : f32
        } -> tensor<f32>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f32> into tensor<512x256x1x1xf32>
        scf.yield %inserted_slice : tensor<512x256x1x1xf32>
      }
      scf.yield %2 : tensor<512x256x1x1xf32>
    }
    return %1 : tensor<512x256x1x1xf32>
  }
  func.func private @Unknown170(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1000x512xf32>
    %1 = scf.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %0) -> (tensor<1000x512xf32>) {
      %2 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1000x512xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3] [1, 1] [1, 1] : tensor<1000x512xf16> to tensor<f16>
        %3 = tensor.empty() : tensor<f32>
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%3 : tensor<f32>) {
        ^bb0(%in: f16, %out: f32):
          %5 = arith.extf %in : f16 to f32
          linalg.yield %5 : f32
        } -> tensor<f32>
        %inserted_slice = tensor.insert_slice %4 into %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<f32> into tensor<1000x512xf32>
        scf.yield %inserted_slice : tensor<1000x512xf32>
      }
      scf.yield %2 : tensor<1000x512xf32>
    }
    return %1 : tensor<1000x512xf32>
  }
  func.func private @Unknown171(%arg0: tensor<4x1000xf16>) -> tensor<1000xf32> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1000xf32>
    %1 = scf.forall (%arg1) in (32) shared_outs(%arg2 = %0) -> (tensor<1000xf32>) {
      %2 = affine.min #map11(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf32>
      %5 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2x32xf32>
      %6 = scf.forall (%arg3, %arg4) in (2, 32) shared_outs(%arg5 = %5) -> (tensor<2x32xf32>) {
        %8 = affine.min #map12(%arg4, %arg1)
        %9 = affine.min #map13(%arg4, %arg1)
        %10 = affine.apply #map3(%9, %8)
        %11 = arith.cmpi ugt, %10, %c0 : index
        %12 = scf.if %11 -> (f16) {
          %19 = affine.apply #map4(%arg3)
          %20 = affine.apply #map14(%arg1)[%8]
          %extracted = tensor.extract %arg0[%19, %20] : tensor<4x1000xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %13 = arith.extf %12 : f16 to f32
        %14 = arith.addf %13, %cst_0 : f32
        %15 = arith.cmpi ugt, %10, %c0 : index
        %16 = scf.if %15 -> (f16) {
          %19 = affine.apply #map5(%arg3)
          %20 = affine.apply #map14(%arg1)[%8]
          %extracted = tensor.extract %arg0[%19, %20] : tensor<4x1000xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %17 = arith.extf %16 : f16 to f32
        %18 = arith.addf %14, %17 : f32
        %extracted_slice_1 = tensor.extract_slice %arg5[%arg3, %arg4] [1, 1] [1, 1] : tensor<2x32xf32> to tensor<f32>
        %inserted = tensor.insert %18 into %extracted_slice_1[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg5[%arg3, %arg4] [1, 1] [1, 1] : tensor<f32> into tensor<2x32xf32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %7 = scf.forall (%arg3) in (32) shared_outs(%arg4 = %4) -> (tensor<32xf32>) {
        %extracted = tensor.extract %6[%c0, %arg3] : tensor<2x32xf32>
        %8 = arith.addf %extracted, %cst_0 : f32
        %extracted_1 = tensor.extract %6[%c1, %arg3] : tensor<2x32xf32>
        %9 = arith.addf %extracted_1, %8 : f32
        %extracted_slice_2 = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<32xf32> to tensor<f32>
        %inserted = tensor.insert %9 into %extracted_slice_2[] : tensor<f32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %inserted into %arg4[%arg3] [1] [1] : tensor<f32> into tensor<32xf32>
        }
      } {mapping = [#gpu.thread<x>]}
      %extracted_slice = tensor.extract_slice %7[0] [%2] [1] : tensor<32xf32> to tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %extracted_slice into %arg2[%3] [%2] [1] : tensor<?xf32> into tensor<1000xf32>
      }
    } {mapping = [#gpu.block<x>]}
    return %1 : tensor<1000xf32>
  }
  func.func private @Unknown172(%arg0: tensor<1000xf32>) -> tensor<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1000xf32>
    %1 = scf.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %0) -> (tensor<1000xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1] [1] [1] : tensor<1000xf32> to tensor<f32>
      %2 = tensor.empty() : tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = arith.truncf %in : f32 to f16
        %5 = arith.extf %4 : f16 to f32
        linalg.yield %5 : f32
      } -> tensor<f32>
      %inserted_slice = tensor.insert_slice %3 into %arg2[%arg1] [1] [1] : tensor<f32> into tensor<1000xf32>
      scf.yield %inserted_slice : tensor<1000xf32>
    }
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
    %7 = call @Unknown3(%arg12) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %8 = call @Unknown3(%arg17) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %9 = call @Unknown3(%arg22) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %10 = call @Unknown7(%arg37) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %11 = call @Unknown8(%arg27) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %12 = call @Unknown9(%arg32) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %13 = call @Unknown9(%arg42) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %14 = call @Unknown9(%arg47) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %15 = call @Unknown12(%arg62) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %16 = call @Unknown13(%arg52) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %17 = call @Unknown14(%arg57) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %18 = call @Unknown14(%arg67) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %19 = call @Unknown14(%arg72) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %20 = call @Unknown17(%arg87) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %21 = call @Unknown18(%arg77) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %22 = call @Unknown19(%arg82) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %23 = call @Unknown19(%arg92) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %24 = call @Unknown19(%arg97) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %25 = call @Unknown22(%arg1) : (tensor<4x1000xf32>) -> tensor<4x1000xf16>
    %26 = call @Unknown23(%arg102) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %27 = call @Unknown24(%arg103) : (tensor<1000xf32>) -> tensor<1000xf16>
    %28 = call @Unknown25(%25) : (tensor<4x1000xf16>) -> tensor<4xf16>
    %29:2 = call @Unknown26(%5) : (tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<4x64x112x112xi1>)
    %30 = tensor.empty() : tensor<4x64x56x56xf16>
    %31 = byre.compute_on_tensor @PoolMaxOp_f16_f16 {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} ins(%29#0 : tensor<4x64x112x112xf16>) outs(%30 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %32 = tensor.empty() : tensor<4x64x56x56xf16>
    %33 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%31, %6 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%32 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %34 = tensor.empty() : tensor<4x64x56x56xf16>
    %35 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%33, %arg8, %arg9 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%34 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %36:2 = call @Unknown28(%35) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %37 = tensor.empty() : tensor<4x64x56x56xf16>
    %38 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%36#0, %7 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%37 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %39 = tensor.empty() : tensor<4x64x56x56xf16>
    %40 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%38, %arg13, %arg14 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%39 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %41:2 = call @Unknown30(%40, %31) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %42 = tensor.empty() : tensor<4x64x56x56xf16>
    %43 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%41#0, %8 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%42 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %44 = tensor.empty() : tensor<4x64x56x56xf16>
    %45 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%43, %arg18, %arg19 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%44 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %46:2 = call @Unknown28(%45) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %47 = tensor.empty() : tensor<4x64x56x56xf16>
    %48 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%46#0, %9 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%47 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %49 = tensor.empty() : tensor<4x64x56x56xf16>
    %50 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%48, %arg23, %arg24 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%49 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %51:2 = call @Unknown30(%50, %41#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %52 = tensor.empty() : tensor<4x128x28x28xf16>
    %53 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%51#0, %10 : tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) outs(%52 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %54 = tensor.empty() : tensor<4x128x28x28xf16>
    %55 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%53, %arg38, %arg39 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%54 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %56 = tensor.empty() : tensor<4x128x28x28xf16>
    %57 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%51#0, %11 : tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) outs(%56 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %58 = tensor.empty() : tensor<4x128x28x28xf16>
    %59 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%57, %arg28, %arg29 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%58 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %60:2 = call @Unknown37(%59) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %61 = tensor.empty() : tensor<4x128x28x28xf16>
    %62 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%60#0, %12 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%61 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %63 = tensor.empty() : tensor<4x128x28x28xf16>
    %64 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%62, %arg33, %arg34 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%63 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %65:2 = call @Unknown39(%64, %55) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %66 = tensor.empty() : tensor<4x128x28x28xf16>
    %67 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%65#0, %13 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%66 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %68 = tensor.empty() : tensor<4x128x28x28xf16>
    %69 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%67, %arg43, %arg44 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%68 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %70:2 = call @Unknown37(%69) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %71 = tensor.empty() : tensor<4x128x28x28xf16>
    %72 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%70#0, %14 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%71 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %73 = tensor.empty() : tensor<4x128x28x28xf16>
    %74 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%72, %arg48, %arg49 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%73 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %75:2 = call @Unknown39(%74, %65#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %76 = tensor.empty() : tensor<4x256x14x14xf16>
    %77 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%75#0, %15 : tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) outs(%76 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %78 = tensor.empty() : tensor<4x256x14x14xf16>
    %79 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%77, %arg63, %arg64 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%78 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %80 = tensor.empty() : tensor<4x256x14x14xf16>
    %81 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%75#0, %16 : tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) outs(%80 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %82 = tensor.empty() : tensor<4x256x14x14xf16>
    %83 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%81, %arg53, %arg54 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%82 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %84:2 = call @Unknown46(%83) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %85 = tensor.empty() : tensor<4x256x14x14xf16>
    %86 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%84#0, %17 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%85 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %87 = tensor.empty() : tensor<4x256x14x14xf16>
    %88 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%86, %arg58, %arg59 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%87 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %89:2 = call @Unknown48(%88, %79) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %90 = tensor.empty() : tensor<4x256x14x14xf16>
    %91 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%89#0, %18 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%90 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %92 = tensor.empty() : tensor<4x256x14x14xf16>
    %93 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%91, %arg68, %arg69 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%92 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %94:2 = call @Unknown46(%93) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %95 = tensor.empty() : tensor<4x256x14x14xf16>
    %96 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%94#0, %19 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%95 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %97 = tensor.empty() : tensor<4x256x14x14xf16>
    %98 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%96, %arg73, %arg74 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%97 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %99:2 = call @Unknown48(%98, %89#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %100 = tensor.empty() : tensor<4x512x7x7xf16>
    %101 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%99#0, %20 : tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) outs(%100 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %102 = tensor.empty() : tensor<4x512x7x7xf16>
    %103 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%101, %arg88, %arg89 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%102 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %104 = tensor.empty() : tensor<4x512x7x7xf16>
    %105 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%99#0, %21 : tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) outs(%104 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %106 = tensor.empty() : tensor<4x512x7x7xf16>
    %107 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%105, %arg78, %arg79 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%106 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %108:2 = call @Unknown55(%107) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %109 = tensor.empty() : tensor<4x512x7x7xf16>
    %110 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%108#0, %22 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%109 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %111 = tensor.empty() : tensor<4x512x7x7xf16>
    %112 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%110, %arg83, %arg84 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%111 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %113:2 = call @Unknown57(%112, %103) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %114 = tensor.empty() : tensor<4x512x7x7xf16>
    %115 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%113#0, %23 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%114 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %116 = tensor.empty() : tensor<4x512x7x7xf16>
    %117 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%115, %arg93, %arg94 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%116 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %118:2 = call @Unknown55(%117) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %119 = tensor.empty() : tensor<4x512x7x7xf16>
    %120 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%118#0, %24 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%119 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %121 = tensor.empty() : tensor<4x512x7x7xf16>
    %122 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%120, %arg98, %arg99 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%121 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %123:2 = call @Unknown57(%122, %113#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %124 = call @Unknown62(%123#0) : (tensor<4x512x7x7xf16>) -> tensor<4x512xf16>
    %125 = call @Unknown63(%124) : (tensor<4x512xf16>) -> tensor<4x512xf16>
    %126 = tensor.empty() : tensor<4x1000xf16>
    %127 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} ins(%125, %26 : tensor<4x512xf16>, tensor<1000x512xf16>) outs(%126 : tensor<4x1000xf16>) : tensor<4x1000xf16>
    %128 = call @Unknown64(%27, %127) : (tensor<1000xf16>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %129 = call @Unknown65(%128) : (tensor<4x1000xf16>) -> tensor<4xf16>
    %130 = call @Unknown66(%129, %128) : (tensor<4xf16>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %131 = call @Unknown67(%130) : (tensor<4x1000xf16>) -> tensor<4xf16>
    %132 = call @Unknown68(%131) : (tensor<4xf16>) -> tensor<4xf16>
    %133:2 = call @Unknown69(%132, %130, %28, %25) : (tensor<4xf16>, tensor<4x1000xf16>, tensor<4xf16>, tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>)
    %134 = tensor.empty() : tensor<4x512xf16>
    %135 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} ins(%133#1, %26 : tensor<4x1000xf16>, tensor<1000x512xf16>) outs(%134 : tensor<4x512xf16>) : tensor<4x512xf16>
    %136 = call @Unknown70(%135, %123#1) : (tensor<4x512xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %137 = tensor.empty() : tensor<4x512x7x7xf16>
    %138 = tensor.empty() : tensor<512xf32>
    %139 = tensor.empty() : tensor<512xf32>
    %140:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%120, %arg98, %136 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%137, %138, %139 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %141 = tensor.empty() : tensor<4x512x7x7xf16>
    %142 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%140#0, %24 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%141 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %143 = tensor.empty() : tensor<512x512x3x3xf16>
    %144 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%118#0, %140#0 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%143 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %145 = call @Unknown74(%118#1, %142) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %146 = tensor.empty() : tensor<4x512x7x7xf16>
    %147 = tensor.empty() : tensor<512xf32>
    %148 = tensor.empty() : tensor<512xf32>
    %149:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%115, %arg93, %145 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%146, %147, %148 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %150 = tensor.empty() : tensor<4x512x7x7xf16>
    %151 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%149#0, %23 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%150 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %152 = tensor.empty() : tensor<512x512x3x3xf16>
    %153 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%113#0, %149#0 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%152 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %154 = call @Unknown78(%136, %151, %113#1) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %155 = tensor.empty() : tensor<4x512x7x7xf16>
    %156 = tensor.empty() : tensor<512xf32>
    %157 = tensor.empty() : tensor<512xf32>
    %158:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%110, %arg83, %154 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%155, %156, %157 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %159 = tensor.empty() : tensor<4x512x7x7xf16>
    %160 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%158#0, %22 : tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%159 : tensor<4x512x7x7xf16>) : tensor<4x512x7x7xf16>
    %161 = tensor.empty() : tensor<512x512x3x3xf16>
    %162 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%108#0, %158#0 : tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) outs(%161 : tensor<512x512x3x3xf16>) : tensor<512x512x3x3xf16>
    %163 = call @Unknown74(%108#1, %160) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %164 = tensor.empty() : tensor<4x512x7x7xf16>
    %165 = tensor.empty() : tensor<512xf32>
    %166 = tensor.empty() : tensor<512xf32>
    %167:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%105, %arg78, %163 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%164, %165, %166 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %168 = tensor.empty() : tensor<4x256x14x14xf16>
    %169 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%167#0, %21 : tensor<4x512x7x7xf16>, tensor<512x256x3x3xf16>) outs(%168 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %170 = tensor.empty() : tensor<512x256x3x3xf16>
    %171 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%99#0, %167#0 : tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) outs(%170 : tensor<512x256x3x3xf16>) : tensor<512x256x3x3xf16>
    %172 = tensor.empty() : tensor<4x512x7x7xf16>
    %173 = tensor.empty() : tensor<512xf32>
    %174 = tensor.empty() : tensor<512xf32>
    %175:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%101, %arg88, %154 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) outs(%172, %173, %174 : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %176 = tensor.empty() : tensor<4x256x14x14xf16>
    %177 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%175#0, %20 : tensor<4x512x7x7xf16>, tensor<512x256x1x1xf16>) outs(%176 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %178 = tensor.empty() : tensor<512x256x1x1xf16>
    %179 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%99#0, %175#0 : tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) outs(%178 : tensor<512x256x1x1xf16>) : tensor<512x256x1x1xf16>
    %180 = call @Unknown89(%177, %169, %99#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %181 = tensor.empty() : tensor<4x256x14x14xf16>
    %182 = tensor.empty() : tensor<256xf32>
    %183 = tensor.empty() : tensor<256xf32>
    %184:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%96, %arg73, %180 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%181, %182, %183 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %185 = tensor.empty() : tensor<4x256x14x14xf16>
    %186 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%184#0, %19 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%185 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %187 = tensor.empty() : tensor<256x256x3x3xf16>
    %188 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%94#0, %184#0 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%187 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %189 = call @Unknown93(%94#1, %186) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %190 = tensor.empty() : tensor<4x256x14x14xf16>
    %191 = tensor.empty() : tensor<256xf32>
    %192 = tensor.empty() : tensor<256xf32>
    %193:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%91, %arg68, %189 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%190, %191, %192 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %194 = tensor.empty() : tensor<4x256x14x14xf16>
    %195 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%193#0, %18 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%194 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %196 = tensor.empty() : tensor<256x256x3x3xf16>
    %197 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%89#0, %193#0 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%196 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %198 = call @Unknown89(%180, %195, %89#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %199 = tensor.empty() : tensor<4x256x14x14xf16>
    %200 = tensor.empty() : tensor<256xf32>
    %201 = tensor.empty() : tensor<256xf32>
    %202:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%86, %arg58, %198 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%199, %200, %201 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %203 = tensor.empty() : tensor<4x256x14x14xf16>
    %204 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%202#0, %17 : tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%203 : tensor<4x256x14x14xf16>) : tensor<4x256x14x14xf16>
    %205 = tensor.empty() : tensor<256x256x3x3xf16>
    %206 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%84#0, %202#0 : tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) outs(%205 : tensor<256x256x3x3xf16>) : tensor<256x256x3x3xf16>
    %207 = call @Unknown93(%84#1, %204) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %208 = tensor.empty() : tensor<4x256x14x14xf16>
    %209 = tensor.empty() : tensor<256xf32>
    %210 = tensor.empty() : tensor<256xf32>
    %211:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%81, %arg53, %207 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%208, %209, %210 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %212 = tensor.empty() : tensor<4x128x28x28xf16>
    %213 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%211#0, %16 : tensor<4x256x14x14xf16>, tensor<256x128x3x3xf16>) outs(%212 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %214 = tensor.empty() : tensor<256x128x3x3xf16>
    %215 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%75#0, %211#0 : tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) outs(%214 : tensor<256x128x3x3xf16>) : tensor<256x128x3x3xf16>
    %216 = tensor.empty() : tensor<4x256x14x14xf16>
    %217 = tensor.empty() : tensor<256xf32>
    %218 = tensor.empty() : tensor<256xf32>
    %219:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%77, %arg63, %198 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) outs(%216, %217, %218 : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %220 = tensor.empty() : tensor<4x128x28x28xf16>
    %221 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%219#0, %15 : tensor<4x256x14x14xf16>, tensor<256x128x1x1xf16>) outs(%220 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %222 = tensor.empty() : tensor<256x128x1x1xf16>
    %223 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%75#0, %219#0 : tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) outs(%222 : tensor<256x128x1x1xf16>) : tensor<256x128x1x1xf16>
    %224 = call @Unknown108(%221, %213, %75#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %225 = tensor.empty() : tensor<4x128x28x28xf16>
    %226 = tensor.empty() : tensor<128xf32>
    %227 = tensor.empty() : tensor<128xf32>
    %228:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%72, %arg48, %224 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%225, %226, %227 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %229 = tensor.empty() : tensor<4x128x28x28xf16>
    %230 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%228#0, %14 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%229 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %231 = tensor.empty() : tensor<128x128x3x3xf16>
    %232 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%70#0, %228#0 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%231 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %233 = call @Unknown112(%70#1, %230) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %234 = tensor.empty() : tensor<4x128x28x28xf16>
    %235 = tensor.empty() : tensor<128xf32>
    %236 = tensor.empty() : tensor<128xf32>
    %237:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%67, %arg43, %233 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%234, %235, %236 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %238 = tensor.empty() : tensor<4x128x28x28xf16>
    %239 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%237#0, %13 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%238 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %240 = tensor.empty() : tensor<128x128x3x3xf16>
    %241 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%65#0, %237#0 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%240 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %242 = call @Unknown108(%224, %239, %65#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %243 = tensor.empty() : tensor<4x128x28x28xf16>
    %244 = tensor.empty() : tensor<128xf32>
    %245 = tensor.empty() : tensor<128xf32>
    %246:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%62, %arg33, %242 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%243, %244, %245 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %247 = tensor.empty() : tensor<4x128x28x28xf16>
    %248 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%246#0, %12 : tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%247 : tensor<4x128x28x28xf16>) : tensor<4x128x28x28xf16>
    %249 = tensor.empty() : tensor<128x128x3x3xf16>
    %250 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%60#0, %246#0 : tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) outs(%249 : tensor<128x128x3x3xf16>) : tensor<128x128x3x3xf16>
    %251 = call @Unknown112(%60#1, %248) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %252 = tensor.empty() : tensor<4x128x28x28xf16>
    %253 = tensor.empty() : tensor<128xf32>
    %254 = tensor.empty() : tensor<128xf32>
    %255:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%57, %arg28, %251 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%252, %253, %254 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %256 = tensor.empty() : tensor<4x64x56x56xf16>
    %257 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%255#0, %11 : tensor<4x128x28x28xf16>, tensor<128x64x3x3xf16>) outs(%256 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %258 = tensor.empty() : tensor<128x64x3x3xf16>
    %259 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%51#0, %255#0 : tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) outs(%258 : tensor<128x64x3x3xf16>) : tensor<128x64x3x3xf16>
    %260 = tensor.empty() : tensor<4x128x28x28xf16>
    %261 = tensor.empty() : tensor<128xf32>
    %262 = tensor.empty() : tensor<128xf32>
    %263:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%53, %arg38, %242 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) outs(%260, %261, %262 : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %264 = tensor.empty() : tensor<4x64x56x56xf16>
    %265 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%263#0, %10 : tensor<4x128x28x28xf16>, tensor<128x64x1x1xf16>) outs(%264 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %266 = tensor.empty() : tensor<128x64x1x1xf16>
    %267 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%51#0, %263#0 : tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) outs(%266 : tensor<128x64x1x1xf16>) : tensor<128x64x1x1xf16>
    %268 = call @Unknown127(%265, %257, %51#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %269 = tensor.empty() : tensor<4x64x56x56xf16>
    %270 = tensor.empty() : tensor<64xf32>
    %271 = tensor.empty() : tensor<64xf32>
    %272:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%48, %arg23, %268 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) outs(%269, %270, %271 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %273 = tensor.empty() : tensor<4x64x56x56xf16>
    %274 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%272#0, %9 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%273 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %275 = tensor.empty() : tensor<64x64x3x3xf16>
    %276 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%46#0, %272#0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%275 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %277 = call @Unknown131(%46#1, %274) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %278 = tensor.empty() : tensor<4x64x56x56xf16>
    %279 = tensor.empty() : tensor<64xf32>
    %280 = tensor.empty() : tensor<64xf32>
    %281:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%43, %arg18, %277 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) outs(%278, %279, %280 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %282 = tensor.empty() : tensor<4x64x56x56xf16>
    %283 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%281#0, %8 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%282 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %284 = tensor.empty() : tensor<64x64x3x3xf16>
    %285 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%41#0, %281#0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%284 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %286 = call @Unknown127(%268, %283, %41#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %287 = tensor.empty() : tensor<4x64x56x56xf16>
    %288 = tensor.empty() : tensor<64xf32>
    %289 = tensor.empty() : tensor<64xf32>
    %290:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%38, %arg13, %286 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) outs(%287, %288, %289 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %291 = tensor.empty() : tensor<4x64x56x56xf16>
    %292 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%290#0, %7 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%291 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %293 = tensor.empty() : tensor<64x64x3x3xf16>
    %294 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%36#0, %290#0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%293 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %295 = call @Unknown131(%36#1, %292) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %296 = tensor.empty() : tensor<4x64x56x56xf16>
    %297 = tensor.empty() : tensor<64xf32>
    %298 = tensor.empty() : tensor<64xf32>
    %299:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%33, %arg8, %295 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) outs(%296, %297, %298 : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %300 = tensor.empty() : tensor<4x64x56x56xf16>
    %301 = byre.compute_on_tensor @ConvBackwardDataOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%299#0, %6 : tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%300 : tensor<4x64x56x56xf16>) : tensor<4x64x56x56xf16>
    %302 = tensor.empty() : tensor<64x64x3x3xf16>
    %303 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%31, %299#0 : tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) outs(%302 : tensor<64x64x3x3xf16>) : tensor<64x64x3x3xf16>
    %304 = call @Unknown143(%286, %301) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %305 = tensor.empty() : tensor<4x64x112x112xf16>
    %306 = byre.compute_on_tensor @PoolMaxGradOp_f16f16_f16 {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} ins(%29#0, %304 : tensor<4x64x112x112xf16>, tensor<4x64x56x56xf16>) outs(%305 : tensor<4x64x112x112xf16>) : tensor<4x64x112x112xf16>
    %307 = call @Unknown144(%29#1, %306) : (tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %308 = tensor.empty() : tensor<4x64x112x112xf16>
    %309 = tensor.empty() : tensor<64xf32>
    %310 = tensor.empty() : tensor<64xf32>
    %311:3 = byre.compute_on_tensor @BatchNormGradOp_f16f32f16_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%3, %arg3, %307 : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<4x64x112x112xf16>) outs(%308, %309, %310 : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
    %312 = tensor.empty() : tensor<64x3x7x7xf16>
    %313 = byre.compute_on_tensor @ConvBackwardFilterOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%0, %311#0 : tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) outs(%312 : tensor<64x3x7x7xf16>) : tensor<64x3x7x7xf16>
    %314 = call @Unknown147(%133#0, %arg1) : (tensor<4x1000xf16>, tensor<4x1000xf32>) -> tensor<f32>
    %315 = call @Unknown148(%314) : (tensor<f32>) -> tensor<f32>
    %316 = call @Unknown149(%313) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %317 = call @Unknown150(%303) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %318 = call @Unknown150(%294) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %319 = call @Unknown150(%285) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %320 = call @Unknown150(%276) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %321 = call @Unknown154(%259) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %322 = call @Unknown155(%250) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %323 = call @Unknown156(%267) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %324 = call @Unknown155(%241) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %325 = call @Unknown155(%232) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %326 = call @Unknown159(%215) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %327 = call @Unknown160(%206) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %328 = call @Unknown161(%223) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %329 = call @Unknown160(%197) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %330 = call @Unknown160(%188) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %331 = call @Unknown164(%171) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %332 = call @Unknown165(%162) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %333 = call @Unknown166(%179) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %334 = call @Unknown165(%153) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %335 = call @Unknown165(%144) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %336 = tensor.empty() : tensor<1000x512xf16>
    %337 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} ins(%125, %133#1 : tensor<4x512xf16>, tensor<4x1000xf16>) outs(%336 : tensor<1000x512xf16>) : tensor<1000x512xf16>
    %338 = call @Unknown170(%337) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %339 = call @Unknown171(%133#1) : (tensor<4x1000xf16>) -> tensor<1000xf32>
    %340 = call @Unknown172(%339) : (tensor<1000xf32>) -> tensor<1000xf32>
    return %315, %316, %311#1, %311#2, %317, %299#1, %299#2, %318, %290#1, %290#2, %319, %281#1, %281#2, %320, %272#1, %272#2, %321, %255#1, %255#2, %322, %246#1, %246#2, %323, %263#1, %263#2, %324, %237#1, %237#2, %325, %228#1, %228#2, %326, %211#1, %211#2, %327, %202#1, %202#2, %328, %219#1, %219#2, %329, %193#1, %193#2, %330, %184#1, %184#2, %331, %167#1, %167#2, %332, %158#1, %158#2, %333, %175#1, %175#2, %334, %149#1, %149#2, %335, %140#1, %140#2, %338, %340 : tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>
  }
}