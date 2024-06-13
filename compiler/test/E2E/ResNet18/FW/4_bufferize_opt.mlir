// RUN: byteir-opt %s -byteir-bufferize-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0 mod 64, 49)>
#map2 = affine_map<(d0) -> (d0 mod 64 + 1, 49)>
#map3 = affine_map<(d0, d1) -> (d0 - d1)>
#map4 = affine_map<(d0) -> (d0 * 2)>
#map5 = affine_map<(d0) -> (d0 * 2 + 1)>
module {
  func.func private @Unknown0(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %c224 = arith.constant 224 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x3x224x224xf16>
    %1 = scf.for %arg1 = %c0 to %c3 step %c1 iter_args(%arg2 = %0) -> (tensor<1x3x224x224xf16>) {
      %2 = scf.for %arg3 = %c0 to %c224 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x3x224x224xf16>) {
        %3 = scf.for %arg5 = %c0 to %c224 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x3x224x224xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x3x224x224xf32> to tensor<f32>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f32>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f32, %out: f16):
            %6 = arith.truncf %in : f32 to f16
            linalg.yield %6 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg6[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x3x224x224xf16>
          scf.yield %inserted_slice : tensor<1x3x224x224xf16>
        }
        scf.yield %3 : tensor<1x3x224x224xf16>
      }
      scf.yield %2 : tensor<1x3x224x224xf16>
    }
    return %1 : tensor<1x3x224x224xf16>
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
  func.func private @Unknown3(%arg0: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %c112 = arith.constant 112 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x112x112xf16>
    %1 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %0) -> (tensor<1x64x112x112xf16>) {
      %2 = scf.for %arg3 = %c0 to %c112 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x64x112x112xf16>) {
        %3 = scf.for %arg5 = %c0 to %c112 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x64x112x112xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x112x112xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %out: f16):
            %6 = arith.maximumf %in, %cst : f16
            linalg.yield %6 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg6[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x64x112x112xf16>
          scf.yield %inserted_slice : tensor<1x64x112x112xf16>
        }
        scf.yield %3 : tensor<1x64x112x112xf16>
      }
      scf.yield %2 : tensor<1x64x112x112xf16>
    }
    return %1 : tensor<1x64x112x112xf16>
  }
  func.func private @Unknown4(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown6(%arg0: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %0) -> (tensor<1x64x56x56xf16>) {
      %2 = scf.for %arg3 = %c0 to %c56 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x64x56x56xf16>) {
        %3 = scf.for %arg5 = %c0 to %c56 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x64x56x56xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x56x56xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %out: f16):
            %6 = arith.maximumf %in, %cst : f16
            linalg.yield %6 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg6[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x64x56x56xf16>
          scf.yield %inserted_slice : tensor<1x64x56x56xf16>
        }
        scf.yield %3 : tensor<1x64x56x56xf16>
      }
      scf.yield %2 : tensor<1x64x56x56xf16>
    }
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown9(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %0) -> (tensor<1x64x56x56xf16>) {
      %2 = scf.for %arg4 = %c0 to %c56 step %c1 iter_args(%arg5 = %arg3) -> (tensor<1x64x56x56xf16>) {
        %3 = scf.for %arg6 = %c0 to %c56 step %c1 iter_args(%arg7 = %arg5) -> (tensor<1x64x56x56xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x56x56xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x56x56xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %6 = arith.addf %in, %in_1 : f16
            %7 = arith.maximumf %6, %cst : f16
            linalg.yield %7 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg7[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x64x56x56xf16>
          scf.yield %inserted_slice : tensor<1x64x56x56xf16>
        }
        scf.yield %3 : tensor<1x64x56x56xf16>
      }
      scf.yield %2 : tensor<1x64x56x56xf16>
    }
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @Unknown16(%arg0: tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown18(%arg0: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown20(%arg0: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %0) -> (tensor<1x128x28x28xf16>) {
      %2 = scf.for %arg3 = %c0 to %c28 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x128x28x28xf16>) {
        %3 = scf.for %arg5 = %c0 to %c28 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x128x28x28xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x128x28x28xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %out: f16):
            %6 = arith.maximumf %in, %cst : f16
            linalg.yield %6 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg6[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x128x28x28xf16>
          scf.yield %inserted_slice : tensor<1x128x28x28xf16>
        }
        scf.yield %3 : tensor<1x128x28x28xf16>
      }
      scf.yield %2 : tensor<1x128x28x28xf16>
    }
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown21(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown23(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %0) -> (tensor<1x128x28x28xf16>) {
      %2 = scf.for %arg4 = %c0 to %c28 step %c1 iter_args(%arg5 = %arg3) -> (tensor<1x128x28x28xf16>) {
        %3 = scf.for %arg6 = %c0 to %c28 step %c1 iter_args(%arg7 = %arg5) -> (tensor<1x128x28x28xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x128x28x28xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x128x28x28xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %6 = arith.addf %in, %in_1 : f16
            %7 = arith.maximumf %6, %cst : f16
            linalg.yield %7 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg7[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x128x28x28xf16>
          scf.yield %inserted_slice : tensor<1x128x28x28xf16>
        }
        scf.yield %3 : tensor<1x128x28x28xf16>
      }
      scf.yield %2 : tensor<1x128x28x28xf16>
    }
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @Unknown30(%arg0: tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown32(%arg0: tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown34(%arg0: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = scf.for %arg1 = %c0 to %c256 step %c1 iter_args(%arg2 = %0) -> (tensor<1x256x14x14xf16>) {
      %2 = scf.for %arg3 = %c0 to %c14 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x256x14x14xf16>) {
        %3 = scf.for %arg5 = %c0 to %c14 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x256x14x14xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x256x14x14xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %out: f16):
            %6 = arith.maximumf %in, %cst : f16
            linalg.yield %6 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg6[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x256x14x14xf16>
          scf.yield %inserted_slice : tensor<1x256x14x14xf16>
        }
        scf.yield %3 : tensor<1x256x14x14xf16>
      }
      scf.yield %2 : tensor<1x256x14x14xf16>
    }
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown35(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown37(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = scf.for %arg2 = %c0 to %c256 step %c1 iter_args(%arg3 = %0) -> (tensor<1x256x14x14xf16>) {
      %2 = scf.for %arg4 = %c0 to %c14 step %c1 iter_args(%arg5 = %arg3) -> (tensor<1x256x14x14xf16>) {
        %3 = scf.for %arg6 = %c0 to %c14 step %c1 iter_args(%arg7 = %arg5) -> (tensor<1x256x14x14xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x256x14x14xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x256x14x14xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %6 = arith.addf %in, %in_1 : f16
            %7 = arith.maximumf %6, %cst : f16
            linalg.yield %7 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg7[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x256x14x14xf16>
          scf.yield %inserted_slice : tensor<1x256x14x14xf16>
        }
        scf.yield %3 : tensor<1x256x14x14xf16>
      }
      scf.yield %2 : tensor<1x256x14x14xf16>
    }
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @Unknown44(%arg0: tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown46(%arg0: tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown48(%arg0: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %0) -> (tensor<1x512x7x7xf16>) {
      %2 = scf.for %arg3 = %c0 to %c7 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x512x7x7xf16>) {
        %3 = scf.for %arg5 = %c0 to %c7 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x512x7x7xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x512x7x7xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %out: f16):
            %6 = arith.maximumf %in, %cst : f16
            linalg.yield %6 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg6[0, %arg1, %arg3, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x512x7x7xf16>
          scf.yield %inserted_slice : tensor<1x512x7x7xf16>
        }
        scf.yield %3 : tensor<1x512x7x7xf16>
      }
      scf.yield %2 : tensor<1x512x7x7xf16>
    }
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown49(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown51(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%arg3 = %0) -> (tensor<1x512x7x7xf16>) {
      %2 = scf.for %arg4 = %c0 to %c7 step %c1 iter_args(%arg5 = %arg3) -> (tensor<1x512x7x7xf16>) {
        %3 = scf.for %arg6 = %c0 to %c7 step %c1 iter_args(%arg7 = %arg5) -> (tensor<1x512x7x7xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x512x7x7xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x512x7x7xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %6 = arith.addf %in, %in_1 : f16
            %7 = arith.maximumf %6, %cst : f16
            linalg.yield %7 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg7[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x512x7x7xf16>
          scf.yield %inserted_slice : tensor<1x512x7x7xf16>
        }
        scf.yield %3 : tensor<1x512x7x7xf16>
      }
      scf.yield %2 : tensor<1x512x7x7xf16>
    }
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @Unknown58(%arg0: tensor<1x512x7x7xf16>) -> tensor<1x512xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<1x512x7x7xf16> into tensor<512x49xf16>
    %0 = tensor.empty() : tensor<512xf16>
    %1 = scf.forall (%arg1) in (512) shared_outs(%arg2 = %0) -> (tensor<512xf16>) {
      %extracted_slice = tensor.extract_slice %collapsed[%arg1, 0] [1, 49] [1, 1] : tensor<512x49xf16> to tensor<49xf16>
      %expanded_0 = tensor.expand_shape %extracted_slice [[0, 1]] : tensor<49xf16> into tensor<1x49xf16>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg1] [1] [1] : tensor<512xf16> to tensor<f16>
      %2 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<64xf16>
      %3 = scf.forall (%arg3) in (64) shared_outs(%arg4 = %2) -> (tensor<64xf16>) {
        %15 = affine.min #map1(%arg3)
        %16 = affine.min #map2(%arg3)
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
        tensor.parallel_insert_slice %14 into %arg2[%arg1] [1] [1] : tensor<f16> into tensor<512xf16>
      }
    } {mapping = [#gpu.block<x>]}
    %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<512xf16> into tensor<1x512xf16>
    return %expanded : tensor<1x512xf16>
  }
  func.func private @Unknown59(%arg0: tensor<1x512xf16>) -> tensor<1x512xf16> attributes {__byteir_elementwise_fusion__} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.040100e-02 : f16
    %0 = tensor.empty() : tensor<1x512xf16>
    %1 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %0) -> (tensor<1x512xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[0, %arg1] [1, 1] [1, 1] : tensor<1x512xf16> to tensor<f16>
      %2 = tensor.empty() : tensor<f16>
      %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%2 : tensor<f16>) {
      ^bb0(%in: f16, %out: f16):
        %4 = arith.mulf %in, %cst : f16
        linalg.yield %4 : f16
      } -> tensor<f16>
      %inserted_slice = tensor.insert_slice %3 into %arg2[0, %arg1] [1, 1] [1, 1] : tensor<f16> into tensor<1x512xf16>
      scf.yield %inserted_slice : tensor<1x512xf16>
    }
    return %1 : tensor<1x512xf16>
  }
  func.func private @Unknown60(%arg0: tensor<1000x512xf32>) -> tensor<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown61(%arg0: tensor<1000xf32>, %arg1: tensor<1x1000xf16>) -> tensor<1x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x1000xf16>
    %1 = scf.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %0) -> (tensor<1x1000xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [1] [1] : tensor<1000xf32> to tensor<f32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg2] [1, 1] [1, 1] : tensor<1x1000xf16> to tensor<f16>
      %2 = tensor.empty() : tensor<f16>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f32>, tensor<f16>) outs(%2 : tensor<f16>) {
      ^bb0(%in: f32, %in_1: f16, %out: f16):
        %4 = arith.truncf %in : f32 to f16
        %5 = arith.addf %in_1, %4 : f16
        linalg.yield %5 : f16
      } -> tensor<f16>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, %arg2] [1, 1] [1, 1] : tensor<f16> into tensor<1x1000xf16>
      scf.yield %inserted_slice : tensor<1x1000xf16>
    }
    return %1 : tensor<1x1000xf16>
  }
  func.func private @Unknown62(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %0) -> (tensor<64xf32>) {
      %extracted_slice = tensor.extract_slice %arg1[%arg2] [1] [1] : tensor<64xf32> to tensor<f32>
      %extracted_slice_1 = tensor.extract_slice %arg0[%arg2] [1] [1] : tensor<64xf32> to tensor<f32>
      %2 = tensor.empty() : tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_1 : tensor<f32>, tensor<f32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %4 = arith.mulf %in, %cst : f32
        %5 = arith.mulf %in_2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2] [1] [1] : tensor<f32> into tensor<64xf32>
      scf.yield %inserted_slice : tensor<64xf32>
    }
    return %1 : tensor<64xf32>
  }
  func.func private @Unknown72(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %0) -> (tensor<128xf32>) {
      %extracted_slice = tensor.extract_slice %arg1[%arg2] [1] [1] : tensor<128xf32> to tensor<f32>
      %extracted_slice_1 = tensor.extract_slice %arg0[%arg2] [1] [1] : tensor<128xf32> to tensor<f32>
      %2 = tensor.empty() : tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_1 : tensor<f32>, tensor<f32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %4 = arith.mulf %in, %cst : f32
        %5 = arith.mulf %in_2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2] [1] [1] : tensor<f32> into tensor<128xf32>
      scf.yield %inserted_slice : tensor<128xf32>
    }
    return %1 : tensor<128xf32>
  }
  func.func private @Unknown82(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = scf.for %arg2 = %c0 to %c256 step %c1 iter_args(%arg3 = %0) -> (tensor<256xf32>) {
      %extracted_slice = tensor.extract_slice %arg1[%arg2] [1] [1] : tensor<256xf32> to tensor<f32>
      %extracted_slice_1 = tensor.extract_slice %arg0[%arg2] [1] [1] : tensor<256xf32> to tensor<f32>
      %2 = tensor.empty() : tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_1 : tensor<f32>, tensor<f32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %4 = arith.mulf %in, %cst : f32
        %5 = arith.mulf %in_2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2] [1] [1] : tensor<f32> into tensor<256xf32>
      scf.yield %inserted_slice : tensor<256xf32>
    }
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown92(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.899999976 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%arg3 = %0) -> (tensor<512xf32>) {
      %extracted_slice = tensor.extract_slice %arg1[%arg2] [1] [1] : tensor<512xf32> to tensor<f32>
      %extracted_slice_1 = tensor.extract_slice %arg0[%arg2] [1] [1] : tensor<512xf32> to tensor<f32>
      %2 = tensor.empty() : tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_1 : tensor<f32>, tensor<f32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %4 = arith.mulf %in, %cst : f32
        %5 = arith.mulf %in_2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2] [1] [1] : tensor<f32> into tensor<512xf32>
      scf.yield %inserted_slice : tensor<512xf32>
    }
    return %1 : tensor<512xf32>
  }
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<1000xf32>, %arg4: tensor<1000x512xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64x64x3x3xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64x64x3x3xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x64x3x3xf32>, %arg22: tensor<128x128x3x3xf32>, %arg23: tensor<128x64x1x1xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128x128x3x3xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x128x3x3xf32>, %arg37: tensor<256x256x3x3xf32>, %arg38: tensor<256x128x1x1xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<256xf32>, %arg44: tensor<256xf32>, %arg45: tensor<256x256x3x3xf32>, %arg46: tensor<256x256x3x3xf32>, %arg47: tensor<512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<512x256x3x3xf32>, %arg52: tensor<512x512x3x3xf32>, %arg53: tensor<512x256x1x1xf32>, %arg54: tensor<512xf32>, %arg55: tensor<512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<512xf32>, %arg59: tensor<512xf32>, %arg60: tensor<512x512x3x3xf32>, %arg61: tensor<512x512x3x3xf32>, %arg62: tensor<i64>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<i64>, %arg66: tensor<64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<i64>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<i64>, %arg72: tensor<64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<i64>, %arg75: tensor<64xf32>, %arg76: tensor<64xf32>, %arg77: tensor<i64>, %arg78: tensor<128xf32>, %arg79: tensor<128xf32>, %arg80: tensor<i64>, %arg81: tensor<128xf32>, %arg82: tensor<128xf32>, %arg83: tensor<i64>, %arg84: tensor<128xf32>, %arg85: tensor<128xf32>, %arg86: tensor<i64>, %arg87: tensor<128xf32>, %arg88: tensor<128xf32>, %arg89: tensor<i64>, %arg90: tensor<128xf32>, %arg91: tensor<128xf32>, %arg92: tensor<i64>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<i64>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<i64>, %arg99: tensor<256xf32>, %arg100: tensor<256xf32>, %arg101: tensor<i64>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<i64>, %arg105: tensor<256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<i64>, %arg108: tensor<512xf32>, %arg109: tensor<512xf32>, %arg110: tensor<i64>, %arg111: tensor<512xf32>, %arg112: tensor<512xf32>, %arg113: tensor<i64>, %arg114: tensor<512xf32>, %arg115: tensor<512xf32>, %arg116: tensor<i64>, %arg117: tensor<512xf32>, %arg118: tensor<512xf32>, %arg119: tensor<i64>, %arg120: tensor<512xf32>, %arg121: tensor<512xf32>, %arg122: tensor<1x3x224x224xf32>) -> (tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg122) : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf16>
    %1 = call @Unknown1(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %2 = tensor.empty() : tensor<1x64x112x112xf16>
    %3 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%0, %1 : tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>) outs(%2 : tensor<1x64x112x112xf16>) : tensor<1x64x112x112xf16>
    %4 = tensor.empty() : tensor<1x64x112x112xf16>
    %5 = tensor.empty() : tensor<64xf32>
    %6 = tensor.empty() : tensor<64xf32>
    %7:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%3, %arg1, %arg0 : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) outs(%4, %5, %6 : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
    %8 = call @Unknown3(%7#0) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %9 = tensor.empty() : tensor<1x64x56x56xf16>
    %10 = byre.compute_on_tensor @PoolMaxOp_f16_f16 {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} ins(%8 : tensor<1x64x112x112xf16>) outs(%9 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %11 = call @Unknown4(%arg9) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %12 = tensor.empty() : tensor<1x64x56x56xf16>
    %13 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%10, %11 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%12 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %14 = tensor.empty() : tensor<1x64x56x56xf16>
    %15 = tensor.empty() : tensor<64xf32>
    %16 = tensor.empty() : tensor<64xf32>
    %17:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%13, %arg6, %arg5 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%14, %15, %16 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %18 = call @Unknown6(%17#0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %19 = call @Unknown4(%arg10) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %20 = tensor.empty() : tensor<1x64x56x56xf16>
    %21 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%18, %19 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%20 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %22 = tensor.empty() : tensor<1x64x56x56xf16>
    %23 = tensor.empty() : tensor<64xf32>
    %24 = tensor.empty() : tensor<64xf32>
    %25:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%21, %arg8, %arg7 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%22, %23, %24 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %26 = call @Unknown9(%25#0, %10) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %27 = call @Unknown4(%arg15) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %28 = tensor.empty() : tensor<1x64x56x56xf16>
    %29 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%26, %27 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%28 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %30 = tensor.empty() : tensor<1x64x56x56xf16>
    %31 = tensor.empty() : tensor<64xf32>
    %32 = tensor.empty() : tensor<64xf32>
    %33:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%29, %arg12, %arg11 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%30, %31, %32 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %34 = call @Unknown6(%33#0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %35 = call @Unknown4(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %36 = tensor.empty() : tensor<1x64x56x56xf16>
    %37 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%34, %35 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) outs(%36 : tensor<1x64x56x56xf16>) : tensor<1x64x56x56xf16>
    %38 = tensor.empty() : tensor<1x64x56x56xf16>
    %39 = tensor.empty() : tensor<64xf32>
    %40 = tensor.empty() : tensor<64xf32>
    %41:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%37, %arg14, %arg13 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) outs(%38, %39, %40 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
    %42 = call @Unknown9(%41#0, %26) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %43 = call @Unknown16(%arg23) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %44 = tensor.empty() : tensor<1x128x28x28xf16>
    %45 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%42, %43 : tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>) outs(%44 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %46 = tensor.empty() : tensor<1x128x28x28xf16>
    %47 = tensor.empty() : tensor<128xf32>
    %48 = tensor.empty() : tensor<128xf32>
    %49:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%45, %arg25, %arg24 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%46, %47, %48 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %50 = call @Unknown18(%arg21) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %51 = tensor.empty() : tensor<1x128x28x28xf16>
    %52 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%42, %50 : tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>) outs(%51 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %53 = tensor.empty() : tensor<1x128x28x28xf16>
    %54 = tensor.empty() : tensor<128xf32>
    %55 = tensor.empty() : tensor<128xf32>
    %56:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%52, %arg18, %arg17 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%53, %54, %55 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %57 = call @Unknown20(%56#0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %58 = call @Unknown21(%arg22) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %59 = tensor.empty() : tensor<1x128x28x28xf16>
    %60 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%57, %58 : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%59 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %61 = tensor.empty() : tensor<1x128x28x28xf16>
    %62 = tensor.empty() : tensor<128xf32>
    %63 = tensor.empty() : tensor<128xf32>
    %64:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%60, %arg20, %arg19 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%61, %62, %63 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %65 = call @Unknown23(%64#0, %49#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %66 = call @Unknown21(%arg30) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %67 = tensor.empty() : tensor<1x128x28x28xf16>
    %68 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%65, %66 : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%67 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %69 = tensor.empty() : tensor<1x128x28x28xf16>
    %70 = tensor.empty() : tensor<128xf32>
    %71 = tensor.empty() : tensor<128xf32>
    %72:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%68, %arg27, %arg26 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%69, %70, %71 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %73 = call @Unknown20(%72#0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %74 = call @Unknown21(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %75 = tensor.empty() : tensor<1x128x28x28xf16>
    %76 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%73, %74 : tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) outs(%75 : tensor<1x128x28x28xf16>) : tensor<1x128x28x28xf16>
    %77 = tensor.empty() : tensor<1x128x28x28xf16>
    %78 = tensor.empty() : tensor<128xf32>
    %79 = tensor.empty() : tensor<128xf32>
    %80:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%76, %arg29, %arg28 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) outs(%77, %78, %79 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
    %81 = call @Unknown23(%80#0, %65) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %82 = call @Unknown30(%arg38) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %83 = tensor.empty() : tensor<1x256x14x14xf16>
    %84 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%81, %82 : tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>) outs(%83 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %85 = tensor.empty() : tensor<1x256x14x14xf16>
    %86 = tensor.empty() : tensor<256xf32>
    %87 = tensor.empty() : tensor<256xf32>
    %88:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%84, %arg40, %arg39 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%85, %86, %87 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %89 = call @Unknown32(%arg36) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %90 = tensor.empty() : tensor<1x256x14x14xf16>
    %91 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%81, %89 : tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>) outs(%90 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %92 = tensor.empty() : tensor<1x256x14x14xf16>
    %93 = tensor.empty() : tensor<256xf32>
    %94 = tensor.empty() : tensor<256xf32>
    %95:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%91, %arg33, %arg32 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%92, %93, %94 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %96 = call @Unknown34(%95#0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %97 = call @Unknown35(%arg37) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %98 = tensor.empty() : tensor<1x256x14x14xf16>
    %99 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%96, %97 : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%98 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %100 = tensor.empty() : tensor<1x256x14x14xf16>
    %101 = tensor.empty() : tensor<256xf32>
    %102 = tensor.empty() : tensor<256xf32>
    %103:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%99, %arg35, %arg34 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%100, %101, %102 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %104 = call @Unknown37(%103#0, %88#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %105 = call @Unknown35(%arg45) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %106 = tensor.empty() : tensor<1x256x14x14xf16>
    %107 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%104, %105 : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%106 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %108 = tensor.empty() : tensor<1x256x14x14xf16>
    %109 = tensor.empty() : tensor<256xf32>
    %110 = tensor.empty() : tensor<256xf32>
    %111:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%107, %arg42, %arg41 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%108, %109, %110 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %112 = call @Unknown34(%111#0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %113 = call @Unknown35(%arg46) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %114 = tensor.empty() : tensor<1x256x14x14xf16>
    %115 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%112, %113 : tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) outs(%114 : tensor<1x256x14x14xf16>) : tensor<1x256x14x14xf16>
    %116 = tensor.empty() : tensor<1x256x14x14xf16>
    %117 = tensor.empty() : tensor<256xf32>
    %118 = tensor.empty() : tensor<256xf32>
    %119:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%115, %arg44, %arg43 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) outs(%116, %117, %118 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
    %120 = call @Unknown37(%119#0, %104) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %121 = call @Unknown44(%arg53) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %122 = tensor.empty() : tensor<1x512x7x7xf16>
    %123 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%120, %121 : tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>) outs(%122 : tensor<1x512x7x7xf16>) : tensor<1x512x7x7xf16>
    %124 = tensor.empty() : tensor<1x512x7x7xf16>
    %125 = tensor.empty() : tensor<512xf32>
    %126 = tensor.empty() : tensor<512xf32>
    %127:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%123, %arg55, %arg54 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%124, %125, %126 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %128 = call @Unknown46(%arg51) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %129 = tensor.empty() : tensor<1x512x7x7xf16>
    %130 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} ins(%120, %128 : tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>) outs(%129 : tensor<1x512x7x7xf16>) : tensor<1x512x7x7xf16>
    %131 = tensor.empty() : tensor<1x512x7x7xf16>
    %132 = tensor.empty() : tensor<512xf32>
    %133 = tensor.empty() : tensor<512xf32>
    %134:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%130, %arg48, %arg47 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%131, %132, %133 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %135 = call @Unknown48(%134#0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %136 = call @Unknown49(%arg52) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %137 = tensor.empty() : tensor<1x512x7x7xf16>
    %138 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%135, %136 : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%137 : tensor<1x512x7x7xf16>) : tensor<1x512x7x7xf16>
    %139 = tensor.empty() : tensor<1x512x7x7xf16>
    %140 = tensor.empty() : tensor<512xf32>
    %141 = tensor.empty() : tensor<512xf32>
    %142:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%138, %arg50, %arg49 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%139, %140, %141 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %143 = call @Unknown51(%142#0, %127#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %144 = call @Unknown49(%arg60) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %145 = tensor.empty() : tensor<1x512x7x7xf16>
    %146 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%143, %144 : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%145 : tensor<1x512x7x7xf16>) : tensor<1x512x7x7xf16>
    %147 = tensor.empty() : tensor<1x512x7x7xf16>
    %148 = tensor.empty() : tensor<512xf32>
    %149 = tensor.empty() : tensor<512xf32>
    %150:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%146, %arg57, %arg56 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%147, %148, %149 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %151 = call @Unknown48(%150#0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %152 = call @Unknown49(%arg61) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %153 = tensor.empty() : tensor<1x512x7x7xf16>
    %154 = byre.compute_on_tensor @ConvOp_f16f16_f16 {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} ins(%151, %152 : tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) outs(%153 : tensor<1x512x7x7xf16>) : tensor<1x512x7x7xf16>
    %155 = tensor.empty() : tensor<1x512x7x7xf16>
    %156 = tensor.empty() : tensor<512xf32>
    %157 = tensor.empty() : tensor<512xf32>
    %158:3 = byre.compute_on_tensor @BatchNormTrainingOp_f16f32f32_f16f32f32 {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} ins(%154, %arg59, %arg58 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) outs(%155, %156, %157 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
    %159 = call @Unknown51(%158#0, %143) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %160 = call @Unknown58(%159) : (tensor<1x512x7x7xf16>) -> tensor<1x512xf16>
    %161 = call @Unknown59(%160) : (tensor<1x512xf16>) -> tensor<1x512xf16>
    %162 = call @Unknown60(%arg4) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %163 = tensor.empty() : tensor<512x1000xf16>
    %164 = byre.compute_on_tensor @TransposeOp_f16_f16 {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} ins(%162 : tensor<1000x512xf16>) outs(%163 : tensor<512x1000xf16>) : tensor<512x1000xf16>
    %165 = tensor.empty() : tensor<1x1000xf16>
    %166 = byre.compute_on_tensor @MatmulOp_f16f16_f16 {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} ins(%161, %162 : tensor<1x512xf16>, tensor<1000x512xf16>) outs(%165 : tensor<1x1000xf16>) : tensor<1x1000xf16>
    %167 = call @Unknown61(%arg3, %166) : (tensor<1000xf32>, tensor<1x1000xf16>) -> tensor<1x1000xf16>
    %168 = call @Unknown62(%7#1, %arg63) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %169 = call @Unknown62(%7#2, %arg64) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %170 = call @Unknown62(%17#1, %arg66) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %171 = call @Unknown62(%17#2, %arg67) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %172 = call @Unknown62(%25#1, %arg69) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %173 = call @Unknown62(%25#2, %arg70) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %174 = call @Unknown62(%33#1, %arg72) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %175 = call @Unknown62(%33#2, %arg73) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %176 = call @Unknown62(%41#1, %arg75) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %177 = call @Unknown62(%41#2, %arg76) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %178 = call @Unknown72(%56#1, %arg78) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %179 = call @Unknown72(%56#2, %arg79) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %180 = call @Unknown72(%64#1, %arg81) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %181 = call @Unknown72(%64#2, %arg82) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %182 = call @Unknown72(%49#1, %arg84) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %183 = call @Unknown72(%49#2, %arg85) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %184 = call @Unknown72(%72#1, %arg87) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %185 = call @Unknown72(%72#2, %arg88) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %186 = call @Unknown72(%80#1, %arg90) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %187 = call @Unknown72(%80#2, %arg91) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %188 = call @Unknown82(%95#1, %arg93) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %189 = call @Unknown82(%95#2, %arg94) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %190 = call @Unknown82(%103#1, %arg96) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %191 = call @Unknown82(%103#2, %arg97) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %192 = call @Unknown82(%88#1, %arg99) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %193 = call @Unknown82(%88#2, %arg100) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %194 = call @Unknown82(%111#1, %arg102) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %195 = call @Unknown82(%111#2, %arg103) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %196 = call @Unknown82(%119#1, %arg105) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %197 = call @Unknown82(%119#2, %arg106) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %198 = call @Unknown92(%134#1, %arg108) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %199 = call @Unknown92(%134#2, %arg109) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %200 = call @Unknown92(%142#1, %arg111) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %201 = call @Unknown92(%142#2, %arg112) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %202 = call @Unknown92(%127#1, %arg114) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %203 = call @Unknown92(%127#2, %arg115) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %204 = call @Unknown92(%150#1, %arg117) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %205 = call @Unknown92(%150#2, %arg118) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %206 = call @Unknown92(%158#1, %arg120) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %207 = call @Unknown92(%158#2, %arg121) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    return %167, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %168, %169, %170, %171, %172, %173, %174, %175, %176, %177, %178, %179, %180, %181, %182, %183, %184, %185, %186, %187, %188, %189, %190, %191, %192, %193, %194, %195, %196, %197, %198, %199, %200, %201, %202, %203, %204, %205, %206, %207, %1, %0, %3, %8, %10, %11, %13, %18, %19, %21, %26, %27, %29, %34, %35, %37, %42, %50, %52, %57, %58, %60, %43, %45, %65, %66, %68, %73, %74, %76, %81, %89, %91, %96, %97, %99, %82, %84, %104, %105, %107, %112, %113, %115, %120, %128, %130, %135, %136, %138, %121, %123, %143, %144, %146, %151, %152, %154, %159, %161, %164 : tensor<1x1000xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x3x7x7xf16>, tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512xf16>, tensor<512x1000xf16>
  }
}