// RUN: byteir-opt %s -byre-tensor-opt="append-arg-types entry-func=main" | FileCheck %s

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
  func.func private @BatchNormTrainingOp2(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x112x112xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    return %1 : tensor<4x64x112x112xf16>
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
        %dim = tensor.dim %expanded_10, %c1 : tensor<1x?xf16>
        %24 = arith.cmpi ugt, %dim, %c0 : index
        %25 = scf.if %24 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %26 = arith.addf %25, %cst : f16
        %dim_11 = tensor.dim %expanded_10, %c1 : tensor<1x?xf16>
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
        %extracted_slice_10 = tensor.extract_slice %arg4[] [] [] : tensor<f16> to tensor<f16>
        %inserted = tensor.insert %24 into %extracted_slice_10[] : tensor<f16>
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
  func.func private @BatchNormTrainingOp27(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %1 = mhlo.convert %output : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
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
  func.func private @BatchNormTrainingOp35(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %1 = mhlo.convert %output : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
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
  func.func private @BatchNormTrainingOp44(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %1 = mhlo.convert %output : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
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
  func.func private @BatchNormTrainingOp53(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %1 = mhlo.convert %output : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %1 : tensor<4x512x7x7xf16>
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
        %dim = tensor.dim %expanded_8, %c1 : tensor<1x?xf16>
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
        %extracted_slice_8 = tensor.extract_slice %arg4[] [] [] : tensor<f16> to tensor<f16>
        %inserted = tensor.insert %18 into %extracted_slice_8[] : tensor<f16>
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
        %dim = tensor.dim %expanded_10, %c1 : tensor<1x?xf16>
        %24 = arith.cmpi ugt, %dim, %c0 : index
        %25 = scf.if %24 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %dim_11 = tensor.dim %expanded_10, %c1 : tensor<1x?xf16>
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
        %extracted_slice_10 = tensor.extract_slice %arg4[] [] [] : tensor<f16> to tensor<f16>
        %inserted = tensor.insert %23 into %extracted_slice_10[] : tensor<f16>
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
        %dim = tensor.dim %expanded_10, %c1 : tensor<1x?xf16>
        %24 = arith.cmpi ugt, %dim, %c0 : index
        %25 = scf.if %24 -> (f16) {
          %extracted = tensor.extract %expanded_10[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %26 = math.exp %25 : f16
        %27 = arith.addf %26, %cst : f16
        %dim_11 = tensor.dim %expanded_10, %c1 : tensor<1x?xf16>
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
        %extracted_slice_10 = tensor.extract_slice %arg4[] [] [] : tensor<f16> to tensor<f16>
        %inserted = tensor.insert %24 into %extracted_slice_10[] : tensor<f16>
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
  func.func private @BatchNormGradOp71(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<4x512x7x7xf32>) -> (tensor<4x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x512x7x7xf32>) -> tensor<4x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp72(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<4x512x7x7xf16>
    return %2 : tensor<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp73(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
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
  func.func private @ConvBackwardDataOp84(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp85(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func.func private @ConvBackwardDataOp87(%arg0: tensor<4x512x7x7xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<4x256x14x14xf16>
    return %1 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp88(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x512x7x7xf16>) -> tensor<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
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
  func.func private @BatchNormGradOp90(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256x14x14xf32>) -> (tensor<4x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x256x14x14xf32>) -> tensor<4x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp91(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<4x256x14x14xf16>
    return %2 : tensor<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp92(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
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
  func.func private @ConvBackwardDataOp103(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp104(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func.func private @ConvBackwardDataOp106(%arg0: tensor<4x256x14x14xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<4x128x28x28xf16>
    return %1 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp107(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x256x14x14xf16>) -> tensor<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
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
  func.func private @BatchNormGradOp109(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<4x128x28x28xf32>) -> (tensor<4x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x128x28x28xf32>) -> tensor<4x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp110(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<4x128x28x28xf16>
    return %2 : tensor<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp111(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
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
  func.func private @ConvBackwardDataOp122(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp123(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func.func private @ConvBackwardDataOp125(%arg0: tensor<4x128x28x28xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<4x64x56x56xf16>
    return %1 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp126(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x128x28x28xf16>) -> tensor<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
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
  func.func private @BatchNormGradOp128(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp129(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<4x64x56x56xf16>
    return %2 : tensor<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp130(%arg0: tensor<4x64x56x56xf16>, %arg1: tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
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
  func.func private @BatchNormGradOp145(%arg0: tensor<4x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %2 = mhlo.convert %arg2 : (tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x64x112x112xf32>) -> (tensor<4x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<4x64x112x112xf32>) -> tensor<4x64x112x112xf16>
    return %3, %grad_scale, %grad_offset : tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardFilterOp146(%arg0: tensor<4x3x224x224xf16>, %arg1: tensor<4x64x112x112xf16>) -> tensor<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
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
        %dim = tensor.dim %expanded_14, %c1 : tensor<1x?xf16>
        %22 = arith.cmpi ugt, %dim, %c0 : index
        %23 = scf.if %22 -> (f16) {
          %extracted = tensor.extract %expanded_14[%c0, %c0] : tensor<1x?xf16>
          scf.yield %extracted : f16
        } else {
          scf.yield %cst : f16
        }
        %dim_17 = tensor.dim %expanded_16, %c1 : tensor<1x?xf32>
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
        %extracted_slice_14 = tensor.extract_slice %arg5[] [] [] : tensor<f32> to tensor<f32>
        %inserted = tensor.insert %22 into %extracted_slice_14[] : tensor<f32>
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
        %extracted_slice_9 = tensor.extract_slice %arg5[] [] [] : tensor<f32> to tensor<f32>
        %inserted = tensor.insert %19 into %extracted_slice_9[] : tensor<f32>
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
  func.func private @MatmulOp169(%arg0: tensor<4x512xf16>, %arg1: tensor<4x1000xf16>) -> tensor<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<4x1000xf16>) -> tensor<512x1000xf16>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %1 : tensor<1000x512xf16>
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
  func.func @main(%arg0: tensor<4x3x224x224xf32>, %arg1: tensor<4x1000xf32>, %arg2: tensor<64x3x7x7xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64x64x3x3xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64x64x3x3xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64x64x3x3xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64xf32>, %arg22: tensor<64x64x3x3xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<128x64x3x3xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128xf32>, %arg32: tensor<128x128x3x3xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128x64x1x1xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128x3x3xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128x128x3x3xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<256x128x3x3xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256xf32>, %arg57: tensor<256x256x3x3xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256x128x1x1xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256x256x3x3xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<256x256x3x3xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<256xf32>, %arg77: tensor<512x256x3x3xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512xf32>, %arg82: tensor<512x512x3x3xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512x256x1x1xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512xf32>, %arg91: tensor<512xf32>, %arg92: tensor<512x512x3x3xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512xf32>, %arg97: tensor<512x512x3x3xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<512xf32>, %arg102: tensor<1000x512xf32>, %arg103: tensor<1000xf32>) -> (tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.constant dense<0xFC00> : tensor<f16>
    %2 = call @Unknown0(%arg0) : (tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf16>
    %3 = call @Unknown1(%arg2) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %4 = mhlo.convolution(%2, %3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<4x64x112x112xf16>
    %5 = call @BatchNormTrainingOp2(%4, %arg3, %arg4) : (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x112x112xf16>
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
    %30 = "mhlo.reduce_window"(%29#0, %1) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<f16>) -> tensor<4x64x56x56xf16>
    %31 = mhlo.convolution(%30, %6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %32 = call @BatchNormTrainingOp27(%31, %arg8, %arg9) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %33:2 = call @Unknown28(%32) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %34 = mhlo.convolution(%33#0, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %35 = call @BatchNormTrainingOp27(%34, %arg13, %arg14) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %36:2 = call @Unknown30(%35, %30) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %37 = mhlo.convolution(%36#0, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %38 = call @BatchNormTrainingOp27(%37, %arg18, %arg19) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %39:2 = call @Unknown28(%38) : (tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %40 = mhlo.convolution(%39#0, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %41 = call @BatchNormTrainingOp27(%40, %arg23, %arg24) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tensor<4x64x56x56xf16>
    %42:2 = call @Unknown30(%41, %36#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>)
    %43 = mhlo.convolution(%42#0, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<4x128x28x28xf16>
    %44 = call @BatchNormTrainingOp35(%43, %arg38, %arg39) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %45 = mhlo.convolution(%42#0, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<4x128x28x28xf16>
    %46 = call @BatchNormTrainingOp35(%45, %arg28, %arg29) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %47:2 = call @Unknown37(%46) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %48 = mhlo.convolution(%47#0, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %49 = call @BatchNormTrainingOp35(%48, %arg33, %arg34) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %50:2 = call @Unknown39(%49, %44) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %51 = mhlo.convolution(%50#0, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %52 = call @BatchNormTrainingOp35(%51, %arg43, %arg44) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %53:2 = call @Unknown37(%52) : (tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %54 = mhlo.convolution(%53#0, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %55 = call @BatchNormTrainingOp35(%54, %arg48, %arg49) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tensor<4x128x28x28xf16>
    %56:2 = call @Unknown39(%55, %50#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>)
    %57 = mhlo.convolution(%56#0, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<4x256x14x14xf16>
    %58 = call @BatchNormTrainingOp44(%57, %arg63, %arg64) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %59 = mhlo.convolution(%56#0, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<4x256x14x14xf16>
    %60 = call @BatchNormTrainingOp44(%59, %arg53, %arg54) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %61:2 = call @Unknown46(%60) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %62 = mhlo.convolution(%61#0, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %63 = call @BatchNormTrainingOp44(%62, %arg58, %arg59) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %64:2 = call @Unknown48(%63, %58) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %65 = mhlo.convolution(%64#0, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %66 = call @BatchNormTrainingOp44(%65, %arg68, %arg69) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %67:2 = call @Unknown46(%66) : (tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %68 = mhlo.convolution(%67#0, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %69 = call @BatchNormTrainingOp44(%68, %arg73, %arg74) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256x14x14xf16>
    %70:2 = call @Unknown48(%69, %64#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>)
    %71 = mhlo.convolution(%70#0, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<4x512x7x7xf16>
    %72 = call @BatchNormTrainingOp53(%71, %arg88, %arg89) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %73 = mhlo.convolution(%70#0, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<4x512x7x7xf16>
    %74 = call @BatchNormTrainingOp53(%73, %arg78, %arg79) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %75:2 = call @Unknown55(%74) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %76 = mhlo.convolution(%75#0, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %77 = call @BatchNormTrainingOp53(%76, %arg83, %arg84) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %78:2 = call @Unknown57(%77, %72) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %79 = mhlo.convolution(%78#0, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %80 = call @BatchNormTrainingOp53(%79, %arg93, %arg94) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %81:2 = call @Unknown55(%80) : (tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %82 = mhlo.convolution(%81#0, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %83 = call @BatchNormTrainingOp53(%82, %arg98, %arg99) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tensor<4x512x7x7xf16>
    %84:2 = call @Unknown57(%83, %78#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>)
    %85 = call @Unknown62(%84#0) : (tensor<4x512x7x7xf16>) -> tensor<4x512xf16>
    %86 = call @Unknown63(%85) : (tensor<4x512xf16>) -> tensor<4x512xf16>
    %87 = "mhlo.dot_general"(%86, %26) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x512xf16>, tensor<1000x512xf16>) -> tensor<4x1000xf16>
    %88 = call @Unknown64(%27, %87) : (tensor<1000xf16>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %89 = call @Unknown65(%88) : (tensor<4x1000xf16>) -> tensor<4xf16>
    %90 = call @Unknown66(%89, %88) : (tensor<4xf16>, tensor<4x1000xf16>) -> tensor<4x1000xf16>
    %91 = call @Unknown67(%90) : (tensor<4x1000xf16>) -> tensor<4xf16>
    %92 = call @Unknown68(%91) : (tensor<4xf16>) -> tensor<4xf16>
    %93:2 = call @Unknown69(%92, %90, %28, %25) : (tensor<4xf16>, tensor<4x1000xf16>, tensor<4xf16>, tensor<4x1000xf16>) -> (tensor<4x1000xf16>, tensor<4x1000xf16>)
    %94 = "mhlo.dot"(%93#1, %26) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x1000xf16>, tensor<1000x512xf16>) -> tensor<4x512xf16>
    %95 = call @Unknown70(%94, %84#1) : (tensor<4x512xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %96:3 = call @BatchNormGradOp71(%82, %arg98, %95) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %97 = call @ConvBackwardDataOp72(%96#0, %24) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %98 = call @ConvBackwardFilterOp73(%81#0, %96#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %99 = call @Unknown74(%81#1, %97) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %100:3 = call @BatchNormGradOp71(%79, %arg93, %99) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %101 = call @ConvBackwardDataOp72(%100#0, %23) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %102 = call @ConvBackwardFilterOp73(%78#0, %100#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %103 = call @Unknown78(%95, %101, %78#1) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>, tensor<4x512x7x7xi1>) -> tensor<4x512x7x7xf16>
    %104:3 = call @BatchNormGradOp71(%76, %arg83, %103) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %105 = call @ConvBackwardDataOp72(%104#0, %22) : (tensor<4x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<4x512x7x7xf16>
    %106 = call @ConvBackwardFilterOp73(%75#0, %104#0) : (tensor<4x512x7x7xf16>, tensor<4x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %107 = call @Unknown74(%75#1, %105) : (tensor<4x512x7x7xi1>, tensor<4x512x7x7xf16>) -> tensor<4x512x7x7xf16>
    %108:3 = call @BatchNormGradOp71(%73, %arg78, %107) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %109 = call @ConvBackwardDataOp84(%108#0, %21) : (tensor<4x512x7x7xf16>, tensor<512x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %110 = call @ConvBackwardFilterOp85(%70#0, %108#0) : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<512x256x3x3xf16>
    %111:3 = call @BatchNormGradOp71(%71, %arg88, %103) : (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<4x512x7x7xf16>) -> (tensor<4x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %112 = call @ConvBackwardDataOp87(%111#0, %20) : (tensor<4x512x7x7xf16>, tensor<512x256x1x1xf16>) -> tensor<4x256x14x14xf16>
    %113 = call @ConvBackwardFilterOp88(%70#0, %111#0) : (tensor<4x256x14x14xf16>, tensor<4x512x7x7xf16>) -> tensor<512x256x1x1xf16>
    %114 = call @Unknown89(%112, %109, %70#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %115:3 = call @BatchNormGradOp90(%68, %arg73, %114) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %116 = call @ConvBackwardDataOp91(%115#0, %19) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %117 = call @ConvBackwardFilterOp92(%67#0, %115#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %118 = call @Unknown93(%67#1, %116) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %119:3 = call @BatchNormGradOp90(%65, %arg68, %118) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %120 = call @ConvBackwardDataOp91(%119#0, %18) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %121 = call @ConvBackwardFilterOp92(%64#0, %119#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %122 = call @Unknown89(%114, %120, %64#1) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>, tensor<4x256x14x14xi1>) -> tensor<4x256x14x14xf16>
    %123:3 = call @BatchNormGradOp90(%62, %arg58, %122) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %124 = call @ConvBackwardDataOp91(%123#0, %17) : (tensor<4x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<4x256x14x14xf16>
    %125 = call @ConvBackwardFilterOp92(%61#0, %123#0) : (tensor<4x256x14x14xf16>, tensor<4x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %126 = call @Unknown93(%61#1, %124) : (tensor<4x256x14x14xi1>, tensor<4x256x14x14xf16>) -> tensor<4x256x14x14xf16>
    %127:3 = call @BatchNormGradOp90(%59, %arg53, %126) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %128 = call @ConvBackwardDataOp103(%127#0, %16) : (tensor<4x256x14x14xf16>, tensor<256x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %129 = call @ConvBackwardFilterOp104(%56#0, %127#0) : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<256x128x3x3xf16>
    %130:3 = call @BatchNormGradOp90(%57, %arg63, %122) : (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<4x256x14x14xf16>) -> (tensor<4x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %131 = call @ConvBackwardDataOp106(%130#0, %15) : (tensor<4x256x14x14xf16>, tensor<256x128x1x1xf16>) -> tensor<4x128x28x28xf16>
    %132 = call @ConvBackwardFilterOp107(%56#0, %130#0) : (tensor<4x128x28x28xf16>, tensor<4x256x14x14xf16>) -> tensor<256x128x1x1xf16>
    %133 = call @Unknown108(%131, %128, %56#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %134:3 = call @BatchNormGradOp109(%54, %arg48, %133) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %135 = call @ConvBackwardDataOp110(%134#0, %14) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %136 = call @ConvBackwardFilterOp111(%53#0, %134#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %137 = call @Unknown112(%53#1, %135) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %138:3 = call @BatchNormGradOp109(%51, %arg43, %137) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %139 = call @ConvBackwardDataOp110(%138#0, %13) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %140 = call @ConvBackwardFilterOp111(%50#0, %138#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %141 = call @Unknown108(%133, %139, %50#1) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>, tensor<4x128x28x28xi1>) -> tensor<4x128x28x28xf16>
    %142:3 = call @BatchNormGradOp109(%48, %arg33, %141) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %143 = call @ConvBackwardDataOp110(%142#0, %12) : (tensor<4x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<4x128x28x28xf16>
    %144 = call @ConvBackwardFilterOp111(%47#0, %142#0) : (tensor<4x128x28x28xf16>, tensor<4x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %145 = call @Unknown112(%47#1, %143) : (tensor<4x128x28x28xi1>, tensor<4x128x28x28xf16>) -> tensor<4x128x28x28xf16>
    %146:3 = call @BatchNormGradOp109(%45, %arg28, %145) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %147 = call @ConvBackwardDataOp122(%146#0, %11) : (tensor<4x128x28x28xf16>, tensor<128x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %148 = call @ConvBackwardFilterOp123(%42#0, %146#0) : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<128x64x3x3xf16>
    %149:3 = call @BatchNormGradOp109(%43, %arg38, %141) : (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<4x128x28x28xf16>) -> (tensor<4x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %150 = call @ConvBackwardDataOp125(%149#0, %10) : (tensor<4x128x28x28xf16>, tensor<128x64x1x1xf16>) -> tensor<4x64x56x56xf16>
    %151 = call @ConvBackwardFilterOp126(%42#0, %149#0) : (tensor<4x64x56x56xf16>, tensor<4x128x28x28xf16>) -> tensor<128x64x1x1xf16>
    %152 = call @Unknown127(%150, %147, %42#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %153:3 = call @BatchNormGradOp128(%40, %arg23, %152) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %154 = call @ConvBackwardDataOp129(%153#0, %9) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %155 = call @ConvBackwardFilterOp130(%39#0, %153#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %156 = call @Unknown131(%39#1, %154) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %157:3 = call @BatchNormGradOp128(%37, %arg18, %156) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %158 = call @ConvBackwardDataOp129(%157#0, %8) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %159 = call @ConvBackwardFilterOp130(%36#0, %157#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %160 = call @Unknown127(%152, %158, %36#1) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>, tensor<4x64x56x56xi1>) -> tensor<4x64x56x56xf16>
    %161:3 = call @BatchNormGradOp128(%34, %arg13, %160) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %162 = call @ConvBackwardDataOp129(%161#0, %7) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %163 = call @ConvBackwardFilterOp130(%33#0, %161#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %164 = call @Unknown131(%33#1, %162) : (tensor<4x64x56x56xi1>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %165:3 = call @BatchNormGradOp128(%31, %arg8, %164) : (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<4x64x56x56xf16>) -> (tensor<4x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %166 = call @ConvBackwardDataOp129(%165#0, %6) : (tensor<4x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<4x64x56x56xf16>
    %167 = call @ConvBackwardFilterOp130(%30, %165#0) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %168 = call @Unknown143(%160, %166) : (tensor<4x64x56x56xf16>, tensor<4x64x56x56xf16>) -> tensor<4x64x56x56xf16>
    %169 = "mhlo.select_and_scatter"(%29#0, %168, %0) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.compare  GE, %arg104, %arg105 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %199 : tensor<i1>
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %199 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %199 : tensor<f16>
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<4x64x112x112xf16>, tensor<4x64x56x56xf16>, tensor<f16>) -> tensor<4x64x112x112xf16>
    %170 = call @Unknown144(%29#1, %169) : (tensor<4x64x112x112xi1>, tensor<4x64x112x112xf16>) -> tensor<4x64x112x112xf16>
    %171:3 = call @BatchNormGradOp145(%4, %arg3, %170) : (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<4x64x112x112xf16>) -> (tensor<4x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %172 = call @ConvBackwardFilterOp146(%2, %171#0) : (tensor<4x3x224x224xf16>, tensor<4x64x112x112xf16>) -> tensor<64x3x7x7xf16>
    %173 = call @Unknown147(%93#0, %arg1) : (tensor<4x1000xf16>, tensor<4x1000xf32>) -> tensor<f32>
    %174 = call @Unknown148(%173) : (tensor<f32>) -> tensor<f32>
    %175 = call @Unknown149(%172) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %176 = call @Unknown150(%167) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %177 = call @Unknown150(%163) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %178 = call @Unknown150(%159) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %179 = call @Unknown150(%155) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %180 = call @Unknown154(%148) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %181 = call @Unknown155(%144) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %182 = call @Unknown156(%151) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %183 = call @Unknown155(%140) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %184 = call @Unknown155(%136) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %185 = call @Unknown159(%129) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %186 = call @Unknown160(%125) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %187 = call @Unknown161(%132) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %188 = call @Unknown160(%121) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %189 = call @Unknown160(%117) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %190 = call @Unknown164(%110) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %191 = call @Unknown165(%106) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %192 = call @Unknown166(%113) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %193 = call @Unknown165(%102) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %194 = call @Unknown165(%98) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %195 = call @MatmulOp169(%86, %93#1) : (tensor<4x512xf16>, tensor<4x1000xf16>) -> tensor<1000x512xf16>
    %196 = call @Unknown170(%195) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %197 = call @Unknown171(%93#1) : (tensor<4x1000xf16>) -> tensor<1000xf32>
    %198 = call @Unknown172(%197) : (tensor<1000xf32>) -> tensor<1000xf32>
    return %174, %175, %171#1, %171#2, %176, %165#1, %165#2, %177, %161#1, %161#2, %178, %157#1, %157#2, %179, %153#1, %153#2, %180, %146#1, %146#2, %181, %142#1, %142#2, %182, %149#1, %149#2, %183, %138#1, %138#2, %184, %134#1, %134#2, %185, %127#1, %127#2, %186, %123#1, %123#2, %187, %130#1, %130#2, %188, %119#1, %119#2, %189, %115#1, %115#2, %190, %108#1, %108#2, %191, %104#1, %104#2, %192, %111#1, %111#2, %193, %100#1, %100#2, %194, %96#1, %96#2, %196, %198 : tensor<f32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>
  }
}