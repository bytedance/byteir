// RUN: byteir-opt %s -byre-tensor-opt="append-arg-types entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<() -> ()>
module {
  func.func private @Unknown0(%arg0: tensor<1x512xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 4.900000e+01 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%arg3 = %0) -> (tensor<1x512x7x7xf16>) {
      %2 = scf.for %arg4 = %c0 to %c7 step %c1 iter_args(%arg5 = %arg3) -> (tensor<1x512x7x7xf16>) {
        %3 = scf.for %arg6 = %c0 to %c7 step %c1 iter_args(%arg7 = %arg5) -> (tensor<1x512x7x7xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg2] [1, 1] [1, 1] : tensor<1x512xf16> to tensor<f16>
          %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x512x7x7xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_1 : tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_2: f16, %out: f16):
            %6 = arith.divf %in, %cst_0 : f16
            %7 = arith.cmpf ogt, %in_2, %cst : f16
            %8 = arith.select %7, %6, %cst : f16
            linalg.yield %8 : f16
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
  func.func private @BatchNormGradOp1(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = mhlo.convert %arg0 : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %2 = mhlo.convert %arg2 : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %3, %grad_scale, %grad_offset : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func.func private @ConvBackwardDataOp2(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp3(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func.func private @Unknown4(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
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
            %6 = arith.cmpf ogt, %in, %cst : f16
            %7 = arith.select %6, %in_1, %cst : f16
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
  func.func private @Unknown8(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>, %arg2: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x512x7x7xf16>
    %1 = scf.for %arg3 = %c0 to %c512 step %c1 iter_args(%arg4 = %0) -> (tensor<1x512x7x7xf16>) {
      %2 = scf.for %arg5 = %c0 to %c7 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x512x7x7xf16>) {
        %3 = scf.for %arg7 = %c0 to %c7 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x512x7x7xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x512x7x7xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x512x7x7xf16> to tensor<f16>
          %extracted_slice_1 = tensor.extract_slice %arg2[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x512x7x7xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<f16>, tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
            %6 = arith.addf %in, %in_2 : f16
            %7 = arith.cmpf ogt, %in_3, %cst : f16
            %8 = arith.select %7, %6, %cst : f16
            linalg.yield %8 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg8[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x512x7x7xf16>
          scf.yield %inserted_slice : tensor<1x512x7x7xf16>
        }
        scf.yield %3 : tensor<1x512x7x7xf16>
      }
      scf.yield %2 : tensor<1x512x7x7xf16>
    }
    return %1 : tensor<1x512x7x7xf16>
  }
  func.func private @ConvBackwardDataOp14(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp15(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func.func private @ConvBackwardDataOp17(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp18(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func.func private @Unknown19(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x256x14x14xf16>
    %1 = scf.for %arg3 = %c0 to %c256 step %c1 iter_args(%arg4 = %0) -> (tensor<1x256x14x14xf16>) {
      %2 = scf.for %arg5 = %c0 to %c14 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x256x14x14xf16>) {
        %3 = scf.for %arg7 = %c0 to %c14 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x256x14x14xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x256x14x14xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x256x14x14xf16> to tensor<f16>
          %extracted_slice_1 = tensor.extract_slice %arg2[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x256x14x14xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<f16>, tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
            %6 = arith.addf %in, %in_2 : f16
            %7 = arith.cmpf ogt, %in_3, %cst : f16
            %8 = arith.select %7, %6, %cst : f16
            linalg.yield %8 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg8[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x256x14x14xf16>
          scf.yield %inserted_slice : tensor<1x256x14x14xf16>
        }
        scf.yield %3 : tensor<1x256x14x14xf16>
      }
      scf.yield %2 : tensor<1x256x14x14xf16>
    }
    return %1 : tensor<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp20(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %1 = mhlo.convert %arg0 : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %2 = mhlo.convert %arg2 : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %3, %grad_scale, %grad_offset : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func.func private @ConvBackwardDataOp21(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp22(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func.func private @Unknown23(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
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
            %6 = arith.cmpf ogt, %in, %cst : f16
            %7 = arith.select %6, %in_1, %cst : f16
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
  func.func private @ConvBackwardDataOp33(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp34(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func.func private @ConvBackwardDataOp36(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp37(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func.func private @Unknown38(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x128x28x28xf16>
    %1 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %0) -> (tensor<1x128x28x28xf16>) {
      %2 = scf.for %arg5 = %c0 to %c28 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x128x28x28xf16>) {
        %3 = scf.for %arg7 = %c0 to %c28 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x128x28x28xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x128x28x28xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x128x28x28xf16> to tensor<f16>
          %extracted_slice_1 = tensor.extract_slice %arg2[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x128x28x28xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<f16>, tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
            %6 = arith.addf %in, %in_2 : f16
            %7 = arith.cmpf ogt, %in_3, %cst : f16
            %8 = arith.select %7, %6, %cst : f16
            linalg.yield %8 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg8[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x128x28x28xf16>
          scf.yield %inserted_slice : tensor<1x128x28x28xf16>
        }
        scf.yield %3 : tensor<1x128x28x28xf16>
      }
      scf.yield %2 : tensor<1x128x28x28xf16>
    }
    return %1 : tensor<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp39(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %1 = mhlo.convert %arg0 : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %2 = mhlo.convert %arg2 : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %3, %grad_scale, %grad_offset : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func.func private @ConvBackwardDataOp40(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp41(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func.func private @Unknown42(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
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
            %6 = arith.cmpf ogt, %in, %cst : f16
            %7 = arith.select %6, %in_1, %cst : f16
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
  func.func private @ConvBackwardDataOp52(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp53(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func.func private @ConvBackwardDataOp55(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp56(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func.func private @Unknown57(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x56x56xf16>
    %1 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %0) -> (tensor<1x64x56x56xf16>) {
      %2 = scf.for %arg5 = %c0 to %c56 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x64x56x56xf16>) {
        %3 = scf.for %arg7 = %c0 to %c56 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x64x56x56xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x56x56xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x56x56xf16> to tensor<f16>
          %extracted_slice_1 = tensor.extract_slice %arg2[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x56x56xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<f16>, tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
            %6 = arith.addf %in, %in_2 : f16
            %7 = arith.cmpf ogt, %in_3, %cst : f16
            %8 = arith.select %7, %6, %cst : f16
            linalg.yield %8 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg8[0, %arg3, %arg5, %arg7] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x64x56x56xf16>
          scf.yield %inserted_slice : tensor<1x64x56x56xf16>
        }
        scf.yield %3 : tensor<1x64x56x56xf16>
      }
      scf.yield %2 : tensor<1x64x56x56xf16>
    }
    return %1 : tensor<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp58(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %2 = mhlo.convert %arg2 : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %3, %grad_scale, %grad_offset : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardDataOp59(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp60(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func.func private @Unknown61(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
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
            %6 = arith.cmpf ogt, %in, %cst : f16
            %7 = arith.select %6, %in_1, %cst : f16
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
  func.func private @Unknown73(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
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
            linalg.yield %6 : f16
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
  func.func private @Unknown74(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %c112 = arith.constant 112 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x64x112x112xf16>
    %1 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %0) -> (tensor<1x64x112x112xf16>) {
      %2 = scf.for %arg4 = %c0 to %c112 step %c1 iter_args(%arg5 = %arg3) -> (tensor<1x64x112x112xf16>) {
        %3 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<1x64x112x112xf16>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x112x112xf16> to tensor<f16>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x64x112x112xf16> to tensor<f16>
          %4 = tensor.empty() : tensor<f16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f16>, tensor<f16>) outs(%4 : tensor<f16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %6 = arith.cmpf ogt, %in, %cst : f16
            %7 = arith.select %6, %in_1, %cst : f16
            linalg.yield %7 : f16
          } -> tensor<f16>
          %inserted_slice = tensor.insert_slice %5 into %arg7[0, %arg2, %arg4, %arg6] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<f16> into tensor<1x64x112x112xf16>
          scf.yield %inserted_slice : tensor<1x64x112x112xf16>
        }
        scf.yield %3 : tensor<1x64x112x112xf16>
      }
      scf.yield %2 : tensor<1x64x112x112xf16>
    }
    return %1 : tensor<1x64x112x112xf16>
  }
  func.func private @BatchNormGradOp75(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x112x112xf16>) -> (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %1 = mhlo.convert %arg0 : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %2 = mhlo.convert %arg2 : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x112x112xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = mhlo.convert %grad_operand : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf16>
    return %3, %grad_scale, %grad_offset : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func private @ConvBackwardFilterOp76(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<1x64x112x112xf16>) -> tensor<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
  }
  func.func private @Unknown77(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown78(%arg0: tensor<1x1000xf16>) -> tensor<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x1000xf32>
    %1 = scf.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %0) -> (tensor<1x1000xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[0, %arg1] [1, 1] [1, 1] : tensor<1x1000xf16> to tensor<f16>
      %2 = tensor.empty() : tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%extracted_slice : tensor<f16>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f16, %out: f32):
        %4 = arith.extf %in : f16 to f32
        %5 = arith.truncf %4 : f32 to f16
        %6 = arith.extf %5 : f16 to f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      %inserted_slice = tensor.insert_slice %3 into %arg2[0, %arg1] [1, 1] [1, 1] : tensor<f32> into tensor<1x1000xf32>
      scf.yield %inserted_slice : tensor<1x1000xf32>
    }
    %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<1x1000xf32> into tensor<1000xf32>
    return %collapsed : tensor<1000xf32>
  }
  func.func private @Unknown79(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown80(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown84(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown85(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown86(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown89(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown90(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown91(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown94(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown95(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func private @Unknown96(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
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
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<256xf32>, %arg30: tensor<512xf32>, %arg31: tensor<512xf32>, %arg32: tensor<512xf32>, %arg33: tensor<512xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<512xf32>, %arg38: tensor<512xf32>, %arg39: tensor<512xf32>, %arg40: tensor<64xf32>, %arg41: tensor<64xf32>, %arg42: tensor<64xf32>, %arg43: tensor<64xf32>, %arg44: tensor<64xf32>, %arg45: tensor<64xf32>, %arg46: tensor<64xf32>, %arg47: tensor<64xf32>, %arg48: tensor<64xf32>, %arg49: tensor<64xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<128xf32>, %arg53: tensor<128xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<512xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<64x3x7x7xf16>, %arg81: tensor<1x3x224x224xf16>, %arg82: tensor<1x64x112x112xf16>, %arg83: tensor<1x64x112x112xf16>, %arg84: tensor<1x64x56x56xf16>, %arg85: tensor<64x64x3x3xf16>, %arg86: tensor<1x64x56x56xf16>, %arg87: tensor<1x64x56x56xf16>, %arg88: tensor<64x64x3x3xf16>, %arg89: tensor<1x64x56x56xf16>, %arg90: tensor<1x64x56x56xf16>, %arg91: tensor<64x64x3x3xf16>, %arg92: tensor<1x64x56x56xf16>, %arg93: tensor<1x64x56x56xf16>, %arg94: tensor<64x64x3x3xf16>, %arg95: tensor<1x64x56x56xf16>, %arg96: tensor<1x64x56x56xf16>, %arg97: tensor<128x64x3x3xf16>, %arg98: tensor<1x128x28x28xf16>, %arg99: tensor<1x128x28x28xf16>, %arg100: tensor<128x128x3x3xf16>, %arg101: tensor<1x128x28x28xf16>, %arg102: tensor<128x64x1x1xf16>, %arg103: tensor<1x128x28x28xf16>, %arg104: tensor<1x128x28x28xf16>, %arg105: tensor<128x128x3x3xf16>, %arg106: tensor<1x128x28x28xf16>, %arg107: tensor<1x128x28x28xf16>, %arg108: tensor<128x128x3x3xf16>, %arg109: tensor<1x128x28x28xf16>, %arg110: tensor<1x128x28x28xf16>, %arg111: tensor<256x128x3x3xf16>, %arg112: tensor<1x256x14x14xf16>, %arg113: tensor<1x256x14x14xf16>, %arg114: tensor<256x256x3x3xf16>, %arg115: tensor<1x256x14x14xf16>, %arg116: tensor<256x128x1x1xf16>, %arg117: tensor<1x256x14x14xf16>, %arg118: tensor<1x256x14x14xf16>, %arg119: tensor<256x256x3x3xf16>, %arg120: tensor<1x256x14x14xf16>, %arg121: tensor<1x256x14x14xf16>, %arg122: tensor<256x256x3x3xf16>, %arg123: tensor<1x256x14x14xf16>, %arg124: tensor<1x256x14x14xf16>, %arg125: tensor<512x256x3x3xf16>, %arg126: tensor<1x512x7x7xf16>, %arg127: tensor<1x512x7x7xf16>, %arg128: tensor<512x512x3x3xf16>, %arg129: tensor<1x512x7x7xf16>, %arg130: tensor<512x256x1x1xf16>, %arg131: tensor<1x512x7x7xf16>, %arg132: tensor<1x512x7x7xf16>, %arg133: tensor<512x512x3x3xf16>, %arg134: tensor<1x512x7x7xf16>, %arg135: tensor<1x512x7x7xf16>, %arg136: tensor<512x512x3x3xf16>, %arg137: tensor<1x512x7x7xf16>, %arg138: tensor<1x512x7x7xf16>, %arg139: tensor<1x512xf16>, %arg140: tensor<512x1000xf16>, %arg141: tensor<1x1000xf16>) -> (tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.dot_general"(%arg141, %arg140) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x1000xf16>, tensor<512x1000xf16>) -> tensor<1x512xf16>
    %2 = call @Unknown0(%1, %arg138) : (tensor<1x512xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %3:3 = call @BatchNormGradOp1(%arg137, %arg39, %2) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %4 = call @ConvBackwardDataOp2(%3#0, %arg136) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %5 = call @ConvBackwardFilterOp3(%arg135, %3#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %6 = call @Unknown4(%arg135, %4) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %7:3 = call @BatchNormGradOp1(%arg134, %arg37, %6) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %8 = call @ConvBackwardDataOp2(%7#0, %arg133) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %9 = call @ConvBackwardFilterOp3(%arg132, %7#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %10 = call @Unknown8(%2, %8, %arg132) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %11:3 = call @BatchNormGradOp1(%arg129, %arg33, %10) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %12 = call @ConvBackwardDataOp2(%11#0, %arg128) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %13 = call @ConvBackwardFilterOp3(%arg127, %11#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %14 = call @Unknown4(%arg127, %12) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %15:3 = call @BatchNormGradOp1(%arg126, %arg31, %14) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %16 = call @ConvBackwardDataOp14(%15#0, %arg125) : (tensor<1x512x7x7xf16>, tensor<512x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %17 = call @ConvBackwardFilterOp15(%arg124, %15#0) : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<512x256x3x3xf16>
    %18:3 = call @BatchNormGradOp1(%arg131, %arg35, %10) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %19 = call @ConvBackwardDataOp17(%18#0, %arg130) : (tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>) -> tensor<1x256x14x14xf16>
    %20 = call @ConvBackwardFilterOp18(%arg124, %18#0) : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<512x256x1x1xf16>
    %21 = call @Unknown19(%19, %16, %arg124) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %22:3 = call @BatchNormGradOp20(%arg123, %arg29, %21) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %23 = call @ConvBackwardDataOp21(%22#0, %arg122) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %24 = call @ConvBackwardFilterOp22(%arg121, %22#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %25 = call @Unknown23(%arg121, %23) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %26:3 = call @BatchNormGradOp20(%arg120, %arg27, %25) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %27 = call @ConvBackwardDataOp21(%26#0, %arg119) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %28 = call @ConvBackwardFilterOp22(%arg118, %26#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %29 = call @Unknown19(%21, %27, %arg118) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %30:3 = call @BatchNormGradOp20(%arg115, %arg23, %29) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %31 = call @ConvBackwardDataOp21(%30#0, %arg114) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %32 = call @ConvBackwardFilterOp22(%arg113, %30#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %33 = call @Unknown23(%arg113, %31) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %34:3 = call @BatchNormGradOp20(%arg112, %arg21, %33) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %35 = call @ConvBackwardDataOp33(%34#0, %arg111) : (tensor<1x256x14x14xf16>, tensor<256x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %36 = call @ConvBackwardFilterOp34(%arg110, %34#0) : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<256x128x3x3xf16>
    %37:3 = call @BatchNormGradOp20(%arg117, %arg25, %29) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %38 = call @ConvBackwardDataOp36(%37#0, %arg116) : (tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>) -> tensor<1x128x28x28xf16>
    %39 = call @ConvBackwardFilterOp37(%arg110, %37#0) : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<256x128x1x1xf16>
    %40 = call @Unknown38(%38, %35, %arg110) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %41:3 = call @BatchNormGradOp39(%arg109, %arg19, %40) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %42 = call @ConvBackwardDataOp40(%41#0, %arg108) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %43 = call @ConvBackwardFilterOp41(%arg107, %41#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %44 = call @Unknown42(%arg107, %42) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %45:3 = call @BatchNormGradOp39(%arg106, %arg17, %44) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %46 = call @ConvBackwardDataOp40(%45#0, %arg105) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %47 = call @ConvBackwardFilterOp41(%arg104, %45#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %48 = call @Unknown38(%40, %46, %arg104) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %49:3 = call @BatchNormGradOp39(%arg101, %arg13, %48) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %50 = call @ConvBackwardDataOp40(%49#0, %arg100) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %51 = call @ConvBackwardFilterOp41(%arg99, %49#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %52 = call @Unknown42(%arg99, %50) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %53:3 = call @BatchNormGradOp39(%arg98, %arg11, %52) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %54 = call @ConvBackwardDataOp52(%53#0, %arg97) : (tensor<1x128x28x28xf16>, tensor<128x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %55 = call @ConvBackwardFilterOp53(%arg96, %53#0) : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<128x64x3x3xf16>
    %56:3 = call @BatchNormGradOp39(%arg103, %arg15, %48) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %57 = call @ConvBackwardDataOp55(%56#0, %arg102) : (tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>) -> tensor<1x64x56x56xf16>
    %58 = call @ConvBackwardFilterOp56(%arg96, %56#0) : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<128x64x1x1xf16>
    %59 = call @Unknown57(%57, %54, %arg96) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %60:3 = call @BatchNormGradOp58(%arg95, %arg9, %59) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %61 = call @ConvBackwardDataOp59(%60#0, %arg94) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %62 = call @ConvBackwardFilterOp60(%arg93, %60#0) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %63 = call @Unknown61(%arg93, %61) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %64:3 = call @BatchNormGradOp58(%arg92, %arg7, %63) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %65 = call @ConvBackwardDataOp59(%64#0, %arg91) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %66 = call @ConvBackwardFilterOp60(%arg90, %64#0) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %67 = call @Unknown57(%59, %65, %arg90) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %68:3 = call @BatchNormGradOp58(%arg89, %arg5, %67) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %69 = call @ConvBackwardDataOp59(%68#0, %arg88) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %70 = call @ConvBackwardFilterOp60(%arg87, %68#0) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %71 = call @Unknown61(%arg87, %69) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %72:3 = call @BatchNormGradOp58(%arg86, %arg3, %71) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %73 = call @ConvBackwardDataOp59(%72#0, %arg85) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %74 = call @ConvBackwardFilterOp60(%arg84, %72#0) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %75 = call @Unknown73(%67, %73) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %76 = "mhlo.select_and_scatter"(%arg83, %75, %0) ({
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %104 = mhlo.compare  GE, %arg142, %arg143 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %104 : tensor<i1>
    }, {
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %104 = mhlo.add %arg142, %arg143 : tensor<f16>
      mhlo.return %104 : tensor<f16>
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<f16>) -> tensor<1x64x112x112xf16>
    %77 = call @Unknown74(%arg83, %76) : (tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %78:3 = call @BatchNormGradOp75(%arg82, %arg1, %77) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<1x64x112x112xf16>) -> (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %79 = call @ConvBackwardFilterOp76(%arg81, %78#0) : (tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>) -> tensor<64x3x7x7xf16>
    %80 = call @Unknown77(%79) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %81 = call @Unknown78(%arg141) : (tensor<1x1000xf16>) -> tensor<1000xf32>
    %82 = mhlo.reshape %arg141 : (tensor<1x1000xf16>) -> tensor<1000x1xf16>
    %83 = "mhlo.dot"(%82, %arg139) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1000x1xf16>, tensor<1x512xf16>) -> tensor<1000x512xf16>
    %84 = call @Unknown79(%83) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %85 = call @Unknown80(%74) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %86 = call @Unknown80(%70) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %87 = call @Unknown80(%66) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %88 = call @Unknown80(%62) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %89 = call @Unknown84(%55) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %90 = call @Unknown85(%51) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %91 = call @Unknown86(%58) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %92 = call @Unknown85(%47) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %93 = call @Unknown85(%43) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %94 = call @Unknown89(%36) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %95 = call @Unknown90(%32) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %96 = call @Unknown91(%39) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %97 = call @Unknown90(%28) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %98 = call @Unknown90(%24) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %99 = call @Unknown94(%17) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %100 = call @Unknown95(%13) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %101 = call @Unknown96(%20) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %102 = call @Unknown95(%9) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %103 = call @Unknown95(%5) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %78#2, %78#1, %80, %81, %84, %72#2, %72#1, %68#2, %68#1, %85, %86, %64#2, %64#1, %60#2, %60#1, %87, %88, %53#2, %53#1, %49#2, %49#1, %89, %90, %91, %56#2, %56#1, %45#2, %45#1, %41#2, %41#1, %92, %93, %34#2, %34#1, %30#2, %30#1, %94, %95, %96, %37#2, %37#1, %26#2, %26#1, %22#2, %22#1, %97, %98, %15#2, %15#1, %11#2, %11#1, %99, %100, %101, %18#2, %18#1, %7#2, %7#1, %3#2, %3#1, %102, %103 : tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>
  }
}