// RUN: byteir-opt %s -byteir-bufferize-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<() -> ()>
module {
  func.func private @Unknown0(%arg0: tensor<512x200xf32>, %arg1: tensor<512x200xf32>) -> tensor<512x200xf32> attributes {__byteir_elementwise_fusion__} {
    %c200 = arith.constant 200 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<512x200xf32>
    %1 = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%arg3 = %0) -> (tensor<512x200xf32>) {
      %2 = scf.for %arg4 = %c0 to %c200 step %c1 iter_args(%arg5 = %arg3) -> (tensor<512x200xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4] [1, 1] [1, 1] : tensor<512x200xf32> to tensor<f32>
        %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4] [1, 1] [1, 1] : tensor<512x200xf32> to tensor<f32>
        %3 = tensor.empty() : tensor<f32>
        %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%extracted_slice, %extracted_slice_0 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %5 = arith.addf %in, %in_1 : f32
          linalg.yield %5 : f32
        } -> tensor<f32>
        %inserted_slice = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [1, 1] [1, 1] : tensor<f32> into tensor<512x200xf32>
        scf.yield %inserted_slice : tensor<512x200xf32>
      }
      scf.yield %2 : tensor<512x200xf32>
    }
    return %1 : tensor<512x200xf32>
  }
  func.func @main(%arg0: tensor<512x200xf32>, %arg1: tensor<512x200xf32>) -> (tensor<256x256xf32>, tensor<512x200xf32>) attributes {__placeholder__byre.entry_point} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [128, 200] [1, 1] : tensor<512x200xf32> to tensor<128x200xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[10, 0] [128, 200] [1, 1] : tensor<512x200xf32> to tensor<128x200xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<128x200xf32> into tensor<25600xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [256, 100] : tensor<25600xf32> into tensor<256x100xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [[0, 1]] : tensor<128x200xf32> into tensor<25600xf32>
    %expanded_2 = tensor.expand_shape %collapsed_1 [[0, 1]] output_shape [100, 256] : tensor<25600xf32> into tensor<100x256xf32>
    %0 = tensor.empty() : tensor<256x256xf32>
    %1 = byre.compute_on_tensor @MatmulOp_f32f32_f32 {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} ins(%expanded, %expanded_2 : tensor<256x100xf32>, tensor<100x256xf32>) outs(%0 : tensor<256x256xf32>) : tensor<256x256xf32>
    %2 = call @Unknown0(%arg0, %arg1) : (tensor<512x200xf32>, tensor<512x200xf32>) -> tensor<512x200xf32>
    return %1, %2 : tensor<256x256xf32>, tensor<512x200xf32>
  }
}