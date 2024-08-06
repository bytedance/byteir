// RUN: byteir-opt %s -byre-tensor-opt="append-arg-types entry-func=main" | FileCheck %s

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
  func.func @main(%arg0: tensor<512x200xf32>, %arg1: tensor<512x200xf32>) -> (tensor<256x256xf32>, tensor<512x200xf32>) {
    %0 = "mhlo.slice"(%arg0) <{limit_indices = dense<[128, 200]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %1 = "mhlo.slice"(%arg1) <{limit_indices = dense<[138, 200]> : tensor<2xi64>, start_indices = dense<[10, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %2 = mhlo.reshape %0 : (tensor<128x200xf32>) -> tensor<256x100xf32>
    %3 = mhlo.reshape %1 : (tensor<128x200xf32>) -> tensor<100x256xf32>
    %4 = "mhlo.dot_general"(%2, %3) <{dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<256x100xf32>, tensor<100x256xf32>) -> tensor<256x256xf32>
    %5 = call @Unknown0(%arg0, %arg1) : (tensor<512x200xf32>, tensor<512x200xf32>) -> tensor<512x200xf32>
    return %4, %5 : tensor<256x256xf32>, tensor<512x200xf32>
  }
}