// RUN: byteir-opt %s -linalg-data-place -cse | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (0, 0, d2)>

// CHECK-LABEL: broadcast_output
func.func @broadcast_output(%arg0: tensor<96x1024x1024xf16>, %arg1: tensor<1x1x1024x1024xf32>) -> (tensor<1x1x1024x1024xi1>, tensor<8x12x1024x1024xf16>) attributes {__byteir_elementwise_fusion__} {
  %c1048576 = arith.constant 1048576 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0xFC00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.250000e-01 : f16
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] : tensor<96x1024x1024xf16> into tensor<8x12x1024x1024xf16>
  %collapsed = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<1x1x1024x1024xf32> into tensor<1x1x1048576xf32>
  %collapsed_2 = tensor.collapse_shape %expanded [[0], [1], [2, 3]] : tensor<8x12x1024x1024xf16> into tensor<8x12x1048576xf16>
  %0 = tensor.empty() : tensor<1x1x1048576xi1>
  %1 = tensor.empty() : tensor<8x12x1048576xf16>
  %2:2 = scf.for %arg2 = %c0 to %c1048576 step %c128 iter_args(%arg3 = %1, %arg4 = %0) -> (tensor<8x12x1048576xf16>, tensor<1x1x1048576xi1>) {
    %extracted_slice = tensor.extract_slice %collapsed[0, 0, %arg2] [1, 1, 128] [1, 1, 1] : tensor<1x1x1048576xf32> to tensor<1x1x128xf32>
    %extracted_slice_5 = tensor.extract_slice %arg4[0, 0, %arg2] [1, 1, 128] [1, 1, 1] : tensor<1x1x1048576xi1> to tensor<1x1x128xi1>
  // CHECK: %[[PROD0:.*]] = linalg.generic
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x1x128xf32>) outs(%extracted_slice_5 : tensor<1x1x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %6 = arith.cmpf oeq, %in, %cst_0 : f32
      linalg.yield %6 : i1
    } -> tensor<1x1x128xi1>
    %extracted_slice_6 = tensor.extract_slice %collapsed_2[0, 0, %arg2] [8, 12, 128] [1, 1, 1] : tensor<8x12x1048576xf16> to tensor<8x12x128xf16>
    %4 = tensor.empty() : tensor<8x12x128xf16>
    // CHECK: %[[COM0:.*]] = linalg.generic
    // CHECK-SAME: ins(%[[PROD0]]
    %5 = linalg.generic {indexing_maps = [#map1, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %extracted_slice_6 : tensor<1x1x128xi1>, tensor<8x12x128xf16>) outs(%4 : tensor<8x12x128xf16>) attrs =  {__root__} {
    ^bb0(%in: i1, %in_8: f16, %out: f16):
      %6 = arith.mulf %in_8, %cst_1 : f16
      %7 = arith.select %in, %cst, %6 : f16
      linalg.yield %7 : f16
    } -> tensor<8x12x128xf16>
    %inserted_slice = tensor.insert_slice %5 into %arg3[0, 0, %arg2] [8, 12, 128] [1, 1, 1] : tensor<8x12x128xf16> into tensor<8x12x1048576xf16>
    // CHECK: %[[COPY0:.*]] = linalg.copy ins(%[[PROD0]]
    // CHECK: %[[INSERT0:.*]] = tensor.insert_slice %[[COPY0]] into
    %inserted_slice_7 = tensor.insert_slice %3 into %arg4[0, 0, %arg2] [1, 1, 128] [1, 1, 1] : tensor<1x1x128xi1> into tensor<1x1x1048576xi1>
    scf.yield %inserted_slice, %inserted_slice_7 : tensor<8x12x1048576xf16>, tensor<1x1x1048576xi1>
  } {__byteir_parallel__, __byteir_loop_to_simt__ = "block_id.x"}
  %expanded_3 = tensor.expand_shape %2#0 [[0], [1], [2, 3]] : tensor<8x12x1048576xf16> into tensor<8x12x1024x1024xf16>
  %expanded_4 = tensor.expand_shape %2#1 [[0], [1], [2, 3]] : tensor<1x1x1048576xi1> into tensor<1x1x1024x1024xi1>
  return %expanded_4, %expanded_3 : tensor<1x1x1024x1024xi1>, tensor<8x12x1024x1024xf16>
}

