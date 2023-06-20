// RUN: byteir-opt %s -graph-canonicalize="blind-fold=true" | FileCheck %s

func.func @add_insert_slices(%arg0: tensor<64x256x384xf32>, %arg1: tensor<64x256x384xf32>, %arg2: tensor<64x256x384xf32>) -> tensor<64x256x1152xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<64x256x1152xf32>
  %inserted_slice = tensor.insert_slice %arg0 into %0[0, 0, 768] [64, 256, 384] [1, 1, 1] : tensor<64x256x384xf32> into tensor<64x256x1152xf32>
  %inserted_slice_0 = tensor.insert_slice %arg1 into %0[0, 0, 384] [64, 256, 384] [1, 1, 1] : tensor<64x256x384xf32> into tensor<64x256x1152xf32>
  %1 = mhlo.add %inserted_slice, %inserted_slice_0 : tensor<64x256x1152xf32>
  %inserted_slice_1 = tensor.insert_slice %arg2 into %0[0, 0, 0] [64, 256, 384] [1, 1, 1] : tensor<64x256x384xf32> into tensor<64x256x1152xf32>
  %2 = mhlo.add %1, %inserted_slice_1 : tensor<64x256x1152xf32>
  return %2 : tensor<64x256x1152xf32>
}
// CHECK-LABEL: add_insert_slices
// CHECK: mhlo.concatenate
// CHECK-NOT: mhlo.add
