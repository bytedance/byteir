# LinalgExt Tiling and Fusion on Attention

Attention mechanisms have reshaped modern machine learning, but their complexity demands advanced computational strategies. Our LinalgExt dialect, tailored to complement upstream Linalg, is specifically designed to optimize these mechanisms. By leveraging enhanced tiling and fusion techniques, our dialect offers a streamlined approach to executing attention-based models efficiently. This document unveils how we tackle the challenges posed by attention and its variations.

## Flash attention example
Here, we demonstrate how to reach [flash attention](https://arxiv.org/abs/2205.14135) in a regular self attention.
In ByteIR, flash attention can be reached from a regular self attention presenting in linalg or linalg-ext ops, such as `matmul` and `softmax`, through just linalg-ext `fuse transformation` of  with proper tiling parameters, proper `tile_sizes` to fully utilize on-chip memory and `tile_interchange` of [2, 1, 0].

```
// input.mlir
func.func @dot_attention(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x32xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %6 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
  %10 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x32xf32>, tensor<32x512xf32>) outs(%7 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %11:4 = linalg_ext.softmax dimension(1) ins(%10 : tensor<1024x512xf32>) outs(%2, %6, %9, %5 : tensor<1024x512xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x512xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  %12 = linalg.matmul {__root__} ins(%11#0, %arg2 : tensor<1024x512xf32>, tensor<512x32xf32>) outs(%8 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  return %12 : tensor<1024x32xf32>
}

// result after transform.structured.fuse_ext {tile_interchange = [2, 1, 0], tile_sizes = [4, 0, 8]}
func.func @dot_attention(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %0 = tensor.empty() : tensor<1024x32xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1024xf32>) -> tensor<1024xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %6:3 = scf.for %arg3 = %c0 to %c512 step %c8 iter_args(%arg4 = %4, %arg5 = %3, %arg6 = %5) -> (tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>) {
    %7:3 = scf.for %arg7 = %c0 to %c1024 step %c4 iter_args(%arg8 = %arg4, %arg9 = %arg5, %arg10 = %arg6) -> (tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg7, 0] [4, 32] [1, 1] : tensor<1024x32xf32> to tensor<4x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg3] [32, 8] [1, 1] : tensor<32x512xf32> to tensor<32x8xf32>
      %8 = tensor.empty() : tensor<4x8xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<4x8xf32>) -> tensor<4x8xf32>
      %10 = linalg.matmul ins(%extracted_slice, %extracted_slice_1 : tensor<4x32xf32>, tensor<32x8xf32>) outs(%9 : tensor<4x8xf32>) -> tensor<4x8xf32>
      %11 = tensor.empty() : tensor<4x8xf32>
      %extracted_slice_2 = tensor.extract_slice %arg9[%arg7] [4] [1] : tensor<1024xf32> to tensor<4xf32>
      %extracted_slice_3 = tensor.extract_slice %arg10[%arg7] [4] [1] : tensor<1024xf32> to tensor<4xf32>
      %12 = tensor.empty() : tensor<4xf32>
      %13:4 = linalg_ext.softmax dimension(1) ins(%10 : tensor<4x8xf32>) outs(%11, %extracted_slice_2, %extracted_slice_3, %12 : tensor<4x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) : tensor<4x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
      %extracted_slice_4 = tensor.extract_slice %arg2[%arg3, 0] [8, 32] [1, 1] : tensor<512x32xf32> to tensor<8x32xf32>
      %extracted_slice_5 = tensor.extract_slice %arg8[%arg7, 0] [4, 32] [1, 1] : tensor<1024x32xf32> to tensor<4x32xf32>
      %14 = tensor.empty() : tensor<4x4xf32>
      %15 = linalg_ext.diag ins(%13#3 : tensor<4xf32>) outs(%14 : tensor<4x4xf32>) : tensor<4x4xf32>
      %16 = tensor.empty() : tensor<4x32xf32>
      %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<4x32xf32>) -> tensor<4x32xf32>
      %18 = linalg.matmul ins(%15, %extracted_slice_5 : tensor<4x4xf32>, tensor<4x32xf32>) outs(%17 : tensor<4x32xf32>) -> tensor<4x32xf32>
      %19 = linalg.matmul {__root__} ins(%13#0, %extracted_slice_4 : tensor<4x8xf32>, tensor<8x32xf32>) outs(%18 : tensor<4x32xf32>) -> tensor<4x32xf32>
      %inserted_slice = tensor.insert_slice %19 into %arg8[%arg7, 0] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<1024x32xf32>
      %inserted_slice_6 = tensor.insert_slice %13#1 into %arg9[%arg7] [4] [1] : tensor<4xf32> into tensor<1024xf32>
      %inserted_slice_7 = tensor.insert_slice %13#2 into %arg10[%arg7] [4] [1] : tensor<4xf32> into tensor<1024xf32>
      scf.yield %inserted_slice, %inserted_slice_6, %inserted_slice_7 : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>
    }
    scf.yield %7#0, %7#1, %7#2 : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>
  }
  return %6#0 : tensor<1024x32xf32>
}

```

And multi-head attention is also supported.
```
// input.mlir
func.func @fuse_multihead_attention(%arg0: tensor<128x16x1024x32xf32>, %arg1: tensor<128x16x32x512xf32>, %arg2: tensor<128x16x512x32xf32>) -> tensor<128x16x1024x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<128x16x1024x512xf32>
  %1 = tensor.empty() : tensor<128x16x1024x32xf32>
  %2 = tensor.empty() : tensor<128x16x1024x512xf32>
  %3 = tensor.empty() : tensor<128x16x1024xf32>
  %4 = tensor.empty() : tensor<128x16x1024xf32>
  %5 = tensor.empty() : tensor<128x16x1024xf32>
  %6 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x16x1024x512xf32>) -> tensor<128x16x1024x512xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%1 : tensor<128x16x1024x32xf32>) -> tensor<128x16x1024x32xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%4 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %10 = linalg_ext.batch_matmul ins(%arg0, %arg1 : tensor<128x16x1024x32xf32>, tensor<128x16x32x512xf32>) outs(%7 : tensor<128x16x1024x512xf32>) layout = "nn"
  %11:4 = linalg_ext.softmax dimension(3) ins(%10 : tensor<128x16x1024x512xf32>) outs(%2, %6, %9, %5 : tensor<128x16x1024x512xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) : tensor<128x16x1024x512xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
  %12 = linalg_ext.batch_matmul ins(%11#0, %arg2 : tensor<128x16x1024x512xf32>, tensor<128x16x512x32xf32>) outs(%8 : tensor<128x16x1024x32xf32>) layout = "nn"  {__root__}
  return %12 : tensor<128x16x1024x32xf32>
}

// result after transform.structured.fuse_ext {tile_sizes = [2, 0, 8, 0, 4], tile_interchange = [0, 1, 4, 3, 2]}
func.func @fuse_multihead_attention(%arg0: tensor<128x16x1024x32xf32>, %arg1: tensor<128x16x32x512xf32>, %arg2: tensor<128x16x512x32xf32>) -> tensor<128x16x1024x32xf32> {
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.empty() : tensor<128x16x1024x32xf32>
  %1 = tensor.empty() : tensor<128x16x1024xf32>
  %2 = tensor.empty() : tensor<128x16x1024xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x16x1024x32xf32>) -> tensor<128x16x1024x32xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %6:3 = scf.for %arg3 = %c0 to %c128 step %c2 iter_args(%arg4 = %4, %arg5 = %3, %arg6 = %5) -> (tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) {
    %7:3 = scf.for %arg7 = %c0 to %c512 step %c4 iter_args(%arg8 = %arg4, %arg9 = %arg5, %arg10 = %arg6) -> (tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) {
      %8:3 = scf.for %arg11 = %c0 to %c1024 step %c8 iter_args(%arg12 = %arg8, %arg13 = %arg9, %arg14 = %arg10) -> (tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, %arg11, 0] [2, 16, 8, 32] [1, 1, 1, 1] : tensor<128x16x1024x32xf32> to tensor<2x16x8x32xf32>
        %extracted_slice_1 = tensor.extract_slice %arg1[%arg3, 0, 0, %arg7] [2, 16, 32, 4] [1, 1, 1, 1] : tensor<128x16x32x512xf32> to tensor<2x16x32x4xf32>
        %9 = tensor.empty() : tensor<2x16x8x4xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<2x16x8x4xf32>) -> tensor<2x16x8x4xf32>
        %11 = linalg_ext.batch_matmul ins(%extracted_slice, %extracted_slice_1 : tensor<2x16x8x32xf32>, tensor<2x16x32x4xf32>) outs(%10 : tensor<2x16x8x4xf32>) layout = "nn"
        %12 = tensor.empty() : tensor<2x16x8x4xf32>
        %extracted_slice_2 = tensor.extract_slice %arg13[%arg3, 0, %arg11] [2, 16, 8] [1, 1, 1] : tensor<128x16x1024xf32> to tensor<2x16x8xf32>
        %extracted_slice_3 = tensor.extract_slice %arg14[%arg3, 0, %arg11] [2, 16, 8] [1, 1, 1] : tensor<128x16x1024xf32> to tensor<2x16x8xf32>
        %13 = tensor.empty() : tensor<2x16x8xf32>
        %14:4 = linalg_ext.softmax dimension(3) ins(%11 : tensor<2x16x8x4xf32>) outs(%12, %extracted_slice_2, %extracted_slice_3, %13 : tensor<2x16x8x4xf32>, tensor<2x16x8xf32>, tensor<2x16x8xf32>, tensor<2x16x8xf32>) : tensor<2x16x8x4xf32>, tensor<2x16x8xf32>, tensor<2x16x8xf32>, tensor<2x16x8xf32>
        %extracted_slice_4 = tensor.extract_slice %arg2[%arg3, 0, %arg7, 0] [2, 16, 4, 32] [1, 1, 1, 1] : tensor<128x16x512x32xf32> to tensor<2x16x4x32xf32>
        %extracted_slice_5 = tensor.extract_slice %arg12[%arg3, 0, %arg11, 0] [2, 16, 8, 32] [1, 1, 1, 1] : tensor<128x16x1024x32xf32> to tensor<2x16x8x32xf32>
        %15 = tensor.empty() : tensor<2x16x8x8xf32>
        %16 = linalg_ext.diag ins(%14#3 : tensor<2x16x8xf32>) outs(%15 : tensor<2x16x8x8xf32>) : tensor<2x16x8x8xf32>
        %17 = tensor.empty() : tensor<2x16x8x32xf32>
        %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<2x16x8x32xf32>) -> tensor<2x16x8x32xf32>
        %19 = linalg_ext.batch_matmul ins(%16, %extracted_slice_5 : tensor<2x16x8x8xf32>, tensor<2x16x8x32xf32>) outs(%18 : tensor<2x16x8x32xf32>) layout = "nn"
        %20 = linalg_ext.batch_matmul ins(%14#0, %extracted_slice_4 : tensor<2x16x8x4xf32>, tensor<2x16x4x32xf32>) outs(%19 : tensor<2x16x8x32xf32>) layout = "nn"  {__root__}
        %inserted_slice = tensor.insert_slice %20 into %arg12[%arg3, 0, %arg11, 0] [2, 16, 8, 32] [1, 1, 1, 1] : tensor<2x16x8x32xf32> into tensor<128x16x1024x32xf32>
        %inserted_slice_6 = tensor.insert_slice %14#1 into %arg13[%arg3, 0, %arg11] [2, 16, 8] [1, 1, 1] : tensor<2x16x8xf32> into tensor<128x16x1024xf32>
        %inserted_slice_7 = tensor.insert_slice %14#2 into %arg14[%arg3, 0, %arg11] [2, 16, 8] [1, 1, 1] : tensor<2x16x8xf32> into tensor<128x16x1024xf32>
        scf.yield %inserted_slice, %inserted_slice_6, %inserted_slice_7 : tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
      }
      scf.yield %8#0, %8#1, %8#2 : tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
    }
    scf.yield %7#0, %7#1, %7#2 : tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
  } {__byteir_parallel__}
  return %6#0 : tensor<128x16x1024x32xf32>
}
```

## Multi-head attention example with tiling on 3 dimensions

```
// input.mlir

func.func @fuse_multihead_attention_tile_3d(%arg0: tensor<128x16x1024x32xf32>, %arg1: tensor<128x16x32x512xf32>, %arg2: tensor<128x16x512x32xf32>) -> tensor<128x16x1024x32xf32> {
  %0 = tensor.empty() : tensor<128x16x1024x512xf32>
  %1 = tensor.empty() : tensor<128x16x1024x32xf32>
  %2 = tensor.empty() : tensor<128x16x1024x512xf32>
  %3 = tensor.empty() : tensor<128x16x1024xf32>
  %4 = tensor.empty() : tensor<128x16x1024xf32>
  %5 = tensor.empty() : tensor<128x16x1024xf32>
  %6 = tensor.empty() : tensor<128x16x32x512xf32>
  %cst = arith.constant 0xFF800000 : f32
  %7 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %8 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x16x1024x512xf32>) -> tensor<128x16x1024x512xf32>
  %9 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<128x16x1024x32xf32>) -> tensor<128x16x1024x32xf32>
  %10 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %11 = linalg_ext.batch_matmul ins(%arg0, %arg1 : tensor<128x16x1024x32xf32>, tensor<128x16x32x512xf32>) outs(%8 : tensor<128x16x1024x512xf32>) layout = "nn"
  %12:4 = linalg_ext.softmax dimension(3) ins(%11 : tensor<128x16x1024x512xf32>) outs(%2, %7, %10, %5 : tensor<128x16x1024x512xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) : tensor<128x16x1024x512xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
  %13 = linalg_ext.batch_matmul ins(%12#0, %arg2 : tensor<128x16x1024x512xf32>, tensor<128x16x512x32xf32>) outs(%9 : tensor<128x16x1024x32xf32>) layout = "nn"  {__root__}
  return %13 : tensor<128x16x1024x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.fuse_ext %0 {tile_sizes = [2, 0, 8, 0, 4], tile_interchange = [0, 1, 4, 3, 2]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
  cleanup
}

// result after transform.structured.fuse_ext

func.func @fuse_multihead_attention_tile_3d(%arg0: tensor<128x16x1024x32xf32>, %arg1: tensor<128x16x32x512xf32>, %arg2: tensor<128x16x512x32xf32>) -> tensor<128x16x1024x32xf32> {
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.empty() : tensor<128x16x1024x32xf32>
  %1 = tensor.empty() : tensor<128x16x1024xf32>
  %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x16x1024x32xf32>) -> tensor<128x16x1024x32xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%1 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %5:3 = scf.for %arg3 = %c0 to %c128 step %c2 iter_args(%arg4 = %3, %arg5 = %2, %arg6 = %4) -> (tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) {
    %6:3 = scf.for %arg7 = %c0 to %c512 step %c4 iter_args(%arg8 = %arg4, %arg9 = %arg5, %arg10 = %arg6) -> (tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) {
      %7:3 = scf.for %arg11 = %c0 to %c1024 step %c8 iter_args(%arg12 = %arg8, %arg13 = %arg9, %arg14 = %arg10) -> (tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, %arg11, 0] [2, 16, 8, 32] [1, 1, 1, 1] : tensor<128x16x1024x32xf32> to tensor<2x16x8x32xf32>
        %extracted_slice_1 = tensor.extract_slice %arg1[%arg3, 0, 0, %arg7] [2, 16, 32, 4] [1, 1, 1, 1] : tensor<128x16x32x512xf32> to tensor<2x16x32x4xf32>
        %8 = tensor.empty() : tensor<2x16x8x4xf32>
        %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<2x16x8x4xf32>) -> tensor<2x16x8x4xf32>
        %10 = linalg_ext.batch_matmul ins(%extracted_slice, %extracted_slice_1 : tensor<2x16x8x32xf32>, tensor<2x16x32x4xf32>) outs(%9 : tensor<2x16x8x4xf32>) layout = "nn" 
        %extracted_slice_2 = tensor.extract_slice %arg13[%arg3, 0, %arg11] [2, 16, 8] [1, 1, 1] : tensor<128x16x1024xf32> to tensor<2x16x8xf32>
        %extracted_slice_3 = tensor.extract_slice %arg14[%arg3, 0, %arg11] [2, 16, 8] [1, 1, 1] : tensor<128x16x1024xf32> to tensor<2x16x8xf32>
        %11 = tensor.empty() : tensor<2x16x8xf32>
        %12:4 = linalg_ext.softmax dimension(3) ins(%10 : tensor<2x16x8x4xf32>) outs(%8, %extracted_slice_2, %extracted_slice_3, %11 : tensor<2x16x8x4xf32>, tensor<2x16x8xf32>, tensor<2x16x8xf32>, tensor<2x16x8xf32>) : tensor<2x16x8x4xf32>, tensor<2x16x8xf32>, tensor<2x16x8xf32>, tensor<2x16x8xf32>
        %extracted_slice_4 = tensor.extract_slice %arg2[%arg3, 0, %arg7, 0] [2, 16, 4, 32] [1, 1, 1, 1] : tensor<128x16x512x32xf32> to tensor<2x16x4x32xf32>
        %extracted_slice_5 = tensor.extract_slice %arg12[%arg3, 0, %arg11, 0] [2, 16, 8, 32] [1, 1, 1, 1] : tensor<128x16x1024x32xf32> to tensor<2x16x8x32xf32>
        %13 = tensor.empty() : tensor<2x16x8x8xf32>
        %14 = linalg_ext.diag ins(%12#3 : tensor<2x16x8xf32>) outs(%13 : tensor<2x16x8x8xf32>) : tensor<2x16x8x8xf32>
        %15 = tensor.empty() : tensor<2x16x8x32xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<2x16x8x32xf32>) -> tensor<2x16x8x32xf32>
        %17 = linalg_ext.batch_matmul ins(%14, %extracted_slice_5 : tensor<2x16x8x8xf32>, tensor<2x16x8x32xf32>) outs(%16 : tensor<2x16x8x32xf32>) layout = "nn" 
        %18 = linalg_ext.batch_matmul ins(%12#0, %extracted_slice_4 : tensor<2x16x8x4xf32>, tensor<2x16x4x32xf32>) outs(%17 : tensor<2x16x8x32xf32>) layout = "nn"  {__root__}
        %inserted_slice = tensor.insert_slice %18 into %arg12[%arg3, 0, %arg11, 0] [2, 16, 8, 32] [1, 1, 1, 1] : tensor<2x16x8x32xf32> into tensor<128x16x1024x32xf32>
        %inserted_slice_6 = tensor.insert_slice %12#1 into %arg13[%arg3, 0, %arg11] [2, 16, 8] [1, 1, 1] : tensor<2x16x8xf32> into tensor<128x16x1024xf32>
        %inserted_slice_7 = tensor.insert_slice %12#2 into %arg14[%arg3, 0, %arg11] [2, 16, 8] [1, 1, 1] : tensor<2x16x8xf32> into tensor<128x16x1024xf32>
        scf.yield %inserted_slice, %inserted_slice_6, %inserted_slice_7 : tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
      }
      scf.yield %7#0, %7#1, %7#2 : tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
    }
    scf.yield %6#0, %6#1, %6#2 : tensor<128x16x1024x32xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
  } {__byteir_parallel__}
  return %5#0 : tensor<128x16x1024x32xf32>
}
```

## Split-head attention with prologue

support split-head attention, see https://arxiv.org/abs/1909.08053

```
// input.mlir
func.func @multihead_attention_with_prologue_proj(%arg0: tensor<4x1024x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<4x8x1024x64xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x512x512xf32>
  %broadcasted = linalg.broadcast ins(%arg1 : tensor<512x512xf32>) outs(%0 : tensor<4x512x512xf32>) dimensions = [0]
  %1 = tensor.empty() : tensor<4x1024x512xf32>
  %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<4x1024x512xf32>) -> tensor<4x1024x512xf32>
  %3 = linalg_ext.batch_matmul ins(%arg0, %broadcasted : tensor<4x1024x512xf32>, tensor<4x512x512xf32>) outs(%2 : tensor<4x1024x512xf32>) layout = "nn"
  %broadcasted_1 = linalg.broadcast ins(%arg2 : tensor<512x512xf32>) outs(%0 : tensor<4x512x512xf32>) dimensions = [0]
  %4 = linalg_ext.batch_matmul ins(%arg0, %broadcasted_1 : tensor<4x1024x512xf32>, tensor<4x512x512xf32>) outs(%2 : tensor<4x1024x512xf32>) layout = "nn"
  %broadcasted_2 = linalg.broadcast ins(%arg3 : tensor<512x512xf32>) outs(%0 : tensor<4x512x512xf32>) dimensions = [0]
  %5 = linalg_ext.batch_matmul ins(%arg0, %broadcasted_2 : tensor<4x1024x512xf32>, tensor<4x512x512xf32>) outs(%2 : tensor<4x1024x512xf32>) layout = "nn"
  %expanded = tensor.expand_shape %3 [[0], [1], [2, 3]] : tensor<4x1024x512xf32> into tensor<4x1024x8x64xf32>
  %expanded_3 = tensor.expand_shape %4 [[0], [1], [2, 3]] : tensor<4x1024x512xf32> into tensor<4x1024x8x64xf32>
  %expanded_4 = tensor.expand_shape %5 [[0], [1], [2, 3]] : tensor<4x1024x512xf32> into tensor<4x1024x8x64xf32>
  %6 = tensor.empty() : tensor<4x8x1024x64xf32>
  %7 = tensor.empty() : tensor<4x8x64x1024xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<4x1024x8x64xf32>) outs(%6 : tensor<4x8x1024x64xf32>) permutation = [0, 2, 1, 3]
  %transposed_5 = linalg.transpose ins(%expanded_3 : tensor<4x1024x8x64xf32>) outs(%7 : tensor<4x8x64x1024xf32>) permutation = [0, 2, 3, 1]
  %transposed_6 = linalg.transpose ins(%expanded_4 : tensor<4x1024x8x64xf32>) outs(%6 : tensor<4x8x1024x64xf32>) permutation = [0, 2, 1, 3]
  %8 = tensor.empty() : tensor<4x8x1024x1024xf32>
  %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<4x8x1024x1024xf32>) -> tensor<4x8x1024x1024xf32>
  %10 = tensor.empty() : tensor<4x8x1024xf32>
  %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<4x8x1024xf32>) -> tensor<4x8x1024xf32>
  %12 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<4x8x1024xf32>) -> tensor<4x8x1024xf32>
  %13 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<4x8x1024x64xf32>) -> tensor<4x8x1024x64xf32>
  %14 = linalg_ext.batch_matmul ins(%transposed, %transposed_5 : tensor<4x8x1024x64xf32>, tensor<4x8x64x1024xf32>) outs(%9 : tensor<4x8x1024x1024xf32>) layout = "nn"
  %15:4 = linalg_ext.softmax dimension(3) ins(%14 : tensor<4x8x1024x1024xf32>) outs(%8, %11, %12, %10 : tensor<4x8x1024x1024xf32>, tensor<4x8x1024xf32>, tensor<4x8x1024xf32>, tensor<4x8x1024xf32>) : tensor<4x8x1024x1024xf32>, tensor<4x8x1024xf32>, tensor<4x8x1024xf32>, tensor<4x8x1024xf32>
  %16 = linalg_ext.batch_matmul ins(%15#0, %transposed_6 : tensor<4x8x1024x1024xf32>, tensor<4x8x1024x64xf32>) outs(%13 : tensor<4x8x1024x64xf32>) layout = "nn"  {__root__}
  return %16 : tensor<4x8x1024x64xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [0, 1, 4, 3, 2], tile_sizes = [2, 4, 0, 0, 0]}
  cleanup
}

// result after transform.structured.fuse_ext

#map = affine_map<(d0) -> (d0 * 64)>

func.func @multihead_attention_with_prologue_proj(%arg0: tensor<4x1024x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<4x8x1024x64xf32> {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %0 = tensor.empty() : tensor<4x1024x512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x1024x512xf32>) -> tensor<4x1024x512xf32>
    %2 = tensor.empty() : tensor<4x8x1024x64xf32>
    %3:5 = scf.for %arg4 = %c0 to %c4 step %c2 iter_args(%arg5 = %2, %arg6 = %0, %arg7 = %1, %arg8 = %1, %arg9 = %0) -> (tensor<4x8x1024x64xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>) {
      %4:5 = scf.for %arg10 = %c0 to %c8 step %c4 iter_args(%arg11 = %arg5, %arg12 = %arg6, %arg13 = %arg7, %arg14 = %arg8, %arg15 = %arg9) -> (tensor<4x8x1024x64xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>) {
        %5 = affine.apply #map(%arg10)
        %extracted_slice = tensor.extract_slice %arg0[%arg4, 0, 0] [2, 1024, 512] [1, 1, 1] : tensor<4x1024x512xf32> to tensor<2x1024x512xf32>
        %extracted_slice_1 = tensor.extract_slice %arg1[0, %5] [512, 256] [1, 1] : tensor<512x512xf32> to tensor<512x256xf32>
        %6 = tensor.empty() : tensor<2x512x256xf32>
        %broadcasted = linalg.broadcast ins(%extracted_slice_1 : tensor<512x256xf32>) outs(%6 : tensor<2x512x256xf32>) dimensions = [0] 
        %7 = tensor.empty() : tensor<2x4x1024x64xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[0, %5] [512, 256] [1, 1] : tensor<512x512xf32> to tensor<512x256xf32>
        %broadcasted_3 = linalg.broadcast ins(%extracted_slice_2 : tensor<512x256xf32>) outs(%6 : tensor<2x512x256xf32>) dimensions = [0] 
        %8 = tensor.empty() : tensor<2x4x64x1024xf32>
        %9 = tensor.empty() : tensor<2x4x1024x1024xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<2x4x1024x1024xf32>) -> tensor<2x4x1024x1024xf32>
        %11 = tensor.empty() : tensor<2x4x1024xf32>
        %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<2x4x1024xf32>) -> tensor<2x4x1024xf32>
        %13 = linalg.fill ins(%cst : f32) outs(%11 : tensor<2x4x1024xf32>) -> tensor<2x4x1024xf32>
        %extracted_slice_4 = tensor.extract_slice %arg3[0, %5] [512, 256] [1, 1] : tensor<512x512xf32> to tensor<512x256xf32>
        %broadcasted_5 = linalg.broadcast ins(%extracted_slice_4 : tensor<512x256xf32>) outs(%6 : tensor<2x512x256xf32>) dimensions = [0] 
        %14 = tensor.empty() : tensor<2x1024x256xf32>
        %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<2x1024x256xf32>) -> tensor<2x1024x256xf32>
        %16 = linalg_ext.batch_matmul ins(%extracted_slice, %broadcasted_5 : tensor<2x1024x512xf32>, tensor<2x512x256xf32>) outs(%15 : tensor<2x1024x256xf32>) layout = "nn" 
        %expanded = tensor.expand_shape %16 [[0], [1], [2, 3]] : tensor<2x1024x256xf32> into tensor<2x1024x4x64xf32>
        %transposed = linalg.transpose ins(%expanded : tensor<2x1024x4x64xf32>) outs(%7 : tensor<2x4x1024x64xf32>) permutation = [0, 2, 1, 3] 
        %17 = linalg.fill ins(%cst : f32) outs(%7 : tensor<2x4x1024x64xf32>) -> tensor<2x4x1024x64xf32>
        %inserted_slice = tensor.insert_slice %16 into %arg12[%arg4, 0, %5] [2, 1024, 256] [1, 1, 1] : tensor<2x1024x256xf32> into tensor<4x1024x512xf32>
        %inserted_slice_6 = tensor.insert_slice %15 into %arg15[%arg4, 0, %5] [2, 1024, 256] [1, 1, 1] : tensor<2x1024x256xf32> into tensor<4x1024x512xf32>
        %18 = linalg_ext.batch_matmul ins(%extracted_slice, %broadcasted : tensor<2x1024x512xf32>, tensor<2x512x256xf32>) outs(%15 : tensor<2x1024x256xf32>) layout = "nn" 
        %expanded_7 = tensor.expand_shape %18 [[0], [1], [2, 3]] : tensor<2x1024x256xf32> into tensor<2x1024x4x64xf32>
        %transposed_8 = linalg.transpose ins(%expanded_7 : tensor<2x1024x4x64xf32>) outs(%7 : tensor<2x4x1024x64xf32>) permutation = [0, 2, 1, 3] 
        %19 = linalg_ext.batch_matmul ins(%extracted_slice, %broadcasted_3 : tensor<2x1024x512xf32>, tensor<2x512x256xf32>) outs(%15 : tensor<2x1024x256xf32>) layout = "nn" 
        %expanded_9 = tensor.expand_shape %19 [[0], [1], [2, 3]] : tensor<2x1024x256xf32> into tensor<2x1024x4x64xf32>
        %transposed_10 = linalg.transpose ins(%expanded_9 : tensor<2x1024x4x64xf32>) outs(%8 : tensor<2x4x64x1024xf32>) permutation = [0, 2, 3, 1] 
        %20 = linalg_ext.batch_matmul ins(%transposed_8, %transposed_10 : tensor<2x4x1024x64xf32>, tensor<2x4x64x1024xf32>) outs(%10 : tensor<2x4x1024x1024xf32>) layout = "nn" 
        %21:4 = linalg_ext.softmax dimension(3) ins(%20 : tensor<2x4x1024x1024xf32>) outs(%9, %12, %13, %11 : tensor<2x4x1024x1024xf32>, tensor<2x4x1024xf32>, tensor<2x4x1024xf32>, tensor<2x4x1024xf32>) : tensor<2x4x1024x1024xf32>, tensor<2x4x1024xf32>, tensor<2x4x1024xf32>, tensor<2x4x1024xf32>
        %22 = linalg_ext.batch_matmul ins(%21#0, %transposed : tensor<2x4x1024x1024xf32>, tensor<2x4x1024x64xf32>) outs(%17 : tensor<2x4x1024x64xf32>) layout = "nn"  {__root__}
        %inserted_slice_11 = tensor.insert_slice %22 into %arg11[%arg4, %arg10, 0, 0] [2, 4, 1024, 64] [1, 1, 1, 1] : tensor<2x4x1024x64xf32> into tensor<4x8x1024x64xf32>
        %inserted_slice_12 = tensor.insert_slice %18 into %arg13[%arg4, 0, %5] [2, 1024, 256] [1, 1, 1] : tensor<2x1024x256xf32> into tensor<4x1024x512xf32>
        %inserted_slice_13 = tensor.insert_slice %19 into %arg14[%arg4, 0, %5] [2, 1024, 256] [1, 1, 1] : tensor<2x1024x256xf32> into tensor<4x1024x512xf32>
        scf.yield %inserted_slice_11, %inserted_slice, %inserted_slice_12, %inserted_slice_13, %inserted_slice_6 : tensor<4x8x1024x64xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>
      }
      scf.yield %4#0, %4#1, %4#2, %4#3, %4#4 : tensor<4x8x1024x64xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>, tensor<4x1024x512xf32>
    }
    return %3#0 : tensor<4x8x1024x64xf32>
  }
```

## Split-head multi-query-attention
see https://arxiv.org/pdf/1911.02150.pdf
```
// input.mlir
// batch size = 4
// sequence length = 1024
// hidden size = 512
// head number = 8
// head dimension = 64
func.func @multiquery_attention_with_prologue_proj(%arg0: tensor<4x1024x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x64xf32>, %arg3: tensor<512x64xf32>) -> tensor<4x8x1024x64xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x512x512xf32>
  %1 = tensor.empty() : tensor<4x512x64xf32>
  %broadcasted = linalg.broadcast ins(%arg1 : tensor<512x512xf32>) outs(%0 : tensor<4x512x512xf32>) dimensions = [0] 
  %2 = tensor.empty() : tensor<4x1024x512xf32>
  %3 = tensor.empty() : tensor<4x1024x64xf32>
  %4 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<4x1024x512xf32>) -> tensor<4x1024x512xf32>
  %5 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
  %6 = linalg_ext.batch_matmul ins(%arg0, %broadcasted : tensor<4x1024x512xf32>, tensor<4x512x512xf32>) outs(%4 : tensor<4x1024x512xf32>) layout = "nn" 
  %broadcasted_1 = linalg.broadcast ins(%arg2 : tensor<512x64xf32>) outs(%1 : tensor<4x512x64xf32>) dimensions = [0] 
  %7 = linalg_ext.batch_matmul ins(%arg0, %broadcasted_1 : tensor<4x1024x512xf32>, tensor<4x512x64xf32>) outs(%5 : tensor<4x1024x64xf32>) layout = "nn" {__stop__}
  %broadcasted_2 = linalg.broadcast ins(%arg3 : tensor<512x64xf32>) outs(%1 : tensor<4x512x64xf32>) dimensions = [0] 
  %8 = linalg_ext.batch_matmul ins(%arg0, %broadcasted_2 : tensor<4x1024x512xf32>, tensor<4x512x64xf32>) outs(%5 : tensor<4x1024x64xf32>) layout = "nn" {__stop__}
  %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] : tensor<4x1024x512xf32> into tensor<4x1024x8x64xf32>
  %9 = tensor.empty() : tensor<4x8x1024x64xf32>
  %10 = tensor.empty() : tensor<4x64x1024xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<4x1024x8x64xf32>) outs(%9 : tensor<4x8x1024x64xf32>) permutation = [0, 2, 1, 3] 
  %transposed_3 = linalg.transpose ins(%7 : tensor<4x1024x64xf32>) outs(%10 : tensor<4x64x1024xf32>) permutation = [0, 2, 1] 
  %11 = tensor.empty() : tensor<4x8x1024x1024xf32>
  %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<4x8x1024x1024xf32>) -> tensor<4x8x1024x1024xf32>
  %13 = tensor.empty() : tensor<4x8x1024xf32>
  %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<4x8x1024xf32>) -> tensor<4x8x1024xf32>
  %15 = linalg.fill ins(%cst_0 : f32) outs(%13 : tensor<4x8x1024xf32>) -> tensor<4x8x1024xf32>
  %16 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<4x8x1024x64xf32>) -> tensor<4x8x1024x64xf32>
  %17 = tensor.empty() : tensor<4x8x64x1024xf32>
  %broadcasted_4 = linalg.broadcast ins(%transposed_3 : tensor<4x64x1024xf32>) outs(%17 : tensor<4x8x64x1024xf32>) dimensions = [1] 
  %broadcasted_5 = linalg.broadcast ins(%8 : tensor<4x1024x64xf32>) outs(%9 : tensor<4x8x1024x64xf32>) dimensions = [1] 
  %18 = linalg_ext.batch_matmul ins(%transposed, %broadcasted_4 : tensor<4x8x1024x64xf32>, tensor<4x8x64x1024xf32>) outs(%12 : tensor<4x8x1024x1024xf32>) layout = "nn" 
  %19:4 = linalg_ext.softmax dimension(3) ins(%18 : tensor<4x8x1024x1024xf32>) outs(%11, %14, %15, %13 : tensor<4x8x1024x1024xf32>, tensor<4x8x1024xf32>, tensor<4x8x1024xf32>, tensor<4x8x1024xf32>) : tensor<4x8x1024x1024xf32>, tensor<4x8x1024xf32>, tensor<4x8x1024xf32>, tensor<4x8x1024xf32>
  %20 = linalg_ext.batch_matmul ins(%19#0, %broadcasted_5 : tensor<4x8x1024x1024xf32>, tensor<4x8x1024x64xf32>) outs(%16 : tensor<4x8x1024x64xf32>) layout = "nn" {__root__}
  return %20 : tensor<4x8x1024x64xf32>
}
transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %stop = transform.structured.match attributes {__stop__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:2 = transform.structured.fuse_ext %0, %stop {tile_interchange = [], tile_sizes = [2, 4, 0, 0, 0]}
  cleanup
}

// result after transform.structured.fuse_ext

#map = affine_map<(d0) -> (d0 * 64)>

func.func @multiquery_attention_with_prologue_proj(%arg0: tensor<4x1024x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x64xf32>, %arg3: tensor<512x64xf32>) -> tensor<4x8x1024x64xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.empty() : tensor<4x512x64xf32>
  %1 = tensor.empty() : tensor<4x1024x512xf32>
  %2 = tensor.empty() : tensor<4x1024x64xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<512x64xf32>) outs(%0 : tensor<4x512x64xf32>) dimensions = [0] 
  %4 = linalg_ext.batch_matmul ins(%arg0, %broadcasted : tensor<4x1024x512xf32>, tensor<4x512x64xf32>) outs(%3 : tensor<4x1024x64xf32>) layout = "nn"  {__stop__}
  %broadcasted_1 = linalg.broadcast ins(%arg3 : tensor<512x64xf32>) outs(%0 : tensor<4x512x64xf32>) dimensions = [0] 
  %5 = linalg_ext.batch_matmul ins(%arg0, %broadcasted_1 : tensor<4x1024x512xf32>, tensor<4x512x64xf32>) outs(%3 : tensor<4x1024x64xf32>) layout = "nn"  {__stop__}
  %6 = tensor.empty() : tensor<4x8x1024x64xf32>
  %7 = tensor.empty() : tensor<4x64x1024xf32>
  %8:3 = scf.for %arg4 = %c0 to %c4 step %c2 iter_args(%arg5 = %6, %arg6 = %7, %arg7 = %1) -> (tensor<4x8x1024x64xf32>, tensor<4x64x1024xf32>, tensor<4x1024x512xf32>) {
    %9:3 = scf.for %arg8 = %c0 to %c8 step %c4 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %arg7) -> (tensor<4x8x1024x64xf32>, tensor<4x64x1024xf32>, tensor<4x1024x512xf32>) {
      %10 = affine.apply #map(%arg8)
      %extracted_slice = tensor.extract_slice %arg0[%arg4, 0, 0] [2, 1024, 512] [1, 1, 1] : tensor<4x1024x512xf32> to tensor<2x1024x512xf32>
      %extracted_slice_2 = tensor.extract_slice %arg1[0, %10] [512, 256] [1, 1] : tensor<512x512xf32> to tensor<512x256xf32>
      %11 = tensor.empty() : tensor<2x512x256xf32>
      %broadcasted_3 = linalg.broadcast ins(%extracted_slice_2 : tensor<512x256xf32>) outs(%11 : tensor<2x512x256xf32>) dimensions = [0] 
      %12 = tensor.empty() : tensor<2x1024x256xf32>
      %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<2x1024x256xf32>) -> tensor<2x1024x256xf32>
      %14 = linalg_ext.batch_matmul ins(%extracted_slice, %broadcasted_3 : tensor<2x1024x512xf32>, tensor<2x512x256xf32>) outs(%13 : tensor<2x1024x256xf32>) layout = "nn" 
      %expanded = tensor.expand_shape %14 [[0], [1], [2, 3]] : tensor<2x1024x256xf32> into tensor<2x1024x4x64xf32>
      %15 = tensor.empty() : tensor<2x4x1024x64xf32>
      %transposed = linalg.transpose ins(%expanded : tensor<2x1024x4x64xf32>) outs(%15 : tensor<2x4x1024x64xf32>) permutation = [0, 2, 1, 3] 
      %extracted_slice_4 = tensor.extract_slice %4[%arg4, 0, 0] [2, 1024, 64] [1, 1, 1] : tensor<4x1024x64xf32> to tensor<2x1024x64xf32>
      %extracted_slice_5 = tensor.extract_slice %arg10[%arg4, 0, 0] [2, 64, 1024] [1, 1, 1] : tensor<4x64x1024xf32> to tensor<2x64x1024xf32>
      %transposed_6 = linalg.transpose ins(%extracted_slice_4 : tensor<2x1024x64xf32>) outs(%extracted_slice_5 : tensor<2x64x1024xf32>) permutation = [0, 2, 1] 
      %16 = tensor.empty() : tensor<2x4x64x1024xf32>
      %broadcasted_7 = linalg.broadcast ins(%transposed_6 : tensor<2x64x1024xf32>) outs(%16 : tensor<2x4x64x1024xf32>) dimensions = [1] 
      %17 = tensor.empty() : tensor<2x4x1024x1024xf32>
      %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<2x4x1024x1024xf32>) -> tensor<2x4x1024x1024xf32>
      %19 = linalg_ext.batch_matmul ins(%transposed, %broadcasted_7 : tensor<2x4x1024x64xf32>, tensor<2x4x64x1024xf32>) outs(%18 : tensor<2x4x1024x1024xf32>) layout = "nn" 
      %20 = tensor.empty() : tensor<2x4x1024xf32>
      %21 = linalg.fill ins(%cst_0 : f32) outs(%20 : tensor<2x4x1024xf32>) -> tensor<2x4x1024xf32>
      %22 = linalg.fill ins(%cst : f32) outs(%20 : tensor<2x4x1024xf32>) -> tensor<2x4x1024xf32>
      %23:4 = linalg_ext.softmax dimension(3) ins(%19 : tensor<2x4x1024x1024xf32>) outs(%17, %21, %22, %20 : tensor<2x4x1024x1024xf32>, tensor<2x4x1024xf32>, tensor<2x4x1024xf32>, tensor<2x4x1024xf32>) : tensor<2x4x1024x1024xf32>, tensor<2x4x1024xf32>, tensor<2x4x1024xf32>, tensor<2x4x1024xf32>
      %extracted_slice_8 = tensor.extract_slice %5[%arg4, 0, 0] [2, 1024, 64] [1, 1, 1] : tensor<4x1024x64xf32> to tensor<2x1024x64xf32>
      %broadcasted_9 = linalg.broadcast ins(%extracted_slice_8 : tensor<2x1024x64xf32>) outs(%15 : tensor<2x4x1024x64xf32>) dimensions = [1] 
      %24 = linalg.fill ins(%cst : f32) outs(%15 : tensor<2x4x1024x64xf32>) -> tensor<2x4x1024x64xf32>
      %25 = linalg_ext.batch_matmul ins(%23#0, %broadcasted_9 : tensor<2x4x1024x1024xf32>, tensor<2x4x1024x64xf32>) outs(%24 : tensor<2x4x1024x64xf32>) layout = "nn"  {__root__}
      %inserted_slice = tensor.insert_slice %25 into %arg9[%arg4, %arg8, 0, 0] [2, 4, 1024, 64] [1, 1, 1, 1] : tensor<2x4x1024x64xf32> into tensor<4x8x1024x64xf32>
      %inserted_slice_10 = tensor.insert_slice %transposed_6 into %arg10[%arg4, 0, 0] [2, 64, 1024] [1, 1, 1] : tensor<2x64x1024xf32> into tensor<4x64x1024xf32>
      %inserted_slice_11 = tensor.insert_slice %14 into %arg11[%arg4, 0, %10] [2, 1024, 256] [1, 1, 1] : tensor<2x1024x256xf32> into tensor<4x1024x512xf32>
      scf.yield %inserted_slice, %inserted_slice_10, %inserted_slice_11 : tensor<4x8x1024x64xf32>, tensor<4x64x1024xf32>, tensor<4x1024x512xf32>
    }
    scf.yield %9#0, %9#1, %9#2 : tensor<4x8x1024x64xf32>, tensor<4x64x1024xf32>, tensor<4x1024x512xf32>
  }
  return %8#0 : tensor<4x8x1024x64xf32>
}

```
