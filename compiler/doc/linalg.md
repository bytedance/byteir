# ByteIR Linalg Extension

ByteIR compiler extends the MLIR linalg dialect to support several non-trivial patterns.
ByteIR implements in a way of introducing a linalg-ext dialect on top of the existing linalg dialect.
Ops and transformations in linalg-ext are expected to work interchangeably with existing ones in linalg, and expected to eventually be upstreamed to LLVM.

## Rationales
### Need of non-trivial patterns for linalg

Several performance-critical patterns are still not covered well in the upstream linalg.
Some of those patterns might not be easily expressible in the linalg dialect either through generic ops or even only relying on existing linalg interfaces. Top-k and Scan (cumsum) might belong to this category. 

Some might be expressible, through composing several generic ops, but might obstruct desired transformations due to lack of proper interfaces. Softmax belongs to this category.


### Implementation of introducing linalg-ext

Introducing linalg-ext can provide several benefits as follows,
* it clearly separate the extension of ops or transformations from the existing linalg, avoiding misusing.
* it can intuitively resolve the patterns that require introducing interfaces.


## Transformation Extension

Several transformations are enhanced or introduced in ByteIR linalg-ext. 

***Tile label transformation*** is introduced 
* to indicate loop type (parallel or reduction) through attributes.

Note this tile label transformation also work with existing linalg tile and fuse transformation.

***Tile transformation*** is enhanced 
* to support linalg-ext ops.

***Fuse transformation*** is enhanced
* to support linalg-ext ops,
* to correctly support tiling along a reduction axis, 
* to support intermediates as outputs within a fusion,
* to support intermediate tensor dim simplification.

Here shows the difference when there is an intermediate as as output.
```
// input.mlir
func.func @tiled_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %2 : tensor<128x128xf32>
}

// result after transform.structured.fuse, wrong tiling result
func.func @tile_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = scf.for %arg2 = %c0 to %c128 step %c8 iter_args(%arg3 = %0) -> (tensor<128x128xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[0, %arg2] [128, 8] [1, 1] : tensor<128x128xf32> to tensor<128x8xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, 0] [8, 128] [1, 1] : tensor<128x128xf32> to tensor<8x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%arg3 : tensor<128x128xf32>) -> tensor<128x128xf32>   // shouldn't fill to zero every step
    %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<128x8xf32>, tensor<8x128xf32>) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %3 : tensor<128x128xf32>
  }
  return %1 : tensor<128x128xf32>
}

// result after transform.structured.fuse_ext
func.func @tile_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %2 = scf.for %arg2 = %c0 to %c128 step %c8 iter_args(%arg3 = %1) -> (tensor<128x128xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[0, %arg2] [128, 8] [1, 1] : tensor<128x128xf32> to tensor<128x8xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, 0] [8, 128] [1, 1] : tensor<128x128xf32> to tensor<8x128xf32>
    %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<128x8xf32>, tensor<8x128xf32>) outs(%arg3 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %3 : tensor<128x128xf32>
  }
  return %2 : tensor<128x128xf32>
}

```

Here shows the difference when tiling along a reduction axis.
```
// input.mlir
func.func @fuse_element(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<512x128xf32>)
                             outs(%arg1: tensor<512x128xf32>) -> tensor<512x128xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<512x128xf32>, tensor<512x128xf32>)
                             outs(%arg1: tensor<512x128xf32>) -> tensor<512x128xf32>
  return %0, %1 : tensor<512x128xf32>, tensor<512x128xf32>
}

// result after transform.structured.fuse 
func.func @fuse_element_static(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %0 = linalg.elemwise_unary ...  // duplicate producer elemwise_unary
  %1 = scf.for %arg2 = %c0 to %c512 step %c32 iter_args(%arg3 = %arg1) -> (tensor<512x128xf32>) {
    %2 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %arg3) -> (tensor<512x128xf32>) {
      ...
      %3 = linalg.elemwise_unary ...  // producer fusion
      ...
      %4 = linalg.elemwise_binary ...
      %inserted_slice = tensor.insert_slice ...
      scf.yield %inserted_slice : tensor<512x128xf32>
    }
    scf.yield %2 : tensor<512x128xf32>
  }
  return %0, %1 : tensor<512x128xf32>, tensor<512x128xf32>
}

// result after transform.structured.fuse_ext
func.func @fuse_element_static(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %0:2 = scf.for %arg2 = %c0 to %c512 step %c32 iter_args(%arg3 = %arg1, %arg4 = %arg1) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
    %1:2 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %arg1, %arg7 = %arg1) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
      ...
      %2 = linalg.elemwise_unary ...  // producer fusion
      ...
      %3 = linalg.elemwise_binary ...
      %inserted_slice = tensor.insert_slice ...
      %inserted_slice_3 = tensor.insert_slice ...
      scf.yield %inserted_slice, %inserted_slice_3 : tensor<512x128xf32>, tensor<512x128xf32>
    }
    scf.yield %1#0, %1#1 : tensor<512x128xf32>, tensor<512x128xf32>
  }
  return %0#1, %0#0 : tensor<512x128xf32>, tensor<512x128xf32>
}
```

***Elementwise fusion transformation*** is enhanced
* to support intermediates as outputs within a fusion,
* to support both producer-consumer fusion and input-sharing fusion,
* to support intermediate tensor dim simplification,
* to support map fusion by automatically converting map ops to generic ops.


Here shows the difference when there is an input-sharing fusion
```
// input.mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.addf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ops, unchanged
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.addf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ext="shared-input"
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %in, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
```

Here shows the difference bewteen support of intermediate tensor dim simplification..
```
// input.mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @may_more_break_outs_dependency(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.mulf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ops, no fusion
func.func @may_more_break_outs_dependency(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %in : f32
    linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  %dim_1 = tensor.dim %1, %c0 : tensor<?x?xf32>
  %dim_2 = tensor.dim %1, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%dim_1, %dim_2) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.mulf %in, %in : f32
    linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ext, perfect fusion
func.func @may_more_break_outs_dependency(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    %3 = arith.mulf %2, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

```

## Linalg-ext Op Extension
### Alias Op
Linalg-ext alias op is introduced as an auxiliary op to help input-sharing fusion. 
It happens internally within a pass, and typically in a generic op. 
It would not be eliminated in a canonicalizer, and it is only removed through `populateRemoveLinalgExtAliasPattern`.
Note Alias op is not a structured op, and has no interface like `LoopIteratorType`. 

### Diag Op
Linalg-ext diag op is introduced to present a diagonal matrix.
It is a structured op, but currently only used as an output IR, often operating with a matmul.
Depending on backends, a matmul with a diag op typically can be rewritten into
1. a matmul with a reduced-load matrix
2. a sparse matmul
3. an elementwise mul with a broadcast

Spec:
- Operands:
  - input: Tensor with a shape of N
- Inits/Results:
  - output: Tensor with a shape of N x N

### Scan Op
Linalg-ext scan op is introduced to present a scan, prefix sum, or `cumsum` pattern
It is a structured op.

Spec:
- Operands:
  - input: Tensor with a dim of N
- Attrs
  - dimension: I64ArrayAttr
  - inclusivie: BoolAttr
- Inits/Results:
  - output: Tensor with a dim of N
  - accumulator: Tensor with a dim of N - 1


### Softmax Op
Linalg-ext softmax op is introduced to present a softmax pattern
It is a structured op.

Spec:
- Operands:
  - input: Tensor with a dim of N
- Attrs
  - dimension: I64ArrayAttr
- Inits/Results:
  - output: Tensor with a dim of N, ```output_result = exp(input - max_result) / accumulator_result```
  - max: Tensor with a dim of N - 1, ```max_result = max(max(input, dimension), max_init)```
  - accumulator: Tensor with a dim of N - 1, ```accumulator_result = accumulator_init * exp(max_init - max_result) + sum(exp(input - max_result), dimension)```
  - scale: Tensor with a dim of N - 1, ```scale_result = accumulator_init * exp(max_init - max_result) / accumulator_result```

Here, Operand `1`, max is defined as ```max_result = max(max(input, dimension), max_init)```. 
Basically, it is a `reduce_max` of `input` along a dimension `dimension` with a initial value `max_init`.
Operand `2`, accumulator is defined as ```accumulator_result = accumulator_init * exp(max_init - max_result) + sum(exp(input - max_result), dimension)```
Basically, it is a `reduce_sum` of `exp(input - max_result)` along a dimension `dimension` with a initial value `accumulator_init * exp(max_init - max_result)`.
Operand `3`, scale is defined as ```scale_result = accumulator_init * exp(max_init - max_result) / accumulator_result```.
Finally, Operand `0`, output is defined as ```output_result = exp(input - max_result) / accumulator_result```. 

#### Flash attention example
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

### Topk Op
Linalg-ext topk op is introduced to present a topk pattern
It is a structured op.

Spec:
- Operands:
  - input_values: Tensor with a dim of N
  - input_indices: Optional Tensor with a dim of N
- Attrs
  - dimension: I64ArrayAttr
- Inits/Results:
  - output_values: Tensor with a dim of N
  - output_indices: Tensor with a dim of N
