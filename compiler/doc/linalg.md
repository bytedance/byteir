# ByteIR Linalg Extension

ByteIR compiler extends the MLIR linalg dialect to support several non-trivial patterns.
ByteIR implements in a way of introducing a linalg-ext dialect on top of the existing linalg dialect.
Ops and transformations in linalg-ext are expected to work interchangeably with existing ones in linalg, and expected to eventually be upstreamed to LLVM.

## Rationales

### Need of non-trivial patterns for linalg

Several performance-critical patterns are still not covered well in the upstream linalg.
Some of those patterns might not be easily expressible in the linalg dialect either through generic ops or even only relying on existing linalg interfaces. Top-k and Scan (cumsum) might belong to this category.

Some might be expressible, through composing several generic ops, but might obstruct desired transformations due to lack of proper interfaces. Softmax belongs to this category.

Some aims to be a more versatile alternative to the existing upstream versions, `linalg_ext.batch_matmul` belongs to this category.

### Implementation of introducing linalg-ext

Introducing linalg-ext can provide several benefits as follows,

- it clearly separate the extension of ops or transformations from the existing linalg, avoiding misusing.
- it can intuitively resolve the patterns that require introducing interfaces.

## Transformation Extension

Several transformations are enhanced or introduced in ByteIR linalg-ext.

**_Collapse dims transformation_** is introduced

- to collapse dimensions of linalg.generic operation

**_Fold unit-extent dims transformation_** is introduced

- to remove unit-extent dimension in Linalg ops on tensors

**_Lower to loops transformation_** is introduced

- to lower the operations to loops.

**_Linalg outline transformation_** is introduced

- outline the linalg operation to a named function, where `libcall` for the external
  library call. If `libcall` was set to False, every outlined function would have a
  unique name , in this situation `func_name` just gives a naming hint. Otherwise,
  all transformed function calls refer to the same external function named `func_name`.

**_Tile label transformation_** is introduced

- to indicate loop type (parallel or reduction) through attributes.

Note this tile label transformation also work with existing linalg tile and fuse transformation.

**_Shared output to distributed style transformation_** is introduced

- convert parallel tiling from shared output style to distributed style.

**_Tile transformation_** is enhanced

- to support linalg-ext ops.

**_Fuse transformation_** is enhanced

- to support linalg-ext ops,
- to correctly support tiling along a reduction axis,
- to support intermediates as outputs within a fusion,
- to support intermediate tensor dim simplification.
- to support diamond structure
- to support optional stop attribute
- to support fusing together with tensor dialect

**_Fuse operands transformation_** is introduced

- to support multiple roots in fusion
- to support check whether all the operations in the func op are fused together

Note this transformation will be merged together with fuse transformation.

Here shows the difference when tiling along a reduction axis.

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

Here shows the difference when there is an intermediate as as output.

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

Here shows the difference when tiling and fusing the resnet block.
BTW, the upstream version runs very slowly when performing on a squence of
resnet blocks because some nodes are visited 2^N time, N is number of blocks.

```
// input.mlir
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @resnet_block(%arg0: tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16> {
  %cst = arith.constant dense_resource<__elided__> : tensor<256x1x1x64xf32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<64x3x3x64xf32>
  %cst_1 = arith.constant dense_resource<__elided__> : tensor<64x1x1x256xf32>
  %cst_2 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1x56x56x256xf16>
  %1 = linalg.fill ins(%cst_2 : f16) outs(%0 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %2 = linalg.elemwise_unary {__revisited__} ins(%arg0 : tensor<1x56x56x256xf16>) outs(%1 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %3 = tensor.empty() : tensor<1x56x56x64xf16>
  %4 = linalg.fill ins(%cst_2 : f16) outs(%3 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %5 = linalg.conv_2d_nhwc_fhwc {__conv_0__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%2, %cst_1 : tensor<1x56x56x256xf16>, tensor<64x1x1x256xf32>) outs(%4 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %padded = tensor.pad %5 nofold low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_2 : f16
  } : tensor<1x56x56x64xf16> to tensor<1x58x58x64xf16>
  %6 = tensor.empty() : tensor<1x56x56x64xf16>
  %7 = linalg.fill ins(%cst_2 : f16) outs(%6 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %8 = linalg.conv_2d_nhwc_fhwc {__conv_1__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %cst_0 : tensor<1x58x58x64xf16>, tensor<64x3x3x64xf32>) outs(%7 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %9 = tensor.empty() : tensor<1x56x56x256xf16>
  %10 = linalg.fill ins(%cst_2 : f16) outs(%9 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %11 = linalg.conv_2d_nhwc_fhwc {__conv_2__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%8, %cst : tensor<1x56x56x64xf16>, tensor<256x1x1x64xf32>) outs(%10 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %12 = tensor.empty() : tensor<1x56x56x256xf16>
  %13 = linalg.generic {__root__, indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %11 : tensor<1x56x56x256xf16>, tensor<1x56x56x256xf16>) outs(%12 : tensor<1x56x56x256xf16>) {
  ^bb0(%in: f16, %in_3: f16, %out: f16):
    %14 = arith.addf %in, %in_3 : f16
    linalg.yield %14 : f16
  } -> tensor<1x56x56x256xf16>
  return %13 : tensor<1x56x56x256xf16>
}
transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [0, 8, 0, 32]}
  cleanup
}

// result after transform.structured.fuse_ext, `linalg.elemwise_unary {__revisited__}` is tiled only once
// and its tile size is calculated by getting the maximum of two paths

#map = affine_map<(d0) -> (-d0 + 1, 0)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (56, d0)>
#map3 = affine_map<(d0) -> (56, d0 + 9)>
#map4 = affine_map<(d0, d1) -> (d0 - d1)>
#map5 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 10)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @resnet_block(%arg0: tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16> {
  %c256 = arith.constant 256 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<64x1x1x256xf32>
  %cst_1 = arith.constant dense_resource<__elided__> : tensor<64x3x3x64xf32>
  %cst_2 = arith.constant dense_resource<__elided__> : tensor<256x1x1x64xf32>
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %0 = tensor.empty() : tensor<1x56x56x256xf16>
  %1 = tensor.empty() : tensor<1x56x56x64xf16>
  %2:7 = scf.for %arg1 = %c0 to %c56 step %c8 iter_args(%arg2 = %0, %arg3 = %1, %arg4 = %1, %arg5 = %1, %arg6 = %0, %arg7 = %1, %arg8 = %0) -> (tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>) {
    %3:7 = scf.for %arg9 = %c0 to %c256 step %c32 iter_args(%arg10 = %arg2, %arg11 = %arg3, %arg12 = %arg4, %arg13 = %arg5, %arg14 = %arg6, %arg15 = %arg7, %arg16 = %arg8) -> (tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>) {
      %4 = arith.addi %arg1, %c8 : index
      %5 = arith.addi %arg9, %c32 : index
      %6 = arith.minsi %arg9, %c0 : index
      %7 = arith.maxsi %5, %c256 : index
      %8 = arith.subi %7, %6 : index
      %9 = arith.subi %arg9, %6 : index
      %10 = arith.subi %c0, %6 : index
      %11 = affine.max #map(%arg1)
      %12 = affine.max #map1(%arg1)
      %13 = affine.min #map2(%12)
      %14 = affine.min #map3(%arg1)
      %15 = affine.apply #map4(%14, %13)
      %16 = affine.apply #map5(%11, %14, %13)
      %extracted_slice = tensor.extract_slice %arg13[0, %13, 0, 0] [1, %15, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf16> to tensor<1x?x56x64xf16>
      %17 = linalg.fill ins(%cst : f16) outs(%extracted_slice : tensor<1x?x56x64xf16>) -> tensor<1x?x56x64xf16>
      %extracted_slice_3 = tensor.extract_slice %arg11[0, %arg1, 0, 0] [1, 8, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf16> to tensor<1x8x56x64xf16>
      %18 = linalg.fill ins(%cst : f16) outs(%extracted_slice_3 : tensor<1x8x56x64xf16>) -> tensor<1x8x56x64xf16>
      %extracted_slice_4 = tensor.extract_slice %cst_2[%arg9, 0, 0, 0] [32, 1, 1, 64] [1, 1, 1, 1] : tensor<256x1x1x64xf32> to tensor<32x1x1x64xf32>
      %19 = tensor.empty() : tensor<1x8x56x32xf16>
      %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<1x8x56x32xf16>) -> tensor<1x8x56x32xf16>
      %inserted_slice = tensor.insert_slice %18 into %arg12[0, %arg1, 0, 0] [1, 8, 56, 64] [1, 1, 1, 1] : tensor<1x8x56x64xf16> into tensor<1x56x56x64xf16>
      %inserted_slice_5 = tensor.insert_slice %17 into %arg15[0, %13, 0, 0] [1, %15, 56, 64] [1, 1, 1, 1] : tensor<1x?x56x64xf16> into tensor<1x56x56x64xf16>
      %21 = arith.minsi %arg1, %13 : index
      %22 = arith.addi %13, %15 : index
      %23 = arith.maxsi %4, %22 : index
      %24 = arith.subi %23, %21 : index
      %extracted_slice_6 = tensor.extract_slice %arg0[0, %21, 0, %6] [1, %24, 56, %8] [1, 1, 1, 1] : tensor<1x56x56x256xf16> to tensor<1x?x56x?xf16>
      %extracted_slice_7 = tensor.extract_slice %arg14[0, %21, 0, %6] [1, %24, 56, %8] [1, 1, 1, 1] : tensor<1x56x56x256xf16> to tensor<1x?x56x?xf16>
      %25 = linalg.fill ins(%cst : f16) outs(%extracted_slice_7 : tensor<1x?x56x?xf16>) -> tensor<1x?x56x?xf16>
      %26 = linalg.elemwise_unary {__revisited__} ins(%extracted_slice_6 : tensor<1x?x56x?xf16>) outs(%25 : tensor<1x?x56x?xf16>) -> tensor<1x?x56x?xf16>
      %27 = arith.subi %arg1, %21 : index
      %extracted_slice_8 = tensor.extract_slice %26[0, %27, 0, %9] [1, 8, 56, 32] [1, 1, 1, 1] : tensor<1x?x56x?xf16> to tensor<1x8x56x32xf16>
      %28 = arith.subi %13, %21 : index
      %extracted_slice_9 = tensor.extract_slice %26[0, %28, 0, %10] [1, %15, 56, 256] [1, 1, 1, 1] : tensor<1x?x56x?xf16> to tensor<1x?x56x256xf16>
      %29 = linalg.conv_2d_nhwc_fhwc {__conv_0__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%extracted_slice_9, %cst_0 : tensor<1x?x56x256xf16>, tensor<64x1x1x256xf32>) outs(%17 : tensor<1x?x56x64xf16>) -> tensor<1x?x56x64xf16>
      %padded = tensor.pad %29 nofold low[0, %11, 1, 0] high[0, %16, 1, 0] {
      ^bb0(%arg17: index, %arg18: index, %arg19: index, %arg20: index):
        tensor.yield %cst : f16
      } : tensor<1x?x56x64xf16> to tensor<1x10x58x64xf16>
      %30 = linalg.conv_2d_nhwc_fhwc {__conv_1__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %cst_1 : tensor<1x10x58x64xf16>, tensor<64x3x3x64xf32>) outs(%18 : tensor<1x8x56x64xf16>) -> tensor<1x8x56x64xf16>
      %31 = linalg.conv_2d_nhwc_fhwc {__conv_2__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%30, %extracted_slice_4 : tensor<1x8x56x64xf16>, tensor<32x1x1x64xf32>) outs(%20 : tensor<1x8x56x32xf16>) -> tensor<1x8x56x32xf16>
      %32 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_8, %31 : tensor<1x8x56x32xf16>, tensor<1x8x56x32xf16>) outs(%19 : tensor<1x8x56x32xf16>) attrs =  {__root__} {
      ^bb0(%in: f16, %in_15: f16, %out: f16):
        %33 = arith.addf %in, %in_15 : f16
        linalg.yield %33 : f16
      } -> tensor<1x8x56x32xf16>
      %inserted_slice_10 = tensor.insert_slice %32 into %arg10[0, %arg1, 0, %arg9] [1, 8, 56, 32] [1, 1, 1, 1] : tensor<1x8x56x32xf16> into tensor<1x56x56x256xf16>
      %inserted_slice_11 = tensor.insert_slice %30 into %arg11[0, %arg1, 0, 0] [1, 8, 56, 64] [1, 1, 1, 1] : tensor<1x8x56x64xf16> into tensor<1x56x56x64xf16>
      %inserted_slice_12 = tensor.insert_slice %29 into %arg13[0, %13, 0, 0] [1, %15, 56, 64] [1, 1, 1, 1] : tensor<1x?x56x64xf16> into tensor<1x56x56x64xf16>
      %inserted_slice_13 = tensor.insert_slice %26 into %arg14[0, %21, 0, %6] [1, %24, 56, %8] [1, 1, 1, 1] : tensor<1x?x56x?xf16> into tensor<1x56x56x256xf16>
      %inserted_slice_14 = tensor.insert_slice %25 into %arg16[0, %21, 0, %6] [1, %24, 56, %8] [1, 1, 1, 1] : tensor<1x?x56x?xf16> into tensor<1x56x56x256xf16>
      scf.yield %inserted_slice_10, %inserted_slice_11, %inserted_slice, %inserted_slice_12, %inserted_slice_13, %inserted_slice_5, %inserted_slice_14 : tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>
    }
    scf.yield %3#0, %3#1, %3#2, %3#3, %3#4, %3#5, %3#6 : tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>
  }
  return %2#0 : tensor<1x56x56x256xf16>
}

```

Please refer [attention examples](attention.md) to understand the tiling and fusion capabilities of the FuseExt Op.

**_Elementwise fusion transformation_** is enhanced

- to support intermediates as outputs within a fusion,
- to support both producer-consumer fusion and input-sharing fusion,
- to support intermediate tensor dim simplification,
- to support map fusion by automatically converting map ops to generic ops.
- to support indexing maps with constant result

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

Here shows the difference bewteen support of indexing maps with constant result.

```
// input.mlir
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @constant_in_affine_map_with_collapse_shape(%arg0: tensor<1x256x1024xf32>, %arg1: tensor<256x1024xf16>, %arg2: tensor<256x1xf32>, %arg3: tensor<256x1xf32>) -> tensor<256x1024xf32> {
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x1024xf16> into tensor<1x256x1024xf16>
  %0 = tensor.empty() : tensor<1x256x1024xf32>
  %1 = tensor.empty() : tensor<256x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0 : tensor<1x256x1024xf16>, tensor<1x256x1024xf32>) outs(%0 : tensor<1x256x1024xf32>) {
  ^bb0(%in: f16, %in_0: f32, %out: f32):
    %4 = arith.extf %in : f16 to f32
    %5 = arith.addf %in_0, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x256x1024xf32>
  %collapsed = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<1x256x1024xf32> into tensor<256x1024xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2, %collapsed : tensor<256x1xf32>, tensor<256x1xf32>, tensor<256x1024xf32>) outs(%1 : tensor<256x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %4 = arith.subf %in_1, %in_0 : f32
    %5 = arith.mulf %4, %in : f32
    linalg.yield %5 : f32
  } -> tensor<256x1024xf32>
  return %3 : tensor<256x1024xf32>
}

// result after linalg-fuse-elementwise-ops, no fusion
func.func @constant_in_affine_map_with_collapse_shape(%arg0: tensor<1x256x1024xf32>, %arg1: tensor<256x1024xf16>, %arg2: tensor<256x1xf32>, %arg3: tensor<256x1xf32>) -> tensor<256x1024xf32> {
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x1024xf16> into tensor<1x256x1024xf16>
  %0 = tensor.empty() : tensor<1x256x1024xf32>
  %1 = tensor.empty() : tensor<256x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0 : tensor<1x256x1024xf16>, tensor<1x256x1024xf32>) outs(%0 : tensor<1x256x1024xf32>) {
  ^bb0(%in: f16, %in_0: f32, %out: f32):
    %4 = arith.extf %in : f16 to f32
    %5 = arith.addf %in_0, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x256x1024xf32>
  %collapsed = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<1x256x1024xf32> into tensor<256x1024xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2, %collapsed : tensor<256x1xf32>, tensor<256x1xf32>, tensor<256x1024xf32>) outs(%1 : tensor<256x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %4 = arith.subf %in_1, %in_0 : f32
    %5 = arith.mulf %4, %in : f32
    linalg.yield %5 : f32
  } -> tensor<256x1024xf32>
  return %3 : tensor<256x1024xf32>
}

// result after linalg-fuse-elementwise-ext, perfect fusion
func.func @constant_in_affine_map_with_collapse_shape(%arg0: tensor<1x256x1024xf32>, %arg1: tensor<256x1024xf16>, %arg2: tensor<256x1xf32>, %arg3: tensor<256x1xf32>) -> tensor<256x1024xf32> {
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x1024xf16> into tensor<1x256x1024xf16>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1], [2]] : tensor<256x1xf32> into tensor<1x256x1xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1], [2]] : tensor<256x1xf32> into tensor<1x256x1xf32>
  %0 = tensor.empty() : tensor<1x256x1024xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0, %expanded_0, %expanded_1 : tensor<1x256x1024xf16>, tensor<1x256x1024xf32>, tensor<1x256x1xf32>, tensor<1x256x1xf32>) outs(%0 : tensor<1x256x1024xf32>) {
  ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
    %2 = arith.extf %in : f16 to f32
    %3 = arith.addf %in_2, %2 : f32
    %4 = arith.subf %3, %in_4 : f32
    %5 = arith.mulf %4, %in_3 : f32
    linalg.yield %5 : f32
  } -> tensor<1x256x1024xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<1x256x1024xf32> into tensor<256x1024xf32>
  return %collapsed : tensor<256x1024xf32>
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

### Scatter Op

Linalg-ext scatter op is introduced to present a scatter pattern
It is a structured op.

Spec:

- Operands:
  - indices: Tensor
  - updates: Tensor
- Inits/Results:
  - src: Tensor

Here, the first `rank(indices) - 1` dimensions of `indices` and `update` are compatible.
The last `rank(update) - rank(indices) + 1` dimensions of `update` and `src` are compatible.
The last dimension of the indices, denoted as `dim(indices, rank(indices) - 1)`, should be static and the rank of `src` is equal to `dim(indices, rank(indices) - 1) + rank(update) - rank(indices) + 1`

### Softmax Op

Linalg-ext softmax op is introduced to present a softmax pattern
It is a structured op.

Spec:

- Operands:
  - input: Tensor with a dim of N
- Attrs
  - dimension: I64ArrayAttr
- Inits/Results:
  - output: Tensor with a dim of N, `output_result = exp(input - max_result) / accumulator_result`
  - max: Tensor with a dim of N - 1, `max_result = max(max(input, dimension), max_init)`
  - accumulator: Tensor with a dim of N - 1, `accumulator_result = accumulator_init * exp(max_init - max_result) + sum(exp(input - max_result), dimension)`
  - scale: Tensor with a dim of N - 1, `scale_result = accumulator_init * exp(max_init - max_result) / accumulator_result`

Here, Operand `1`, max is defined as `max_result = max(max(input, dimension), max_init)`.
Basically, it is a `reduce_max` of `input` along a dimension `dimension` with a initial value `max_init`.
Operand `2`, accumulator is defined as `accumulator_result = accumulator_init * exp(max_init - max_result) + sum(exp(input - max_result), dimension)`
Basically, it is a `reduce_sum` of `exp(input - max_result)` along a dimension `dimension` with a initial value `accumulator_init * exp(max_init - max_result)`.
Operand `3`, scale is defined as `scale_result = accumulator_init * exp(max_init - max_result) / accumulator_result`.
Finally, Operand `0`, output is defined as `output_result = exp(input - max_result) / accumulator_result`.

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
