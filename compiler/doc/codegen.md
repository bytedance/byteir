# Codegen pipeline

## hlo-opt

This pass pipeline is mainly used for clustering fusion group on mhlo dialect, each fusion group was expected to fused into a single kernel in later codegen pipeline and would be outlined as a indepedent kernel function.

- `ReductionFusionPass` reduction fusion in producer direction

- `ElementFusionPass` elementwise/broadcast/collapse_shape/expand_shape/etc. producer-consumer bi-directional fusion

- `FusionOutliningPass` fusion group outlining

## linalg-tensor-opt

### reduction codegen transformations

```
  func.func private @Unknown0(%arg0: tensor<8192x50257xf16>) -> tensor<50257xf32> attributes {__byteir_reduction_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.convert %arg0 : (tensor<8192x50257xf16>) -> tensor<8192x50257xf32>
    %2 = mhlo.reduce(%1 init: %0) across dimensions = [0] : (tensor<8192x50257xf32>, tensor<f32>) -> tensor<50257xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %3 = mhlo.add %arg1, %arg2 : tensor<f32>
      mhlo.return %3 : tensor<f32>
    }
    return %2 : tensor<50257xf32>
  }
```

This pass pipeline first convert outlined mhlo fusion group into linalg dialect and try to fuse linalg op with its producer/consumer.

- `createLinalgElementwiseFusionExtPass` linalg fusion pass with our extension, see [linalg pass](linalg.md) for more details

```
func.func private @Unknown0(%arg0: tensor<8192x50257xf16>) -> tensor<50257xf32> attributes {__byteir_reduction_fusion__} {
	%cst = arith.constant 0.000000e+00 : f32
	%0 = tensor.empty() : tensor<50257xf32>
	%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<50257xf32>) -> tensor<50257xf32>
	%2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<8192x50257xf16>) outs(%1 : tensor<50257xf32>) {
	^bb0(%in: f16, %out: f32):
		%3 = arith.extf %in : f16 to f32
		%4 = arith.addf %out, %3 : f32
		linalg.yield %4 : f32
	} -> tensor<50257xf32>
	return %2 : tensor<50257xf32>
}
```

[optional] Split grid-level reduction on `reduction` dimensions

```
func.func private @Unknown0(%arg0: tensor<8192x50257xf16>) -> tensor<50257xf32> attributes {__byteir_reduction_fusion__} {
	%cst = arith.constant 0.000000e+00 : f32
	%0 = tensor.empty() : tensor<50257xf32>
	%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<50257xf32>) -> tensor<50257xf32>
	%expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<8192x50257xf16> into tensor<32x256x50257xf16>
	%2 = tensor.empty() : tensor<32x50257xf32>
	%3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x50257xf32>) -> tensor<32x50257xf32>
	%4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded : tensor<32x256x50257xf16>) outs(%3 : tensor<32x50257xf32>) attrs =  {__grid_reduction__} {
	^bb0(%in: f16, %out: f32):
		%6 = arith.extf %in : f16 to f32
		%7 = arith.addf %out, %6 : f32
		linalg.yield %7 : f32
	} -> tensor<32x50257xf32>
	%5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel"]} ins(%4 : tensor<32x50257xf32>) outs(%1 : tensor<50257xf32>) attrs =  {__grid_reduction__} {
	^bb0(%in: f32, %out: f32):
		%6 = arith.addf %in, %out : f32
		linalg.yield %6 : f32
	} -> tensor<50257xf32>
	return %5 : tensor<50257xf32>
}
```

- Tiling reduction on `parallel` dimension and mapping tiled reductions to thread blocks

```
func.func private @Unknown0(%arg0: tensor<8192x50257xf16>) -> tensor<50257xf32> attributes {__byteir_reduction_fusion__} {
	%cst = arith.constant 0.000000e+00 : f32
	%0 = tensor.empty() : tensor<50257xf32>
	%expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<8192x50257xf16> into tensor<32x256x50257xf16>
	%1 = tensor.empty() : tensor<32x50257xf32>
	%2 = scf.forall (%arg1, %arg2) in (32, 1571) shared_outs(%arg3 = %1) -> (tensor<32x50257xf32>) {
		%4 = affine.min #map(%arg2)
		%5 = affine.apply #map1(%arg2)
		%extracted_slice = tensor.extract_slice %expanded[%arg1, 0, %5] [1, 256, %4] [1, 1, 1] : tensor<32x256x50257xf16> to tensor<256x?xf16>
		%extracted_slice_0 = tensor.extract_slice %arg3[%arg1, %5] [1, %4] [1, 1] : tensor<32x50257xf32> to tensor<?xf32>
		%6 = linalg.fill ins(%cst : f32) outs(%extracted_slice_0 : tensor<?xf32>) -> tensor<?xf32>
		%7 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<256x?xf16>) outs(%6 : tensor<?xf32>) {
		^bb0(%in: f16, %out: f32):
			%8 = arith.extf %in : f16 to f32
			%9 = arith.addf %out, %8 : f32
			linalg.yield %9 : f32
		} -> tensor<?xf32>
		scf.forall.in_parallel {
			tensor.parallel_insert_slice %7 into %arg3[%arg1, %5] [1, %4] [1, 1] : tensor<?xf32> into tensor<32x50257xf32>
		}
	} {mapping = [#gpu.block<y>, #gpu.block<x>]}
	%3 = scf.forall (%arg1) in (1571) shared_outs(%arg2 = %0) -> (tensor<50257xf32>) {
		// ...
	} {mapping = [#gpu.block<x>]}
	return %3 : tensor<50257xf32>
}
```

- Block-level reduction codegen

```
%2 = scf.forall (%arg1, %arg2) in (32, 1571) shared_outs(%arg3 = %1) -> (tensor<32x50257xf32>) {
	%4 = affine.min #map(%arg2)
	%5 = affine.apply #map1(%arg2)
	%extracted_slice = tensor.extract_slice %expanded[%arg1, 0, %5] [1, 256, %4] [1, 1, 1] : tensor<32x256x50257xf16> to tensor<256x?xf16>
	%6 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf32>
	%7 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x32xf32>
	%8 = scf.forall (%arg4, %arg5) in (16, 32) shared_outs(%arg6 = %7) -> (tensor<16x32xf32>) {
		%17 = affine.min #map2(%arg4)
		%18 = affine.min #map3(%arg4)
		%19 = affine.apply #map4(%18, %17)
		%20 = affine.min #map5(%arg5, %arg2)
		%21 = affine.min #map6(%arg5, %arg2)
		%22 = affine.apply #map4(%21, %20)
		%23 = affine.apply #map7(%21, %20)
		%extracted_slice_6 = tensor.extract_slice %extracted_slice[%17, %20] [%19, %22] [1, 1] : tensor<256x?xf16> to tensor<?x?xf16>
		%padded = tensor.pad %extracted_slice_6 low[0, 0] high[0, %23] {
		^bb0(%arg7: index, %arg8: index):
			tensor.yield %cst : f16
		} : tensor<?x?xf16> to tensor<16x1xf16>
		%extracted_slice_7 = tensor.extract_slice %arg6[%arg4, %arg5] [1, 1] [1, 1] : tensor<16x32xf32> to tensor<f32>
		%collapsed = tensor.collapse_shape %padded [[0, 1]] : tensor<16x1xf16> into tensor<16xf16>
		%24 = linalg.fill ins(%cst_0 : f32) outs(%extracted_slice_7 : tensor<f32>) -> tensor<f32>
		%25 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["reduction"]} ins(%collapsed : tensor<16xf16>) outs(%24 : tensor<f32>) {
		^bb0(%in: f16, %out: f32):
			%26 = arith.extf %in : f16 to f32
			%27 = arith.addf %out, %26 : f32
			linalg.yield %27 : f32
		} -> tensor<f32>
		scf.forall.in_parallel {
			tensor.parallel_insert_slice %25 into %arg6[%arg4, %arg5] [1, 1] [1, 1] : tensor<f32> into tensor<16x32xf32>
		}
	} {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
	%expanded_1 = tensor.expand_shape %8 [[0, 1], [2]] : tensor<16x32xf32> into tensor<8x2x32xf32>
	%9 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<8x32xf32>
	%10 = scf.forall (%arg4, %arg5) in (8, 32) shared_outs(%arg6 = %9) -> (tensor<8x32xf32>) {
		// ...
	} {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
	%expanded_2 = tensor.expand_shape %10 [[0, 1], [2]] : tensor<8x32xf32> into tensor<4x2x32xf32>
	%11 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4x32xf32>
	%12 = scf.forall (%arg4, %arg5) in (4, 32) shared_outs(%arg6 = %11) -> (tensor<4x32xf32>) {
		// ...
	} {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
	%expanded_3 = tensor.expand_shape %12 [[0, 1], [2]] : tensor<4x32xf32> into tensor<2x2x32xf32>
	%13 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2x32xf32>
	%14 = scf.forall (%arg4, %arg5) in (2, 32) shared_outs(%arg6 = %13) -> (tensor<2x32xf32>) {
		// ...
	} {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
	%15 = scf.forall (%arg4) in (32) shared_outs(%arg5 = %6) -> (tensor<32xf32>) {
		// ...
	} {mapping = [#gpu.thread<x>]}
	%extracted_slice_4 = tensor.extract_slice %15[0] [%4] [1] : tensor<32xf32> to tensor<?xf32>
	%extracted_slice_5 = tensor.extract_slice %arg3[%arg1, %5] [1, %4] [1, 1] : tensor<32x50257xf32> to tensor<?xf32>
	%16 = scf.forall (%arg4) in (512) shared_outs(%arg5 = %extracted_slice_5) -> (tensor<?xf32>) {
		// ...
	} {mapping = [#gpu.linear<x>]}
	scf.forall.in_parallel {
		tensor.parallel_insert_slice %16 into %arg3[%arg1, %5] [1, %4] [1, 1] : tensor<?xf32> into tensor<32x50257xf32>
	}
} {mapping = [#gpu.block<y>, #gpu.block<x>]}
```

- Detensorize scalar linalg ops to arith ops and specialize `tensor.pad`

```
%2 = scf.forall (%arg1, %arg2) in (32, 1571) shared_outs(%arg3 = %1) -> (tensor<32x50257xf32>) {
	%4 = affine.min #map(%arg2)
	%5 = affine.apply #map1(%arg2)
	%6 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32xf32>
	%7 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x32xf32>
	%8 = scf.forall (%arg4, %arg5) in (16, 32) shared_outs(%arg6 = %7) -> (tensor<16x32xf32>) {
		%17 = affine.min #map2(%arg5, %arg2)
		%18 = affine.min #map3(%arg5, %arg2)
		%19 = affine.apply #map4(%18, %17)
		%20 = arith.cmpi ugt, %19, %c0 : index
		%21 = scf.if %20 -> (f16) {
			%84 = affine.apply #map5(%arg4)
			%85 = affine.apply #map6(%arg2)[%17]
			%extracted = tensor.extract %expanded[%arg1, %84, %85] : tensor<32x256x50257xf16>
			scf.yield %extracted : f16
		} else {
			scf.yield %cst : f16
		}
		// ...
		%78 = arith.extf %77 : f16 to f32
		%79 = arith.addf %75, %78 : f32
		%80 = arith.cmpi ugt, %19, %c0 : index
		%81 = scf.if %80 -> (f16) {
			%84 = affine.apply #map21(%arg4)
			%85 = affine.apply #map6(%arg2)[%17]
			%extracted = tensor.extract %expanded[%arg1, %84, %85] : tensor<32x256x50257xf16>
			scf.yield %extracted : f16
		} else {
			scf.yield %cst : f16
		}
		%82 = arith.extf %81 : f16 to f32
		%83 = arith.addf %79, %82 : f32
		%extracted_slice_5 = tensor.extract_slice %arg6[%arg4, %arg5] [1, 1] [1, 1] : tensor<16x32xf32> to tensor<f32>
		%inserted = tensor.insert %83 into %extracted_slice_5[] : tensor<f32>
		scf.forall.in_parallel {
			tensor.parallel_insert_slice %inserted into %arg6[%arg4, %arg5] [1, 1] [1, 1] : tensor<f32> into tensor<16x32xf32>
		}
	} {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
	
	// ...
	%extracted_slice = tensor.extract_slice %15[0] [%4] [1] : tensor<32xf32> to tensor<?xf32>
	%extracted_slice_4 = tensor.extract_slice %arg3[%arg1, %5] [1, %4] [1, 1] : tensor<32x50257xf32> to tensor<?xf32>
	%16 = scf.forall (%arg4) in (512) shared_outs(%arg5 = %extracted_slice_4) -> (tensor<?xf32>) {
		%17 = affine.min #map22(%arg4)[%4]
		%18 = affine.max #map23(%17)
		%19 = affine.apply #map24(%arg4)[%4]
		%extracted_slice_5 = tensor.extract_slice %extracted_slice[%19] [%18] [1] : tensor<?xf32> to tensor<?xf32>
		%extracted_slice_6 = tensor.extract_slice %arg5[%19] [%18] [1] : tensor<?xf32> to tensor<?xf32>
		%20 = linalg.copy {__byteir_gpu_tile_block_reduction_10} ins(%extracted_slice_5 : tensor<?xf32>) outs(%extracted_slice_6 : tensor<?xf32>) -> tensor<?xf32>
		scf.forall.in_parallel {
			tensor.parallel_insert_slice %20 into %arg5[%19] [%18] [1] : tensor<?xf32> into tensor<?xf32>
		}
	} {mapping = [#gpu.linear<x>]}
	scf.forall.in_parallel {
		tensor.parallel_insert_slice %16 into %arg3[%arg1, %5] [1, %4] [1, 1] : tensor<?xf32> into tensor<32x50257xf32>
	}
} {mapping = [#gpu.block<y>, #gpu.block<x>]}
```

- `structured.split_reduction` split reduction op along `reduction` dimension for increasing parallelism

- `structured.tile_to_forall_op` tile reduction op along `parallel` dimensions to `forall` op and mapping to block/linear/thread

- `structured.fuse_into_containing_op` fuse init and pad operands into `scf.forall`

- `structured.annotate` attach any attribute to target ops, used to annotate reduction op and attach memory space to `allot_tensor`

- `structured.tile` tile reduction op along `reduction` dimension to sequential for loop

- `structured.detensorize` use to inline computation region of linalg op which operands have scalar tensor type

- `LinalgCollapseLoopsPass` collapse consecutive `parallel` and `reduction` loops, this pass could work on both tensor and memref

- `TensorPadSpecializationPass` specialize `tensor.extract` of pad op to conditional read
