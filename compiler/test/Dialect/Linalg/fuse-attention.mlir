// RUN: byteir-opt %s --transform-dialect-interpreter --canonicalize-ext --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fuse_dot_attention
func.func @fuse_dot_attention(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
// CHECK-DAG: linalg.fill
// CHECK-DAG: linalg.fill
// CHECK-DAG: linalg.fill
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         linalg.fill
// CHECK:         linalg.matmul
// CHECK:         linalg_ext.softmax
// CHECK:         linalg_ext.diag
// CHECK:         linalg.fill
// CHECK:         linalg.matmul
// CHECK:         linalg.matmul
// CHECK:         scf.yield
// CHECK:       }
// CHECK:       scf.yield
// CHECK:     }
  %0 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = tensor.empty() : tensor<1024x512xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %6 = tensor.empty() : tensor<1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_4 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_0 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %fill_2 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %fill_5 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1024xf32>) -> tensor<1024xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%fill_0: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>
  %7:4 = linalg_ext.softmax dimension(1) 
    ins(%1 : tensor<1024x512xf32>) outs(%3, %fill_4, %fill_5, %6 : tensor<1024x512xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x512xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>  
  %8 = linalg.matmul {__root__} ins(%7#0, %arg2: tensor<1024x512xf32>, tensor<512x32xf32>)
                     outs(%fill_2: tensor<1024x32xf32>)
    -> tensor<1024x32xf32>
  return %8: tensor<1024x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 0, 8], tile_interchange = [2, 1, 0]}
  transform.structured.tile_loop_hint %1 
}

// -----

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
// CHECK-LABEL: func.func @fuse_multihead_attention_tile_3d
// CHECK-DAG: linalg.fill
// CHECK-DAG: linalg.fill
// CHECK-DAG: linalg.fill
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           tensor.empty
// CHECK:           linalg.fill
// CHECK:           linalg_ext.batch_matmul
// CHECK:           tensor.empty
// CHECK:           linalg_ext.softmax
// CHECK:           tensor.empty
// CHECK:           linalg_ext.diag
// CHECK:           tensor.empty
// CHECK:           linalg.fill
// CHECK:           linalg_ext.batch_matmul
// CHECK:           linalg_ext.batch_matmul
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         scf.yield
// CHECK:       }
// CHECK:       scf.yield

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:3 = transform.structured.fuse_ext %0 {tile_sizes = [2, 0, 8, 0, 4], tile_interchange = [0, 1, 4, 3, 2]}
  transform.structured.tile_loop_hint %1 
}


// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0
  %transformed, %loops:4 = transform.structured.fuse_ext %0 {tile_interchange = [0, 1, 4, 3, 2], tile_sizes = [32, 2, 4, 0, 8]}
}

// support split-head attention, see https://arxiv.org/abs/1909.08053
func.func @multi_head_attention_with_prologue_tile_head(%arg0: tensor<128x1024x256xf32>, %arg1: tensor<128x1024x256xf32>, %arg2: tensor<128x1024x256xf32>, %arg3: tensor<128x1024x1024xi1>) -> tensor<128x16x1024x256xf32> {
  %cst = arith.constant -1.000000e+09 : f32
  %cst_0 = arith.constant 1.600000e+01 : f32
  %cst_1 = arith.constant 0xFF800000 : f32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %cst_3 = arith.constant dense_resource<__elided__> : tensor<4096x256xf32>
  %cst_4 = arith.constant dense_resource<__elided__> : tensor<4096xf32>
  %0 = tensor.empty() : tensor<256x4096xf32>
  %transposed = linalg.transpose ins(%cst_3 : tensor<4096x256xf32>) outs(%0 : tensor<256x4096xf32>) permutation = [1, 0] 
  %1 = tensor.empty() : tensor<128x256x4096xf32>
  %broadcasted = linalg.broadcast ins(%transposed : tensor<256x4096xf32>) outs(%1 : tensor<128x256x4096xf32>) dimensions = [0] 
  %2 = tensor.empty() : tensor<128x1024x4096xf32>
  %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %4 = linalg.batch_matmul ins(%arg0, %broadcasted : tensor<128x1024x256xf32>, tensor<128x256x4096xf32>) outs(%3 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %5 = tensor.empty() : tensor<128x1024x4096xf32>
  %broadcasted_5 = linalg.broadcast ins(%cst_4 : tensor<4096xf32>) outs(%5 : tensor<128x1024x4096xf32>) dimensions = [0, 1] 
  %expanded = tensor.expand_shape %4 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %expanded_6 = tensor.expand_shape %broadcasted_5 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %6 = tensor.empty() : tensor<128x1024x16x256xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded, %expanded_6 : tensor<128x1024x16x256xf32>, tensor<128x1024x16x256xf32>) outs(%6 : tensor<128x1024x16x256xf32>) {
  ^bb0(%in: f32, %in_23: f32, %out: f32):
    %43 = arith.addf %in, %in_23 : f32
    linalg.yield %43 : f32
  } -> tensor<128x1024x16x256xf32>
  %8 = tensor.empty() : tensor<128x16x1024x256xf32>
  %transposed_7 = linalg.transpose ins(%7 : tensor<128x1024x16x256xf32>) outs(%8 : tensor<128x16x1024x256xf32>) permutation = [0, 2, 1, 3] 
  %9 = tensor.empty() : tensor<256x4096xf32>
  %transposed_8 = linalg.transpose ins(%cst_3 : tensor<4096x256xf32>) outs(%9 : tensor<256x4096xf32>) permutation = [1, 0] 
  %10 = tensor.empty() : tensor<128x256x4096xf32>
  %broadcasted_9 = linalg.broadcast ins(%transposed_8 : tensor<256x4096xf32>) outs(%10 : tensor<128x256x4096xf32>) dimensions = [0] 
  %11 = tensor.empty() : tensor<128x1024x4096xf32>
  %12 = linalg.fill ins(%cst_2 : f32) outs(%11 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %13 = linalg.batch_matmul ins(%arg1, %broadcasted_9 : tensor<128x1024x256xf32>, tensor<128x256x4096xf32>) outs(%12 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %14 = tensor.empty() : tensor<128x1024x4096xf32>
  %broadcasted_10 = linalg.broadcast ins(%cst_4 : tensor<4096xf32>) outs(%14 : tensor<128x1024x4096xf32>) dimensions = [0, 1] 
  %expanded_11 = tensor.expand_shape %13 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %expanded_12 = tensor.expand_shape %broadcasted_10 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %15 = tensor.empty() : tensor<128x1024x16x256xf32>
  %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_11, %expanded_12 : tensor<128x1024x16x256xf32>, tensor<128x1024x16x256xf32>) outs(%15 : tensor<128x1024x16x256xf32>) {
  ^bb0(%in: f32, %in_23: f32, %out: f32):
    %43 = arith.addf %in, %in_23 : f32
    linalg.yield %43 : f32
  } -> tensor<128x1024x16x256xf32>
  %17 = tensor.empty() : tensor<256x4096xf32>
  %transposed_13 = linalg.transpose ins(%cst_3 : tensor<4096x256xf32>) outs(%17 : tensor<256x4096xf32>) permutation = [1, 0] 
  %18 = tensor.empty() : tensor<128x256x4096xf32>
  %broadcasted_14 = linalg.broadcast ins(%transposed_13 : tensor<256x4096xf32>) outs(%18 : tensor<128x256x4096xf32>) dimensions = [0] 
  %19 = tensor.empty() : tensor<128x1024x4096xf32>
  %20 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %21 = linalg.batch_matmul ins(%arg2, %broadcasted_14 : tensor<128x1024x256xf32>, tensor<128x256x4096xf32>) outs(%20 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %22 = tensor.empty() : tensor<128x1024x4096xf32>
  %broadcasted_15 = linalg.broadcast ins(%cst_4 : tensor<4096xf32>) outs(%22 : tensor<128x1024x4096xf32>) dimensions = [0, 1] 
  %expanded_16 = tensor.expand_shape %21 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %expanded_17 = tensor.expand_shape %broadcasted_15 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %23 = tensor.empty() : tensor<128x1024x16x256xf32>
  %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_16, %expanded_17 : tensor<128x1024x16x256xf32>, tensor<128x1024x16x256xf32>) outs(%23 : tensor<128x1024x16x256xf32>) {
  ^bb0(%in: f32, %in_23: f32, %out: f32):
    %43 = arith.addf %in, %in_23 : f32
    linalg.yield %43 : f32
  } -> tensor<128x1024x16x256xf32>
  %25 = tensor.empty() : tensor<128x16x1024x256xf32>
  %transposed_18 = linalg.transpose ins(%24 : tensor<128x1024x16x256xf32>) outs(%25 : tensor<128x16x1024x256xf32>) permutation = [0, 2, 1, 3] 
  %expanded_19 = tensor.expand_shape %arg3 [[0, 1], [2, 3, 4], [5, 6]] : tensor<128x1024x1024xi1> into tensor<1x128x1x1x1024x1x1024xi1>
  %26 = tensor.empty() : tensor<1x128x16x1x1x1024x1x1024xi1>
  %broadcasted_20 = linalg.broadcast ins(%expanded_19 : tensor<1x128x1x1x1024x1x1024xi1>) outs(%26 : tensor<1x128x16x1x1x1024x1x1024xi1>) dimensions = [2] 
  %27 = tensor.empty() : tensor<128x16x256x1024xf32>
  %transposed_21 = linalg.transpose ins(%16 : tensor<128x1024x16x256xf32>) outs(%27 : tensor<128x16x256x1024xf32>) permutation = [0, 2, 3, 1] 
  %28 = tensor.empty() : tensor<128x16x1024x1024xf32>
  %29 = linalg.fill ins(%cst_2 : f32) outs(%28 : tensor<128x16x1024x1024xf32>) -> tensor<128x16x1024x1024xf32>
  %30 = linalg_ext.batch_matmul ins(%transposed_7, %transposed_21 : tensor<128x16x1024x256xf32>, tensor<128x16x256x1024xf32>) outs(%29 : tensor<128x16x1024x1024xf32>) layout = "nn" 
  %expanded_22 = tensor.expand_shape %30 [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<128x16x1024x1024xf32> into tensor<1x128x16x1x1x1024x1x1024xf32>
  %31 = tensor.empty() : tensor<1x128x16x1x1x1024x1x1024xf32>
  %32 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%broadcasted_20, %expanded_22 : tensor<1x128x16x1x1x1024x1x1024xi1>, tensor<1x128x16x1x1x1024x1x1024xf32>) outs(%31 : tensor<1x128x16x1x1x1024x1x1024xf32>) {
  ^bb0(%in: i1, %in_23: f32, %out: f32):
    %43 = arith.divf %in_23, %cst_0 : f32
    %44 = arith.select %in, %cst, %43 : f32
    linalg.yield %44 : f32
  } -> tensor<1x128x16x1x1x1024x1x1024xf32>
  %collapsed = tensor.collapse_shape %32 [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x128x16x1x1x1024x1x1024xf32> into tensor<128x16x1024x1024xf32>
  %33 = tensor.empty() : tensor<128x16x1024x1024xf32>
  %34 = tensor.empty() : tensor<128x16x1024xf32>
  %35 = tensor.empty() : tensor<128x16x1024xf32>
  %36 = tensor.empty() : tensor<128x16x1024xf32>
  %37 = linalg.fill ins(%cst_1 : f32) outs(%34 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %38 = linalg.fill ins(%cst_2 : f32) outs(%35 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %39:4 = linalg_ext.softmax dimension(3) ins(%collapsed : tensor<128x16x1024x1024xf32>) outs(%33, %37, %38, %36 : tensor<128x16x1024x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) : tensor<128x16x1024x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
  %40 = tensor.empty() : tensor<128x16x1024x256xf32>
  %41 = linalg.fill ins(%cst_2 : f32) outs(%40 : tensor<128x16x1024x256xf32>) -> tensor<128x16x1024x256xf32>
  %42 = linalg_ext.batch_matmul ins(%39#0, %transposed_18 : tensor<128x16x1024x1024xf32>, tensor<128x16x1024x256xf32>) outs(%41 : tensor<128x16x1024x256xf32>) layout = "nn"  {__root__}
  return %42 : tensor<128x16x1024x256xf32>
}
// CHECK-LABEL: func.func @multi_head_attention_with_prologue_tile_head
// CHECK: scf.for
// CHECK:   linalg.batch_matmul
// CHECK:   linalg.batch_matmul
// CHECK:   linalg_ext.batch_matmul
// CHECK:   linalg_ext.softmax
// CHECK:   linalg_ext.batch_matmul
// CHECK:   scf.yield

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0
  %transformed, %loops:4 = transform.structured.fuse_ext %0 {tile_interchange = [0, 1, 4, 3, 2], tile_sizes = [32, 2, 4, 0, 8]}
}

func.func @multi_head_attention_with_prologue_tile_4d(%arg0: tensor<128x1024x256xf32>, %arg1: tensor<128x1024x256xf32>, %arg2: tensor<128x1024x256xf32>, %arg3: tensor<128x1024x1024xi1>) -> tensor<128x16x1024x256xf32> {
  %cst = arith.constant -1.000000e+09 : f32
  %cst_0 = arith.constant 1.600000e+01 : f32
  %cst_1 = arith.constant 0xFF800000 : f32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %cst_3 = arith.constant dense_resource<__elided__> : tensor<4096x256xf32>
  %cst_4 = arith.constant dense_resource<__elided__> : tensor<4096xf32>
  %0 = tensor.empty() : tensor<256x4096xf32>
  %transposed = linalg.transpose ins(%cst_3 : tensor<4096x256xf32>) outs(%0 : tensor<256x4096xf32>) permutation = [1, 0] 
  %1 = tensor.empty() : tensor<128x256x4096xf32>
  %broadcasted = linalg.broadcast ins(%transposed : tensor<256x4096xf32>) outs(%1 : tensor<128x256x4096xf32>) dimensions = [0] 
  %2 = tensor.empty() : tensor<128x1024x4096xf32>
  %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %4 = linalg.batch_matmul ins(%arg0, %broadcasted : tensor<128x1024x256xf32>, tensor<128x256x4096xf32>) outs(%3 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %5 = tensor.empty() : tensor<128x1024x4096xf32>
  %broadcasted_5 = linalg.broadcast ins(%cst_4 : tensor<4096xf32>) outs(%5 : tensor<128x1024x4096xf32>) dimensions = [0, 1] 
  %expanded = tensor.expand_shape %4 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %expanded_6 = tensor.expand_shape %broadcasted_5 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %6 = tensor.empty() : tensor<128x1024x16x256xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded, %expanded_6 : tensor<128x1024x16x256xf32>, tensor<128x1024x16x256xf32>) outs(%6 : tensor<128x1024x16x256xf32>) {
  ^bb0(%in: f32, %in_23: f32, %out: f32):
    %43 = arith.addf %in, %in_23 : f32
    linalg.yield %43 : f32
  } -> tensor<128x1024x16x256xf32>
  %8 = tensor.empty() : tensor<128x16x1024x256xf32>
  %transposed_7 = linalg.transpose ins(%7 : tensor<128x1024x16x256xf32>) outs(%8 : tensor<128x16x1024x256xf32>) permutation = [0, 2, 1, 3] 
  %9 = tensor.empty() : tensor<256x4096xf32>
  %transposed_8 = linalg.transpose ins(%cst_3 : tensor<4096x256xf32>) outs(%9 : tensor<256x4096xf32>) permutation = [1, 0] 
  %10 = tensor.empty() : tensor<128x256x4096xf32>
  %broadcasted_9 = linalg.broadcast ins(%transposed_8 : tensor<256x4096xf32>) outs(%10 : tensor<128x256x4096xf32>) dimensions = [0] 
  %11 = tensor.empty() : tensor<128x1024x4096xf32>
  %12 = linalg.fill ins(%cst_2 : f32) outs(%11 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %13 = linalg.batch_matmul ins(%arg1, %broadcasted_9 : tensor<128x1024x256xf32>, tensor<128x256x4096xf32>) outs(%12 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %14 = tensor.empty() : tensor<128x1024x4096xf32>
  %broadcasted_10 = linalg.broadcast ins(%cst_4 : tensor<4096xf32>) outs(%14 : tensor<128x1024x4096xf32>) dimensions = [0, 1] 
  %expanded_11 = tensor.expand_shape %13 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %expanded_12 = tensor.expand_shape %broadcasted_10 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %15 = tensor.empty() : tensor<128x1024x16x256xf32>
  %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_11, %expanded_12 : tensor<128x1024x16x256xf32>, tensor<128x1024x16x256xf32>) outs(%15 : tensor<128x1024x16x256xf32>) {
  ^bb0(%in: f32, %in_23: f32, %out: f32):
    %43 = arith.addf %in, %in_23 : f32
    linalg.yield %43 : f32
  } -> tensor<128x1024x16x256xf32>
  %17 = tensor.empty() : tensor<256x4096xf32>
  %transposed_13 = linalg.transpose ins(%cst_3 : tensor<4096x256xf32>) outs(%17 : tensor<256x4096xf32>) permutation = [1, 0] 
  %18 = tensor.empty() : tensor<128x256x4096xf32>
  %broadcasted_14 = linalg.broadcast ins(%transposed_13 : tensor<256x4096xf32>) outs(%18 : tensor<128x256x4096xf32>) dimensions = [0] 
  %19 = tensor.empty() : tensor<128x1024x4096xf32>
  %20 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %21 = linalg.batch_matmul ins(%arg2, %broadcasted_14 : tensor<128x1024x256xf32>, tensor<128x256x4096xf32>) outs(%20 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %22 = tensor.empty() : tensor<128x1024x4096xf32>
  %broadcasted_15 = linalg.broadcast ins(%cst_4 : tensor<4096xf32>) outs(%22 : tensor<128x1024x4096xf32>) dimensions = [0, 1] 
  %expanded_16 = tensor.expand_shape %21 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %expanded_17 = tensor.expand_shape %broadcasted_15 [[0], [1], [2, 3]] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %23 = tensor.empty() : tensor<128x1024x16x256xf32>
  %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_16, %expanded_17 : tensor<128x1024x16x256xf32>, tensor<128x1024x16x256xf32>) outs(%23 : tensor<128x1024x16x256xf32>) {
  ^bb0(%in: f32, %in_23: f32, %out: f32):
    %43 = arith.addf %in, %in_23 : f32
    linalg.yield %43 : f32
  } -> tensor<128x1024x16x256xf32>
  %25 = tensor.empty() : tensor<128x16x1024x256xf32>
  %transposed_18 = linalg.transpose ins(%24 : tensor<128x1024x16x256xf32>) outs(%25 : tensor<128x16x1024x256xf32>) permutation = [0, 2, 1, 3] 
  %expanded_19 = tensor.expand_shape %arg3 [[0, 1], [2, 3, 4], [5, 6]] : tensor<128x1024x1024xi1> into tensor<1x128x1x1x1024x1x1024xi1>
  %26 = tensor.empty() : tensor<1x128x16x1x1x1024x1x1024xi1>
  %broadcasted_20 = linalg.broadcast ins(%expanded_19 : tensor<1x128x1x1x1024x1x1024xi1>) outs(%26 : tensor<1x128x16x1x1x1024x1x1024xi1>) dimensions = [2] 
  %27 = tensor.empty() : tensor<128x16x256x1024xf32>
  %transposed_21 = linalg.transpose ins(%16 : tensor<128x1024x16x256xf32>) outs(%27 : tensor<128x16x256x1024xf32>) permutation = [0, 2, 3, 1] 
  %28 = tensor.empty() : tensor<128x16x1024x1024xf32>
  %29 = linalg.fill ins(%cst_2 : f32) outs(%28 : tensor<128x16x1024x1024xf32>) -> tensor<128x16x1024x1024xf32>
  %30 = linalg_ext.batch_matmul ins(%transposed_7, %transposed_21 : tensor<128x16x1024x256xf32>, tensor<128x16x256x1024xf32>) outs(%29 : tensor<128x16x1024x1024xf32>) layout = "nn" 
  %expanded_22 = tensor.expand_shape %30 [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<128x16x1024x1024xf32> into tensor<1x128x16x1x1x1024x1x1024xf32>
  %31 = tensor.empty() : tensor<1x128x16x1x1x1024x1x1024xf32>
  %32 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%broadcasted_20, %expanded_22 : tensor<1x128x16x1x1x1024x1x1024xi1>, tensor<1x128x16x1x1x1024x1x1024xf32>) outs(%31 : tensor<1x128x16x1x1x1024x1x1024xf32>) {
  ^bb0(%in: i1, %in_23: f32, %out: f32):
    %43 = arith.divf %in_23, %cst_0 : f32
    %44 = arith.select %in, %cst, %43 : f32
    linalg.yield %44 : f32
  } -> tensor<1x128x16x1x1x1024x1x1024xf32>
  %collapsed = tensor.collapse_shape %32 [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x128x16x1x1x1024x1x1024xf32> into tensor<128x16x1024x1024xf32>
  %33 = tensor.empty() : tensor<128x16x1024x1024xf32>
  %34 = tensor.empty() : tensor<128x16x1024xf32>
  %35 = tensor.empty() : tensor<128x16x1024xf32>
  %36 = tensor.empty() : tensor<128x16x1024xf32>
  %37 = linalg.fill ins(%cst_1 : f32) outs(%34 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %38 = linalg.fill ins(%cst_2 : f32) outs(%35 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %39:4 = linalg_ext.softmax dimension(3) ins(%collapsed : tensor<128x16x1024x1024xf32>) outs(%33, %37, %38, %36 : tensor<128x16x1024x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) : tensor<128x16x1024x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
  %40 = tensor.empty() : tensor<128x16x1024x256xf32>
  %41 = linalg.fill ins(%cst_2 : f32) outs(%40 : tensor<128x16x1024x256xf32>) -> tensor<128x16x1024x256xf32>
  %42 = linalg_ext.batch_matmul ins(%39#0, %transposed_18 : tensor<128x16x1024x1024xf32>, tensor<128x16x1024x256xf32>) outs(%41 : tensor<128x16x1024x256xf32>) layout = "nn"  {__root__}
  return %42 : tensor<128x16x1024x256xf32>
}
// CHECK-LABEL: func.func @multi_head_attention_with_prologue_tile_4d
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         linalg.batch_matmul
// CHECK:         linalg.batch_matmul
// CHECK:         linalg_ext.batch_matmul
// CHECK:         linalg_ext.softmax
// CHECK:         linalg_ext.diag
// CHECK:         linalg_ext.batch_matmul
// CHECK:         linalg_ext.batch_matmul
// CHECK:         scf.yield
// CHECK:       scf.yield
// CHECK:     scf.yield
// CHECK:   scf.yield
