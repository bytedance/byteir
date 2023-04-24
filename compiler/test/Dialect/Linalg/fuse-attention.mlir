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
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 0, 8], tile_interchange = [2, 1, 0]}
  transform.structured.tile_loop_hint %1 
}

// -----

// CHECK-LABEL: func.func @fuse_multihead_attention_tile_3d
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
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.fuse_ext %0 {tile_sizes = [2, 0, 8, 0, 4], tile_interchange = [0, 1, 4, 3, 2]}
  transform.structured.tile_loop_hint %1 
}

// -----

// support split-head attention, see https://arxiv.org/abs/1909.08053
// CHECK-LABEL: func.func @multihead_attention_with_prologue_proj
func.func @multihead_attention_with_prologue_proj(%arg0: tensor<4x1024x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<4x8x1024x64xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK-DAG: linalg.broadcast
// CHECK-DAG: linalg_ext.batch_matmul
// CHECK-DAG: linalg.broadcast
// CHECK-DAG: linalg_ext.batch_matmul
// CHECK-DAG: linalg.broadcast
// CHECK-DAG: linalg_ext.batch_matmul
// CHECK-DAG: tensor.expand_shape
// CHECK-DAG: tensor.expand_shape
// CHECK-DAG: tensor.expand_shape
// CHECK-DAG: linalg.transpose
// CHECK-DAG: linalg.transpose
// CHECK-DAG: linalg.transpose
// CHECK-DAG: linalg_ext.batch_matmul
// CHECK-DAG: linalg_ext.softmax
// CHECK-DAG: linalg_ext.batch_matmul
// CHECK:     scf.yield
// CHECK:   scf.yield
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
}
