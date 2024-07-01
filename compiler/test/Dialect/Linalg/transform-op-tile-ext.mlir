// RUN: byteir-opt --transform-dialect-interpreter --split-input-file %s | FileCheck %s

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.tile_ext %0 [2, 4, 8] {interchange = [2, 1, 0]}
}

// CHECK-LABEL: func @tile_linalg_matmul(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<2x8xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<8x4xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<2x4xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<2x8xf32>, tensor<8x4xf32>)
// CHECK-SAME:                                   outs(%[[sTC]] : tensor<2x4xf32>)  -> tensor<2x4xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<2x4xf32> into tensor<128x128xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

//      CHECK: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.tile_ext %0 [4] 
}

func.func @softmax_tensor(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4:4 = linalg_ext.softmax
    dimension(1) 
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %1, %2, %3 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  return %4#0 : tensor<1024x64xf32>
}
// CHECK-LABEL: func @softmax_tensor
// CHECK: scf.for
// CHECK:   linalg_ext.softmax
// CHECK:   scf.yield
// CHECK: }

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops = transform.structured.tile_ext %0 [4] 
}

func.func @softmax_memref(%arg0: memref<1024x64xf32>) -> (memref<1024x64xf32>) {
  %0 = memref.alloc() : memref<1024x64xf32>
  %1 = memref.alloc() : memref<1024xf32>
  %2 = memref.alloc() : memref<1024xf32>
  %3 = memref.alloc() : memref<1024xf32>
  linalg_ext.softmax
    dimension(1)
    ins(%arg0 : memref<1024x64xf32>) outs(%0, %1, %2, %3 : memref<1024x64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>)
  return %0 : memref<1024x64xf32>
}
//CHECK-LABEL: func.func @softmax_memref
//CHECK: scf.for
//CHECK:   linalg_ext.softmax
//CHECK: }
//CHECK: return

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops = transform.structured.tile_ext %0 [4] 
}

func.func @map_binary(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>,
                      %init: tensor<64xf32>) -> tensor<64xf32> {
   %add = linalg.map
          ins(%lhs, %rhs: tensor<64xf32>, tensor<64xf32>)
          outs(%init:tensor<64xf32>)  {__root__}
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            linalg.yield %0: f32
          }
  func.return %add : tensor<64xf32>
}
//CHECK-LABEL: func.func @map_binary
//CHECK: scf.for
//CHECK:   linalg.map
//CHECK: }
//CHECK: return

// -----

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.topk"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.tile_ext %0 [32, 16] {interchange = [0, 1, 2]}
}

func.func @topk_tensor(%input_values: tensor<1024x64xf32>, %input_indices: tensor<1024x64xi32>) -> (tensor<1024x3xf32>, tensor<1024x3xi32>) {
  %out_values = tensor.empty() : tensor<1024x3xf32>
  %out_indices = tensor.empty() : tensor<1024x3xi32>
  %0:2 = linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<1024x64xf32> , tensor<1024x64xi32>)
        outs(%out_values, %out_indices : tensor<1024x3xf32>, tensor<1024x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          linalg_ext.yield %0 : i1
        } -> tensor<1024x3xf32>, tensor<1024x3xi32>
  return %0#0, %0#1 : tensor<1024x3xf32>, tensor<1024x3xi32>
}
//CHECK-LABEL: func.func @topk_tensor
//CHECK: scf.for
//CHECK:   scf.for
//CHECK:     linalg_ext.topk
//CHECK:   scf.yield
//CHECK: scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.topk"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.tile_ext %0 [32, 16] {interchange = [0, 1, 2]}
}

func.func @topk_tensor_optional(%input_values: tensor<1024x64xf32>) -> (tensor<1024x3xf32>, tensor<1024x3xi32>) {
  %out_values = tensor.empty() : tensor<1024x3xf32>
  %out_indices = tensor.empty() : tensor<1024x3xi32>
  %0:2 = linalg_ext.topk
        dimension(1)
        ins(%input_values : tensor<1024x64xf32>)
        outs(%out_values, %out_indices : tensor<1024x3xf32>, tensor<1024x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          linalg_ext.yield %0 : i1
        } -> tensor<1024x3xf32>, tensor<1024x3xi32>
  return %0#0, %0#1 : tensor<1024x3xf32>, tensor<1024x3xi32>
}
//CHECK-LABEL: func.func @topk_tensor_optional
//CHECK: scf.for
//CHECK:   scf.for
//CHECK:     linalg_ext.topk
//CHECK:   scf.yield
//CHECK: scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.batch_matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.tile_ext %0 [2, 4, 8] {interchange = [0, 2, 1]}
}

func.func @batch_matmul_3d(%ta3: tensor<8x32x128xf32>, %tb3: tensor<8x128x64xf32>, %tc3: tensor<8x32x64xf32>) -> (tensor<8x32x64xf32>)
{
  %res = linalg_ext.batch_matmul
                    ins(%ta3, %tb3: tensor<8x32x128xf32>, tensor<8x128x64xf32>)
                    outs(%tc3: tensor<8x32x64xf32>)
                    layout = "nn"
  return %res : tensor<8x32x64xf32>
}
// CHECK-LABEL: func.func @batch_matmul_3d
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       tensor.extract_slice
// CHECK:       tensor.extract_slice
// CHECK:       tensor.extract_slice
// CHECK:       linalg_ext.batch_matmul
// CHECK:       tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield
// CHECK: scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops = transform.structured.tile_ext %0 [4]
}

func.func @expand_shape_simple(%arg0: tensor<128x1024x4096xf32>) -> tensor<128x1024x16x256xf32> {
  %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [128, 1024, 16, 256] {__root__} : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  return %expanded : tensor<128x1024x16x256xf32>
}
// CHECK-LABEL: func.func @expand_shape_simple
// CHECK: scf.for
// CHECK:   tensor.extract_slice {{.*}} [4, 1024, 4096]
// CHECK:   tensor.expand_shape
// CHECK:   tensor.insert_slice
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.tile_ext %0 [4, 0, 4]
}

func.func @expand_shape_tiling_on_expanded_dim(%arg0: tensor<128x1024x4096xf32>) -> tensor<128x1024x512x8xf32> {
  %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [128, 1024, 512, 8] {__root__} : tensor<128x1024x4096xf32> into tensor<128x1024x512x8xf32>
  return %expanded : tensor<128x1024x512x8xf32>
}
// CHECK-LABEL: func.func @expand_shape_tiling_on_expanded_dim
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice {{.*}} tensor<128x1024x4096xf32> to tensor<4x1024x32xf32>
// CHECK:     tensor.expand_shape {{.*}} tensor<4x1024x32xf32> into tensor<4x1024x4x8xf32>
// CHECK:     tensor.insert_slice {{.*}} tensor<4x1024x4x8xf32> into tensor<128x1024x512x8xf32>
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.tile_ext %0 [4, 0, 8]
}

func.func @expand_shape_tiling_on_dynamic_dim(%arg0: tensor<128x?xf32>) -> tensor<128x1x?x1xf32> {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<128x?xf32>
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3]] output_shape [128, 1, %dim, 1] {__root__} : tensor<128x?xf32> into tensor<128x1x?x1xf32>
  return %expanded : tensor<128x1x?x1xf32>
}
// CHECK-LABEL: func.func @expand_shape_tiling_on_dynamic_dim
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice {{.*}} tensor<128x?xf32> to tensor<4x?xf32>
// CHECK:     tensor.expand_shape {{.*}} tensor<4x?xf32> into tensor<4x1x?x1xf32>
// CHECK:     tensor.insert_slice {{.*}} tensor<4x1x?x1xf32> into tensor<128x1x?x1xf32>
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops = transform.structured.tile_ext %0 [4]
}

func.func @collapse_shape_simple(%arg0: tensor<128x16x1xf32>) ->tensor<128x16xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] {__root__} : tensor<128x16x1xf32> into tensor<128x16xf32>
  return %collapsed : tensor<128x16xf32>
}
// CHECK-LABEL: func.func @collapse_shape_simple
// CHECK: scf.for
// CHECK:   tensor.extract_slice {{.*}} [4, 16, 1]
// CHECK:   tensor.collapse_shape {{.*}} tensor<4x16x1xf32> into tensor<4x16xf32>
// CHECK:   tensor.insert_slice
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.tile_ext %0 [8, 4]
}

func.func @collapse_shape_tiling_on_collapse_dim(%arg0: tensor<128x1x8x2xf32>) ->tensor<128x16xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] {__root__} : tensor<128x1x8x2xf32> into tensor<128x16xf32>
  return %collapsed : tensor<128x16xf32>
}
// CHECK-LABEL: func.func @collapse_shape_tiling_on_collapse_dim
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice {{.*}} tensor<128x1x8x2xf32> to tensor<8x1x2x2xf32>
// CHECK:     tensor.collapse_shape {{.*}} tensor<8x1x2x2xf32> into tensor<8x4xf32>
// CHECK:     tensor.insert_slice {{.*}} tensor<8x4xf32> into tensor<128x16xf32>
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.tile_ext %0 [8, 4]
}

func.func @collapse_shape_tiling_on_dynamic_dim(%arg0: tensor<128x1x?x1xf32>) ->tensor<128x?xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] {__root__} : tensor<128x1x?x1xf32> into tensor<128x?xf32>
  return %collapsed : tensor<128x?xf32>
}
// CHECK-LABEL: func.func @collapse_shape_tiling_on_dynamic_dim
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice {{.*}} tensor<128x1x?x1xf32> to tensor<8x1x?x1xf32>
// CHECK:     tensor.collapse_shape {{.*}} tensor<8x1x?x1xf32> into tensor<8x?xf32>
// CHECK:     tensor.insert_slice {{.*}} tensor<8x?xf32> into tensor<128x?xf32>
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.scatter"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.tile_ext %0 [2, 4, 8] {interchange = [1, 2, 0]}
}

func.func @scatter(%src: tensor<2x3x32x64xf32>, %indices: tensor<100x2xi64>, %update: tensor<100x32x64xf32>) -> (tensor<2x3x32x64xf32>)
{
  %res = linalg_ext.scatter
    ins(%indices, %update: tensor<100x2xi64>, tensor<100x32x64xf32>)
    outs(%src: tensor<2x3x32x64xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg_ext.yield %0 : f32
      } -> tensor<2x3x32x64xf32>
  return %res : tensor<2x3x32x64xf32>
}
// CHECK-LABEL: func.func @scatter
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       tensor.extract_slice
// CHECK:       tensor.extract_slice
// CHECK:       tensor.extract_slice
// CHECK:       linalg_ext.scatter
// CHECK:       tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.layer_norm"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1:2, %loop = transform.structured.tile_ext %0 [4, 8] 
}

func.func @layer_norm_3d(%arg0: tensor<8x32x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<8x32x128xf32>) {
  %0 = tensor.empty() : tensor<8x32x128xf32>
  %1 = linalg_ext.layer_norm
                    axis([2])
                    epsilon(9.9999999747524271E-7)
                    ins(%arg0, %arg1, %arg2: tensor<8x32x128xf32>, tensor<128xf32>, tensor<128xf32>)
                    outs(%0: tensor<8x32x128xf32>) : tensor<8x32x128xf32>
  return %1 : tensor<8x32x128xf32>
}

// CHECK-LABEL: func.func @layer_norm_3d
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     tensor.extract_slice
// CHECK:     linalg_ext.layer_norm
// CHECK:     tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.layer_norm"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.tile_ext %0 [4] 
}

func.func @layer_norm_3d_axis_2d(%arg0: tensor<8x32x128xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<32x128xf32>) -> (tensor<8x32x128xf32>) {
  %0 = tensor.empty() : tensor<8x32x128xf32>
  %1 = linalg_ext.layer_norm
                    axis([1, 2])
                    epsilon(9.9999999747524271E-7)
                    ins(%arg0, %arg1, %arg2: tensor<8x32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>)
                    outs(%0: tensor<8x32x128xf32>) : tensor<8x32x128xf32>
  return %1 : tensor<8x32x128xf32>
}

// CHECK-LABEL: func.func @layer_norm_3d_axis_2d
// CHECK: scf.for
// CHECK:   tensor.extract_slice
// CHECK:   tensor.extract_slice
// CHECK:   linalg_ext.layer_norm
// CHECK:   tensor.insert_slice
// CHECK: scf.yield
