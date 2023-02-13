// RUN: byteir-opt --transform-dialect-interpreter --split-input-file %s | FileCheck %s

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
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
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg0
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
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg0
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
  %0 = transform.structured.match attributes{"__root__"} in %arg0
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
  %0 = transform.structured.match ops{["linalg_ext.topk"]} in %arg1
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
  %0 = transform.structured.match ops{["linalg_ext.topk"]} in %arg1
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
  %0 = transform.structured.match ops{["linalg_ext.batch_matmul"]} in %arg1
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
