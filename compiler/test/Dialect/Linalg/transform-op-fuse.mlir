// RUN: byteir-opt %s --transform-dialect-interpreter --canonicalize-ext --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fuse_unary
func.func @fuse_unary(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:    scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: return %[[RES]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<64x128xf32>)
                             outs(%arg1: tensor<64x128xf32>) -> tensor<64x128xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<64x128xf32>, tensor<64x128xf32>)
                             outs(%arg1: tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]} :
    (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

// -----

// CHECK-LABEL: func.func @fuse_elementwise_dynamic_of_intermediate
  //CHECK-SAME:  (%[[ARG0:.+]]: {{.+}}, %[[ARG1:.+]]: {{.+}})
func.func @fuse_elementwise_dynamic_of_intermediate(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
  // CHECK-DAG: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
  // CHECK: %[[RES:.*]] = scf.for
  // CHECK:    scf.for
  // CHECK:       linalg.elemwise_unary
  // CHECK:       linalg.elemwise_binary
  // CHECK: return %[[RES]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]}
}

// -----

// CHECK-LABEL: func.func @fuse_unary
func.func @fuse_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[PARTIAL_RES:.*]] = scf.for
  //     CHECK:     scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: %[[RES:.*]] = scf.for {{.*}}%[[PARTIAL_RES]]
  //     CHECK:     scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: return %[[RES]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]}
  %loop = transform.cast %loops#0 : !pdl.operation to !transform.op<"scf.for">
  transform.loop.peel %loop : (!transform.op<"scf.for">) -> (!pdl.operation, !pdl.operation)
}

// -----

// CHECK-LABEL: func.func @fuse_with_multi_stops
func.func @fuse_with_multi_stops(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = linalg.elemwise_unary ins(%arg0 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  // CHECK: __stop__
  // CHECK: __stop__
  %2 = linalg.elemwise_unary {__stop__} ins(%arg1 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %3 = linalg.elemwise_unary {__stop__} ins(%arg2 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  // CHECK: scf.for
  // CHECK:     linalg.elemwise_unary
  // CHECK:     linalg.elemwise_binary
  // CHECK:     linalg.elemwise_binary
  // CHECK:     __root__
  // CHECK:     scf.yield
  %4 = linalg.elemwise_binary ins(%1, %2 : tensor<1024x512xf32>, tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %5 = linalg.elemwise_binary {__root__} ins(%4, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %5 : tensor<1024x512xf32>
}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %stops = transform.structured.match attributes {__stop__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_ext %0, %stops {tile_interchange = [], tile_sizes = [4]}
  cleanup
}

// -----

// CHECK-LABEL: func.func @interchange_reduction
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<12x7x25xf32>)
func.func @interchange_reduction(%input: tensor<12x7x25xf32>) -> tensor<12x25xf32> {
  %five = arith.constant 5.0 : f32
  %init = tensor.empty() : tensor<12x25xf32>

//   CHECK-DAG: %[[INIT:.+]] = tensor.empty()
//   CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG: %[[C7:.+]] = arith.constant 7 : index
//   CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
//       CHECK: %[[RES:.*]] = scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %[[C5]] iter_args(%[[FOR_ARG0:.+]] = %[[INIT]])
//       CHECK:   scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %[[C7]] iter_args(%[[FOR_ARG1:.+]] = %[[FOR_ARG0]])
//       CHECK:     %[[OUT_SLICE0:.+]] = tensor.extract_slice %[[INPUT]][%[[IV0]], 0, %[[IV1]]]
//       CHECK:     %[[OUT_SLICE1:.+]] = tensor.extract_slice %[[FOR_ARG1]][%[[IV0]], %[[IV1]]]
//       CHECK:     %[[FILL:.+]] = linalg.fill {{.+}} outs(%[[OUT_SLICE1]] : tensor<?x?xf32>)
//
// Extra 4 constant is introduced, discard it.
//       CHECK:     scf.for %[[IV2:.+]] = %{{.+}} to %{{.+}} step %[[C4]] iter_args(%[[FOR_ARG2:.+]] = %[[FILL]])
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[OUT_SLICE0]]
//       CHECK:       %[[OUT_SLICE2:.+]] = tensor.extract_slice %[[FOR_ARG2]][0, 0]
//       CHECK:       linalg.generic {{.+}} ins(%[[IN_SLICE]] : tensor<?x?x?xf32>) outs(%[[OUT_SLICE2]] : tensor<?x?xf32>)
//       CHECK: return %[[RES]]

  %fill = linalg.fill ins(%five : f32) outs(%init : tensor<12x25xf32>) -> tensor<12x25xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>],
    iterator_types = ["parallel", "reduction", "parallel"]
  } ins(%input : tensor<12x7x25xf32>) outs(%fill : tensor<12x25xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %2 = arith.addf %arg0, %arg1 : f32
    linalg.yield %2 : f32
  } -> tensor<12x25xf32>
  func.return %0 : tensor<12x25xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [5, 0, 7], tile_interchange = [0, 2, 1]} :
    (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  %2, %loops_2 = transform.structured.tile_using_for %1 tile_sizes [0, 4] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// -----

// CHECK-LABEL: func.func @fuse_unary_softmax
func.func @fuse_unary_softmax(%arg0: tensor<1024x64xf32>, %arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32> {
  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK-DAG:   linalg.elemwise_unary
  //     CHECK-DAG:   linalg.fill
  //     CHECK-DAG:   linalg.fill
  //     CHECK:       linalg_ext.softmax
  //     CHECK: } {__byteir_parallel__}
  //     CHECK: return %[[RES]]
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_1 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_2 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %4 = linalg.elemwise_unary ins(%arg0 : tensor<1024x64xf32>)
                             outs(%arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32>
  %5:4 = linalg_ext.softmax
    dimension(1)
    ins(%4 : tensor<1024x64xf32>) outs(%0, %fill_1, %fill_2, %3 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  return %5#0 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops = transform.structured.fuse_ext %0 {tile_sizes = [4], tile_interchange = [0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_softmax_unary_tile_1D
func.func @fuse_softmax_unary_tile_1D(%arg0: tensor<1024x64xf32>, %arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32> {
  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK-DAG:   linalg.fill
  //     CHECK-DAG:   linalg.fill
  //     CHECK:       linalg_ext.softmax
  //     CHECK:       linalg.elemwise_unary
  //     CHECK: } {__byteir_parallel__}
  //     CHECK: return %[[RES]]
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_1 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_2 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %4:4 = linalg_ext.softmax
    dimension(1)
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %fill_1, %fill_2, %3 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  %5 = linalg.elemwise_unary ins(%4#0 : tensor<1024x64xf32>)
                             outs(%arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32>
  return %5 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops = transform.structured.fuse_ext %0 {tile_sizes = [4], tile_interchange = [0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_softmax_unary_tile_2D
func.func @fuse_softmax_unary_tile_2D(%arg0: tensor<1024x64xf32>, %arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32> {
  // CHECK-DAG: linalg.fill
  // CHECK-DAG: linalg.fill
  // CHECK:     linalg_ext.softmax
  // CHECK:     %[[RES:.*]] = scf.for
  // CHECK:        linalg.elemwise_unary
  // CHECK:     } {__byteir_parallel__}
  // CHECK:     return %[[RES]]
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_1 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_2 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %4:4 = linalg_ext.softmax
    dimension(1)
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %fill_1, %fill_2, %3 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  %5 = linalg.elemwise_unary ins(%4#0 : tensor<1024x64xf32>)
                             outs(%arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32>
  return %5 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [0, 1]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_matmul_softmax
func.func @fuse_matmul_softmax(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<1024x64xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK-DAG:   linalg.matmul
  //     CHECK-DAG:   linalg.fill
  //     CHECK-DAG:   linalg.fill
  //     CHECK:       linalg_ext.softmax
  //     CHECK:     } {__byteir_parallel__}
  //     CHECK: return %[[RES]]
  %0 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024x64xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_3 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_4 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x64xf32>)
                     outs(%0: tensor<1024x64xf32>)
    -> tensor<1024x64xf32>


  %6:4 = linalg_ext.softmax
    dimension(1)
    ins(%1 : tensor<1024x64xf32>) outs(%2, %fill_3, %fill_4, %5 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>

  return %6#0 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.fuse_ext %0 {tile_sizes = [4], tile_interchange = [0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_softmax_matmul_tile_0
func.func @fuse_softmax_matmul_tile_0(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<1024x64xf32> {
  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK-DAG:   linalg.fill
  //     CHECK-DAG:   linalg.fill
  //     CHECK:       linalg_ext.softmax
  //     CHECK:       linalg.matmul
  //     CHECK: } {__byteir_parallel__}
  //     CHECK: return %[[RES]]
  %0 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_3 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_4 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
  %6:4 = linalg_ext.softmax
    dimension(1)
    ins(%arg0 : tensor<1024x32xf32>) outs(%2, %fill_3, %fill_4, %5 : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  %7 = linalg.matmul  ins(%6#0, %arg1: tensor<1024x32xf32>, tensor<32x64xf32>)
                     outs(%0: tensor<1024x64xf32>)
    -> tensor<1024x64xf32>

  return %7 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.fuse_ext %0 {tile_sizes = [4], tile_interchange = [0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_softmax_matmul_tile_1
func.func @fuse_softmax_matmul_tile_1(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<1024x64xf32> {
  //     CHECK: scf.for
  //     CHECK-DAG:   linalg.fill
  //     CHECK-DAG:   linalg.fill
  //     CHECK:       linalg_ext.softmax
  //     CHECK:       linalg.matmul
  //     CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_3 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_4 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
  %6:4 = linalg_ext.softmax
    dimension(1)
    ins(%arg0 : tensor<1024x32xf32>) outs(%2, %fill_3, %fill_4, %5 : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  %7 = linalg.matmul  ins(%6#0, %arg1: tensor<1024x32xf32>, tensor<32x64xf32>)
                     outs(%0: tensor<1024x64xf32>)
    -> tensor<1024x64xf32>

  return %7 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.fuse_ext %0 {tile_sizes = [0, 4], tile_interchange = [0, 1]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_softmax_matmul_tile_2
func.func @fuse_softmax_matmul_tile_2(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<1024x64xf32> {
  //     CHECK-DAG: linalg.fill
  //     CHECK-DAG: linalg.fill
  //     CHECK:     scf.for
  //     CHECK:       linalg_ext.softmax
  //     CHECK:       linalg_ext.diag
  //     CHECK:       linalg.matmul
  //     CHECK:       linalg.matmul
  //     CHECK:     }
  %0 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_3 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_4 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
  %6:4 = linalg_ext.softmax
    dimension(1)
    ins(%arg0 : tensor<1024x32xf32>) outs(%2, %fill_3, %fill_4, %5 : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  %7 = linalg.matmul  ins(%6#0, %arg1: tensor<1024x32xf32>, tensor<32x64xf32>)
                     outs(%0: tensor<1024x64xf32>)
    -> tensor<1024x64xf32>

  return %7 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.fuse_ext %0 {tile_sizes = [0, 0, 4], tile_interchange = [0, 1, 2]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----


// CHECK-LABEL: func.func @fuse_softmax_matmul_broadcast_elementwise
func.func @fuse_softmax_matmul_broadcast_elementwise(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<1024x64xf32> {
  //     CHECK-DAG: linalg.fill
  //     CHECK-DAG: linalg.fill
  //     CHECK:     scf.for
  //     CHECK:       linalg_ext.softmax
  //     CHECK:       linalg_ext.diag
  //     CHECK:       linalg.matmul
  //     CHECK:       linalg.matmul
  //     CHECK:     }
  //     CHECK:     linalg.broadcast
  //     CHECK:     linalg.elemwise_binary
  %0 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %t1 = tensor.empty() : tensor<1024x1024xf32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_3 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_4 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
  %6:4 = linalg_ext.softmax
    dimension(1)
    ins(%arg0 : tensor<1024x32xf32>) outs(%2, %fill_3, %fill_4, %5 : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  %7 = linalg.matmul {__root__} ins(%6#0, %arg1: tensor<1024x32xf32>, tensor<32x64xf32>)
                     outs(%0: tensor<1024x64xf32>)
    -> tensor<1024x64xf32>
  %broadcasted = linalg.broadcast ins(%6#2 : tensor<1024xf32>) outs(%0 : tensor<1024x64xf32>) dimensions = [1]
  %8 = linalg.elemwise_binary ins(%7, %broadcasted : tensor<1024x64xf32>, tensor<1024x64xf32>)
                              outs(%0: tensor<1024x64xf32>) -> tensor<1024x64xf32>
  return %8 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.fuse_ext %0 {tile_sizes = [0, 0, 4], tile_interchange = [0, 1, 2]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_matmul_matmul
func.func @fuse_matmul_matmul(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.matmul
// CHECK:     linalg.matmul
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%0: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>

  %3 = linalg.matmul {__root__} ins(%1, %arg2: tensor<1024x512xf32>, tensor<512x32xf32>)
                     outs(%2: tensor<1024x32xf32>)
    -> tensor<1024x32xf32>
  return %3 : tensor<1024x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 0, 8], tile_interchange = [0, 1, 2]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_2_matmul_2_output
func.func @fuse_2_matmul_2_output(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> (tensor<1024x32xf32>, tensor<1024x512xf32>) {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.matmul
// CHECK:     linalg.matmul
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: }
  %0 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = tensor.empty() : tensor<1024x512xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %6 = tensor.empty() : tensor<1024xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%0: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>

  %8 = linalg.matmul {__root__} ins(%1, %arg2: tensor<1024x512xf32>, tensor<512x32xf32>)
                     outs(%2: tensor<1024x32xf32>)
    -> tensor<1024x32xf32>
  return %8, %1: tensor<1024x32xf32>, tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 0, 8], tile_interchange = [2, 1, 0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_2_matmul_add
func.func @fuse_2_matmul_add(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<1024x32xf32>, %arg3: tensor<32x512xf32>) -> tensor<1024x512xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.matmul
// CHECK:     linalg.matmul
// CHECK:     linalg.elemwise_binary
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %4 = tensor.empty() : tensor<1024x512xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%0: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>
  %3 = linalg.matmul  ins(%arg2, %arg3: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%2: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>
  %5 = linalg.elemwise_binary {__root__} ins(%1, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%4: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %5: tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_fork_add
func.func @fuse_fork_add(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.elemwise_binary
// CHECK:     linalg.elemwise_binary
// CHECK:     linalg.elemwise_binary
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = linalg.elemwise_binary ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%0: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %4 = linalg.elemwise_binary ins(%arg1, %arg2 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%1: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %5 = linalg.elemwise_binary {__root__} ins(%3, %4 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%2: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %5: tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_fork_map
func.func @fuse_fork_map(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.map
// CHECK:     linalg.map
// CHECK:     linalg.map
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = linalg.map
      ins(%arg0, %arg1: tensor<1024x512xf32>, tensor<1024x512xf32>)
      outs(%0:tensor<1024x512xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %6 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %6: f32
      }
  %4 = linalg.map
      ins(%arg1, %arg2: tensor<1024x512xf32>, tensor<1024x512xf32>)
      outs(%1:tensor<1024x512xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %6 = arith.mulf %lhs_elem, %rhs_elem: f32
        linalg.yield %6: f32
      }
  %5 = linalg.map
      ins(%3, %4: tensor<1024x512xf32>, tensor<1024x512xf32>)
      outs(%2:tensor<1024x512xf32>)  {__root__}
      (%lhs_elem: f32, %rhs_elem: f32) {
        %6 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %6: f32
      }
  return %5: tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_2_add_sharing_add
func.func @fuse_2_add_sharing_add(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>, %arg3: tensor<1024x512xf32>) -> (tensor<1024x512xf32>,tensor<1024x512xf32>) {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.elemwise_binary
// CHECK:     linalg.elemwise_binary {__root__}
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
// CHECK: linalg.elemwise_binary
// CHECK: linalg.elemwise_binary {__other__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = linalg.elemwise_binary ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%0: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %4 = linalg.elemwise_binary ins(%arg2, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%1: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %6 = linalg.elemwise_binary {__other__} ins(%arg2, %4 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%1: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %5 = linalg.elemwise_binary {__root__} ins(%arg3, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%2: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %6, %5: tensor<1024x512xf32>, tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @max_pool_generic
func.func @max_pool_generic(%arg0: tensor<4x126x126x16xf32>) -> tensor<4x63x63x16xf32> {
// CHECK: scf.for
// CHECK:   linalg.generic
// CHEKC-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:     arith.maxnumf
// CHECK:     linalg.yield
// CHECK: scf.yield
  %cst = arith.constant dense<0xFC00> : tensor<4x63x63x16xf32>
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %0 : tensor<4x126x126x16xf32>, tensor<2x2xf32>) outs(%cst : tensor<4x63x63x16xf32>) attrs =  {__root__} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.maxnumf %out, %in : f32
    linalg.yield %5 : f32
  } -> tensor<4x63x63x16xf32>
  return %1 : tensor<4x63x63x16xf32>
}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [1]}
}

// -----

// CHECK-LABEL: func.func @max_pool
func.func @max_pool(%arg0: tensor<4x126x126x16xf32>) -> tensor<4x63x63x16xf32> attributes {__test__} {
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = tensor.empty() : tensor<4x63x63x16xf32>
// CHECK: scf.for
// CHECK:   linalg.fill
// CHECK:   linalg.pooling_nhwc_max
// CHECK: scf.yield
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4x63x63x16xf32>) -> tensor<4x63x63x16xf32>
  %3 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, __root__} ins(%arg0, %0 : tensor<4x126x126x16xf32>, tensor<2x2xf32>) outs(%2 : tensor<4x63x63x16xf32>) -> tensor<4x63x63x16xf32>
  return %3 : tensor<4x63x63x16xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [1]}
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @add2_generic
func.func @add2_generic(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {__test__} {
  %0 = tensor.empty() : tensor<4x4xf32>
// CHECK: scf.for
// CHECK:   linalg.generic
// CHECK: scf.yield
  %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) attrs =  {__root__} {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %2 = arith.addf %in, %in_0 : f32
    %3 = arith.addf %2, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [1]}
}

// -----

// CHECK-LABEL: func.func @fuse_scan_unary_tile_1D
func.func @fuse_scan_unary_tile_1D(%arg0: tensor<1024x64xf32>, %arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32> {
  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:   linalg.fill
  //     CHECK:   linalg_ext.scan
  //     CHECK:   linalg.elemwise_unary
  //     CHECK: } {__byteir_parallel__}
  //     CHECK: return %[[RES]]
  %0 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_2 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %4:2 = linalg_ext.scan
    dimension(1) inclusive(true)
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %fill_2 : tensor<1024x64xf32>, tensor<1024xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %sum = arith.addf %arg2, %arg3 : f32
      linalg_ext.yield %sum : f32
  } -> tensor<1024x64xf32>, tensor<1024xf32>

  %5 = linalg.elemwise_unary ins(%4#0 : tensor<1024x64xf32>)
                             outs(%arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32>
  return %5 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops = transform.structured.fuse_ext %0 {tile_sizes = [4], tile_interchange = [0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_scan_unary_tile_2D
func.func @fuse_scan_unary_tile_2D(%arg0: tensor<1024x64xf32>) -> tensor<1024x64xf32> {
  // CHECK: linalg.fill
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     linalg_ext.scan
  // CHECK:     linalg.elemwise_unary
  // CHECK:   }
  // CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_2 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %4:2 = linalg_ext.scan
    dimension(1) inclusive(true)
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %fill_2 : tensor<1024x64xf32>, tensor<1024xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %sum = arith.addf %arg2, %arg3 : f32
      linalg_ext.yield %sum : f32
  } -> tensor<1024x64xf32>, tensor<1024xf32>

  %5 = linalg.elemwise_unary ins(%4#0 : tensor<1024x64xf32>)
                             outs(%1: tensor<1024x64xf32>) -> tensor<1024x64xf32>
  return %5 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [0, 1]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func.func @fuse_unary_scan
func.func @fuse_unary_scan(%arg0: tensor<1024x64xf32>, %arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32> {
  //     CHECK: linalg.fill
  //     CHECK: scf.for
  //     CHECK:   scf.for
  //     CHECK:     linalg.elemwise_unary
  //     CHECK:     linalg_ext.scan
  //     CHECK:   }
  //     CHECK: } {__byteir_parallel__}

  %0 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %cst_0 = arith.constant 0.0 : f32
  %fill_2 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>

  %4 = linalg.elemwise_unary ins(%arg0 : tensor<1024x64xf32>)
                             outs(%arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32>

  %5:2 = linalg_ext.scan
    dimension(1) inclusive(true)
    ins(%4 : tensor<1024x64xf32>) outs(%0, %fill_2 : tensor<1024x64xf32>, tensor<1024xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %sum = arith.addf %arg2, %arg3 : f32
      linalg_ext.yield %sum : f32
  } -> tensor<1024x64xf32>, tensor<1024xf32>

  return %5#0 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.scan"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [0, 1]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

//CHECK-LABEL: func.func @elementwise_topk
func.func @elementwise_topk(%input_values: tensor<1024x64xf32>) -> (tensor<1024x3xf32>, tensor<1024x3xi32>) {
  //     CHECK: linalg.fill
  //     CHECK: scf.for
  //     CHECK:   scf.for
  //     CHECK:     linalg.elemwise_unary
  //     CHECK:     linalg_ext.topk
  //     CHECK:   }
  //     CHECK: } {__byteir_parallel__}
  %out_values = tensor.empty() : tensor<1024x3xf32>
  %out_indices = tensor.empty() : tensor<1024x3xi32>
  %cst = arith.constant 0xFF800000 : f32
  %fill_values = linalg.fill ins(%cst : f32) outs(%out_values : tensor<1024x3xf32>) -> tensor<1024x3xf32>
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = linalg.elemwise_unary ins(%input_values : tensor<1024x64xf32>)
                             outs(%0: tensor<1024x64xf32>) -> tensor<1024x64xf32>
  %2:2 = linalg_ext.topk
        dimension(1)
        ins(%1 : tensor<1024x64xf32>)
        outs(%fill_values, %out_indices : tensor<1024x3xf32>, tensor<1024x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %3 = arith.cmpf ogt, %arg0, %arg1 : f32
          linalg_ext.yield %3 : i1
        } -> tensor<1024x3xf32>, tensor<1024x3xi32>
  return %2#0, %2#1 : tensor<1024x3xf32>, tensor<1024x3xi32>
}


transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.topk"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [32, 16], tile_interchange = [0, 1]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

func.func @batch_matmul_elementwise(%ta3: tensor<8x32x128xf32>, %tb3: tensor<8x128x64xf32>, %tc3: tensor<8x32x64xf32>) -> (tensor<8x32x64xf32>)
{
  %empty= tensor.empty() : tensor<8x32x64xf32>
  %0 = linalg_ext.batch_matmul
                    ins(%ta3, %tb3: tensor<8x32x128xf32>, tensor<8x128x64xf32>)
                    outs(%tc3: tensor<8x32x64xf32>)
                    layout = "nn"
  %1 = linalg.elemwise_unary ins(%0 : tensor<8x32x64xf32>)
                             outs(%empty: tensor<8x32x64xf32>) -> tensor<8x32x64xf32>
  return %1 : tensor<8x32x64xf32>
}
// CHECK-LABEL: func.func @batch_matmul_elementwise
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg_ext.batch_matmul
// CHECK:     linalg.elemwise_unary
// CHECK:   scf.yield
// CHECK: scf.yield

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [2, 4], tile_interchange = [0, 1]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

func.func @elementwise_batch_matmul(%ta3: tensor<8x32x128xf32>, %tb3: tensor<8x128x64xf32>, %tc3: tensor<8x32x64xf32>) -> (tensor<8x32x64xf32>)
{
  %empty= tensor.empty() : tensor<8x32x128xf32>
  %0 = linalg.elemwise_unary ins(%ta3 : tensor<8x32x128xf32>)
                             outs(%empty: tensor<8x32x128xf32>) -> tensor<8x32x128xf32>
  %1 = linalg_ext.batch_matmul
                    ins(%0, %tb3: tensor<8x32x128xf32>, tensor<8x128x64xf32>)
                    outs(%tc3: tensor<8x32x64xf32>)
                    layout = "nn"
  return %1 : tensor<8x32x64xf32>
}
// CHECK-LABEL: func.func @elementwise_batch_matmul
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.elemwise_unary
// CHECK:     linalg_ext.batch_matmul
// CHECK:   scf.yield
// CHECK: scf.yield

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.batch_matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [2, 4], tile_interchange = [0, 1]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

//CHECK-LABEL: func.func @fuse_fill
func.func @fuse_fill() -> tensor<1024x32xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.empty
// CHECK:     linalg.fill
// CHECK:     tensor.empty
// CHECK:     linalg.elemwise_unary
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
  %0 = tensor.empty() : tensor<1024x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = linalg.elemwise_unary {__root__} ins(%1 : tensor<1024x32xf32>)
                             outs(%2: tensor<1024x32xf32>) -> tensor<1024x32xf32>
  return %3 : tensor<1024x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [8, 4], tile_interchange = [0, 1]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [4]}
}

func.func @fuse_expand_shape(%arg0: tensor<128x1024x256xf32>, %arg1: tensor<128x256x4096xf32>) -> tensor<128x16x1024x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x1024x4096xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<128x1024x256xf32>, tensor<128x256x4096xf32>) outs(%1 : tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %expanded = tensor.expand_shape %2 [[0], [1], [2, 3]] output_shape [128, 1024, 16, 256] : tensor<128x1024x4096xf32> into tensor<128x1024x16x256xf32>
  %3 = tensor.empty() : tensor<128x16x1024x256xf32>
  %transposed = linalg.transpose
    ins(%expanded : tensor<128x1024x16x256xf32>)
    outs(%3 : tensor<128x16x1024x256xf32>)
    permutation = [0, 2, 1, 3]  {__root__}
  return %transposed : tensor<128x16x1024x256xf32>
}
// CHECK-LABEL: func.func @fuse_expand_shape
// CHECK: scf.for
// CHECK:   linalg.batch_matmul
// CHECK:   tensor.expand_shape
// CHECK:   linalg.transpose
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [4, 0, 4]}
}

func.func @elementwise_expand_shape(%arg0: tensor<128x1024x4096xf32>) -> tensor<128x1024x512x8xf32> {
  %empty= tensor.empty() : tensor<128x1024x4096xf32>
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<128x1024x4096xf32>)
                             outs(%empty: tensor<128x1024x4096xf32>) -> tensor<128x1024x4096xf32>
  %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] output_shape [128, 1024, 512, 8] {__root__} : tensor<128x1024x4096xf32> into tensor<128x1024x512x8xf32>
  return %expanded : tensor<128x1024x512x8xf32>
}

// CHECK-LABEL: func.func @elementwise_expand_shape
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     linalg.elemwise_unary
// CHECK:     tensor.expand_shape
// CHECK:     tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [4, 0, 4]}
}

func.func @expand_shape_elementwise(%arg0: tensor<128x1024x4096xf32>) -> tensor<128x1024x512x8xf32> {
  %empty= tensor.empty() : tensor<128x1024x512x8xf32>
  %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [128, 1024, 512, 8] : tensor<128x1024x4096xf32> into tensor<128x1024x512x8xf32>
  %0 = linalg.elemwise_unary ins(%expanded : tensor<128x1024x512x8xf32>)
                             outs(%empty: tensor<128x1024x512x8xf32>) -> tensor<128x1024x512x8xf32>
  return %0 : tensor<128x1024x512x8xf32>
}

// CHECK-LABEL: func.func @expand_shape_elementwise
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     tensor.expand_shape
// CHECK:     linalg.elemwise_unary
// CHECK:     tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:3 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [1, 1, 256]}
}

func.func @expand_shape_elementwise_tile_1x1xN(%arg0: tensor<131072xf32>) -> tensor<8x16x1024xf32> {
  %empty= tensor.empty() : tensor<8x16x1024xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [8, 16, 1024] : tensor<131072xf32> into tensor<8x16x1024xf32>
  %0 = linalg.elemwise_unary ins(%expanded : tensor<8x16x1024xf32>)
                             outs(%empty: tensor<8x16x1024xf32>) -> tensor<8x16x1024xf32>
  return %0 : tensor<8x16x1024xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 1024 + d2)>
// CHECK-LABEL: func.func @expand_shape_elementwise_tile_1x1xN
//   CHECK-SAME: (%[[ARG0:.+]]: tensor<131072xf32>)
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: scf.for %[[ARG1:.+]] = %[[C0]]
// CHECK:   scf.for %[[ARG2:.+]] = %[[C0]]
// CHECK:     scf.for %[[ARG3:.+]] = %[[C0]]
// CHECK:       %[[OFFSET:.+]] = affine.apply #[[MAP]](%[[ARG1]], %[[ARG2]], %[[ARG3]])
// CHECK:       tensor.extract_slice %[[ARG0]][%[[OFFSET]]] [256] [1]
// CHECK:       tensor.expand_shape
// CHECK:       linalg.elemwise_unary
// CHECK:       tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield
// CHECK: scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [8, 4]}
}

func.func @elementwise_collapse_shape(%arg0: tensor<128x1x8x2xf32>) ->tensor<128x16xf32> {
  %empty= tensor.empty() : tensor<128x1x8x2xf32>
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<128x1x8x2xf32>)
                             outs(%empty: tensor<128x1x8x2xf32>) -> tensor<128x1x8x2xf32>
  %collapsed = tensor.collapse_shape %0 [[0], [1, 2, 3]] {__root__} : tensor<128x1x8x2xf32> into tensor<128x16xf32>
  return %collapsed : tensor<128x16xf32>
}
// CHECK-LABEL: func.func @elementwise_collapse_shape
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     linalg.elemwise_unary
// CHECK:     tensor.collapse_shape
// CHECK:     tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [8, 4]}
}

func.func @collapse_shape_elementwise(%arg0: tensor<128x1x8x2xf32>) ->tensor<128x16xf32> {
  %empty= tensor.empty() : tensor<128x16xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<128x1x8x2xf32> into tensor<128x16xf32>
  %0 = linalg.elemwise_unary ins(%collapsed : tensor<128x16xf32>)
                             outs(%empty: tensor<128x16xf32>) -> tensor<128x16xf32>
  return %0 : tensor<128x16xf32>
}
// CHECK-LABEL: func.func @collapse_shape_elementwise
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     tensor.collapse_shape
// CHECK:     linalg.elemwise_unary
// CHECK:     tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [256]}
}

func.func @collapse_shape_elementwise_slice_1x1xN(%arg0: tensor<8x16x1024xf32>) -> tensor<131072xf32> {
  %empty= tensor.empty() : tensor<131072xf32>
  %expanded = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<8x16x1024xf32> into tensor<131072xf32>
  %0 = linalg.elemwise_unary ins(%expanded : tensor<131072xf32>)
                             outs(%empty: tensor<131072xf32>) -> tensor<131072xf32>
  return %0 : tensor<131072xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> (((d0 floordiv 1024) floordiv 16) mod 8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> ((d0 floordiv 1024) mod 16)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (d0 mod 1024)>
// CHECK-LABEL: func.func @collapse_shape_elementwise_slice_1x1xN
//   CHECK-SAME: (%[[ARG0:.+]]: tensor<8x16x1024xf32>)
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: scf.for %[[ARG1:.+]] = %[[C0]]
// CHECK-DAG: %[[OFFSET0:.+]] = affine.apply #[[MAP0]](%[[ARG1]])
// CHECK-DAG: %[[OFFSET1:.+]] = affine.apply #[[MAP1]](%[[ARG1]])
// CHECK-DAG: %[[OFFSET2:.+]] = affine.apply #[[MAP2]](%[[ARG1]])
// CHECK:   tensor.extract_slice %[[ARG0]][%[[OFFSET0]], %[[OFFSET1]], %[[OFFSET2]]] [1, 1, 256] [1, 1, 1]
// CHECK:   linalg.elemwise_unary
// CHECK:   tensor.insert_slice
// CHECK: scf.yield

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [7, 7]}
}
func.func @elew_pad_elew(%arg0: tensor<12x12xf32>) -> tensor<14x14xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<12x12xf32>
  %1 = linalg.elemwise_unary ins(%arg0 : tensor<12x12xf32>) outs(%0 : tensor<12x12xf32>) -> tensor<12x12xf32>
  %padded = tensor.pad %1 nofold low[1, 1] high[1, 1] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<12x12xf32> to tensor<14x14xf32>
  %2 = tensor.empty() : tensor<14x14xf32>
  %3 = linalg.elemwise_unary {__root__} ins(%padded : tensor<14x14xf32>) outs(%2 : tensor<14x14xf32>) -> tensor<14x14xf32>
  return %3 : tensor<14x14xf32>
}
// CHECK-LABEL: func.func @elew_pad_elew
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     tensor.extract_slice
// CHECK:     linalg.elemwise_unary
// CHECK:     tensor.pad
// CHECK:     tensor.empty
// CHECK:     linalg.elemwise_unary
// CHECK:     tensor.insert_slice
// CHECK:     scf.yield
// CHECK:   scf.yield

// -----

#map = affine_map<(d0) -> (d0)>

func.func @multi_results_one_in_tile_fuse_path_one_in_terminator(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
  // CHECK-LABEL: func.func @multi_results_one_in_tile_fuse_path_one_in_terminator
  // CHECK: %[[E0:.*]] = tensor.empty() : tensor<1024xf32>
  // CHECK: scf.for {{.*}} iter_args(%[[ARG0:.*]] = %[[E0]], %[[ARG1:.*]] = %[[E0]])
  // CHECK:     %[[V0:.*]]:2 = linalg.generic{{.*}}__g0__
  // CHECK:     %[[V1:.*]] = linalg.generic{{.*}}__g1__
  // CHECK-DAG: %[[INS0:.*]] = tensor.insert_slice %[[V1]] into %[[ARG0]]
  // CHECK-DAG: %[[INS1:.*]] = tensor.insert_slice %[[V0]]#1 into %[[ARG1]]
  // CHECK:     scf.yield %[[INS0]], %[[INS1]]
  %0 = tensor.empty() : tensor<1024xf32>
  %1:2 = linalg.generic {__g0__, indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%0, %0 : tensor<1024xf32>, tensor<1024xf32>) {
  ^bb0(%in_0: f32, %in_1: f32, %out_0: f32, %out_1: f32):
    %2 = arith.addf %in_0, %in_1 : f32
    %3 = arith.subf %in_0, %in_1 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<1024xf32>, tensor<1024xf32>)
  %2 = linalg.generic {__root__, __g1__, indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg2, %1#0 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in_0: f32, %in_1: f32, %out : f32):
    %2 = arith.addf %in_0, %in_1 : f32
    linalg.yield %2 : f32
  } -> tensor<1024xf32>
  return %1#1, %2 : tensor<1024xf32>, tensor<1024xf32>
}

transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [1]}
  cleanup
}
