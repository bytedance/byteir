// RUN: byteir-opt %s --transform-dialect-interpreter --split-input-file -allow-unregistered-dialect -verify-diagnostics | FileCheck %s


func.func @tile_two_roots(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>, %arg3: tensor<64x128xf32>) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %0 = linalg.elemwise_unary {__op0__} ins(%arg0 : tensor<64x128xf32>)
                             outs(%arg1: tensor<64x128xf32>) -> tensor<64x128xf32>
  %1 = linalg.elemwise_unary {__op1__} ins(%arg2 : tensor<64x128xf32>)
                             outs(%arg3: tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0, %1: tensor<64x128xf32>, tensor<64x128xf32>
}
// CHECK-LABEL: func.func @tile_two_roots
// CHECK:  %[[RES:.*]]:2 = scf.for
// CHECK-DAG: %[[V0:.*]] = {{.*}} {__op0__}
// CHECK-DAG: %[[V1:.*]] = {{.*}} {__op1__}
// CHECK-DAG: %[[V2:.*]] =  tensor.insert_slice %[[V0]]
// CHECK-DAG: %[[V3:.*]] =  tensor.insert_slice %[[V1]]
// CHECK: return %[[RES]]#{{0|1}}, %[[RES]]#{{0|1}}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops {["func.return"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.fuse_operands %0 {tile_nums = [32]}
  cleanup
}

// -----

func.func @root_with_producer(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = linalg.elemwise_unary {__op0__} ins(%arg0 : tensor<64x128xf32>)
                             outs(%arg1: tensor<64x128xf32>) -> tensor<64x128xf32>
  %1 = linalg.elemwise_unary {__op1__} ins(%0 : tensor<64x128xf32>)
                             outs(%arg2: tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1: tensor<64x128xf32>
}
// CHECK-LABEL: func.func @root_with_producer
// CHECK:  %[[RES:.*]] = scf.for
// CHECK-DAG: %[[V0:.*]] = {{.*}} {__op0__}
// CHECK-DAG: %[[V1:.*]] = {{.*}} {__op1__} ins(%[[V0]]{{.*}}
// CHECK-DAG: %[[V2:.*]] =  tensor.insert_slice %[[V1]]
// CHECK:  scf.yield %[[V2]]
// CHECK: return %[[RES]]

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops {["func.return"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.fuse_operands %0 {tile_nums = [32]}
  cleanup
}

// -----

func.func @return_both_root_and_its_producer(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %0 = linalg.elemwise_unary {__op0__} ins(%arg0 : tensor<64x128xf32>)
                             outs(%arg1: tensor<64x128xf32>) -> tensor<64x128xf32>
  %1 = linalg.elemwise_unary {__op1__} ins(%0 : tensor<64x128xf32>)
                             outs(%arg2: tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0, %1: tensor<64x128xf32>, tensor<64x128xf32>
}
// CHECK-LABEL: func.func @return_both_root_and_its_producer
// CHECK:  %[[RES:.*]]:2 = scf.for
// CHECK-DAG: %[[V0:.*]] = linalg.elemwise_unary {__op0__}
// CHECK-DAG: %[[V1:.*]] = linalg.elemwise_unary {__op1__} ins(%[[V0]]{{.*}}
// CHECK-DAG: %[[V2:.*]] =  tensor.insert_slice %[[V0]]
// CHECK-DAG: %[[V3:.*]] =  tensor.insert_slice %[[V1]]
// CHECK: return %[[RES]]#{{0|1}}, %[[RES]]#{{0|1}}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops {["func.return"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.fuse_operands %0 {tile_nums = [32]}
  cleanup
}

// -----

func.func @tile_with_interchange(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %0 = linalg.elemwise_unary {__op0__} ins(%arg0 : tensor<64x128xf32>)
                             outs(%arg1: tensor<64x128xf32>) -> tensor<64x128xf32>
  %1 = linalg.elemwise_unary {__op1__} ins(%0 : tensor<64x128xf32>)
                             outs(%arg2: tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0, %1: tensor<64x128xf32>, tensor<64x128xf32>
}
// CHECK-LABEL: func.func @tile_with_interchange
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK:  %[[RES:.*]]:2 = scf.for {{.*}} %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:  {{.*}} scf.for {{.*}} %[[C0]] to %[[C16]] step %[[C1]]
// CHECK-DAG: %[[V0:.*]] = linalg.elemwise_unary {__op0__}
// CHECK-DAG: %[[V1:.*]] = linalg.elemwise_unary {__op1__} ins(%[[V0]]{{.*}}
// CHECK-DAG: %[[V2:.*]] =  tensor.insert_slice %[[V0]]
// CHECK-DAG: %[[V3:.*]] =  tensor.insert_slice %[[V1]]
// CHECK: return %[[RES]]#{{0|1}}, %[[RES]]#{{0|1}}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops {["func.return"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_operands %0 {tile_nums = [32, 16], tile_interchange = [1, 0]}
  cleanup
}

// -----

func.func @all_reduce(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<64x128xf32>) outs(%arg1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %reduced = linalg.reduce { arith.addf } ins(%0 : tensor<64x128xf32>) outs(%arg2 : tensor<64xf32>) dimensions = [1] 
  return %reduced : tensor<64xf32>
}
// CHECK-LABEL: func.func @all_reduce
// CHECK:  {{.*}} scf.for
// CHECK-DAG: %[[V0:.*]] = linalg.elemwise_unary
// CHECK-DAG: %[[V1:.*]] = linalg.reduce { arith.addf } ins(%[[V0]] : tensor<64x4xf32>)
// CHECK-DAG: %[[V2:.*]] = "ccl.all_reduce"(%[[V1]])
// CHECK: scf.yield %[[V2]] : tensor<64xf32>

transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["func.return"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_operands %0 {tile_interchange = [], tile_nums = [1, 32], use_distributed = [0, 1]}
}

// -----

func.func @expect_whole_graph_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = "op_cannot_tile"(%arg0) : (tensor<64x128xf32>) -> (tensor<64x128xf32>)
  %1 = linalg.elemwise_unary {__op0__} ins(%0 : tensor<64x128xf32>)
                             outs(%arg1: tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = linalg.elemwise_unary {__op1__} ins(%1 : tensor<64x128xf32>)
                             outs(%arg2: tensor<64x128xf32>) -> tensor<64x128xf32>
  return %2: tensor<64x128xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops {["func.return"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{tiling and fusion fails}}
  %1, %loops:2 = transform.structured.fuse_operands %0 {tile_nums = [32, 16], tile_interchange = [1, 0], expect_whole_graph_fusion = true}
  cleanup
}

// -----

func.func @only_replace_not_tiled_user(%arg0: tensor<1x1024x16x16xf16>, %arg1: tensor<1x1024x16x16xf16>) -> (tensor<1x1024x16x16xf16>, tensor<1x16x16x1024xf16>) {
  %0 = tensor.empty() : tensor<1x1024x16x16xf16>
  %1 = linalg.elemwise_binary ins(%arg0, %arg1 : tensor<1x1024x16x16xf16>, tensor<1x1024x16x16xf16>) outs(%0 : tensor<1x1024x16x16xf16>) -> tensor<1x1024x16x16xf16>
  %2 = tensor.empty() : tensor<1x16x16x1024xf16>
  %transposed = linalg.transpose ins(%1 : tensor<1x1024x16x16xf16>) outs(%2 : tensor<1x16x16x1024xf16>) permutation = [0, 2, 3, 1] 
  %3 = tensor.empty() : tensor<1x16x16x1024xf16>
  %4 = tensor.empty() : tensor<1x16x16x1024xf16>
  %5 = linalg.elemwise_binary ins(%transposed, %4 : tensor<1x16x16x1024xf16>, tensor<1x16x16x1024xf16>) outs(%3 : tensor<1x16x16x1024xf16>) -> tensor<1x16x16x1024xf16>
  return %1, %5 : tensor<1x1024x16x16xf16>, tensor<1x16x16x1024xf16>
}

// CHECK-LABEL: func.func @only_replace_not_tiled_user
// CHECK:         %[[arg_1:.*]] = linalg.elemwise_binary
// CHECK:         %[[transposed:.*]] = linalg.transpose ins(%[[arg_1]] : tensor<1x1024x16x16xf16>)
// CHECK:         %[[arg_5:.*]] = linalg.elemwise_binary ins(%[[transposed]], {{.*}})
// CHECK:         %[[arg_6:.*]]:2 = scf.for
// CHECK:         return %[[arg_6]]#1, %[[arg_6]]#0

transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["func.return"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_operands %0 {expect_whole_graph_fusion = false, tile_interchange = [], tile_nums = [1, 8], use_distributed = []}
}