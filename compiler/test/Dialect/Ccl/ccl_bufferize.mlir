// RUN: byteir-opt %s  -byteir-one-shot-bufferize -split-input-file | FileCheck %s

func.func @broadcast(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = "ccl.broadcast"(%arg0) {replica_groups = [[2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>   
  return %0 : tensor<2x3x8xf32>
}

// CHECK-LABEL:   func.func @broadcast(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<2x3x8xf32>) -> memref<2x3x8xf32> {
// CHECK:           "lccl.broadcast"(%[[VAL_0]]) <{replica_groups = {{\[\[}}2, 3]], synchronous = true}> : (memref<2x3x8xf32>) -> ()
// CHECK:           return %[[VAL_0]] : memref<2x3x8xf32>
// CHECK:         }

// -----

func.func @broadcast_dynamic(%arg0: tensor<2x3x8xf32>, %arg1: tensor<1x4xindex>) -> tensor<2x3x8xf32> {
  %0 = "ccl.broadcast"(%arg0, %arg1) {synchronous = true} : (tensor<2x3x8xf32>, tensor<1x4xindex>) -> tensor<2x3x8xf32>   
  return %0 : tensor<2x3x8xf32>
}
// CHECK-LABEL:   func.func @broadcast_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<2x3x8xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x4xindex>) -> memref<2x3x8xf32> {
// CHECK:           "lccl.broadcast"(%[[VAL_0]], %[[VAL_1]]) <{synchronous = true}> : (memref<2x3x8xf32>, memref<1x4xindex>) -> ()
// CHECK:           return %[[VAL_0]] : memref<2x3x8xf32>
// CHECK:         }

// -----

func.func @send(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "ccl.send"(%arg0){ synchronous = true, target_index = 0 : i64 }: (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
// CHECK-LABEL:   func.func @send(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32>) -> memref<3xf32> {
// CHECK:           "lccl.send"(%[[VAL_0]]) <{synchronous = true, target_index = 0 : i64}> : (memref<3xf32>) -> ()
// CHECK:           return %[[VAL_0]] : memref<3xf32>
// CHECK:         }

// -----

func.func @send_dynamic(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %target_index = arith.constant 0 : i64
  %0 = "ccl.send"(%arg0, %target_index) { synchronous = true } : (tensor<3xf32>, i64) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
// CHECK-LABEL:   func.func @send_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32>) -> memref<3xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           "lccl.send"(%[[VAL_0]], %[[VAL_1]]) <{synchronous = true}> : (memref<3xf32>, i64) -> ()
// CHECK:           return %[[VAL_0]] : memref<3xf32>
// CHECK:         }

// -----

func.func @recv(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "ccl.recv"(%arg0){ synchronous = true, source_index = 0 : i64 } : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
// CHECK-LABEL:   func.func @recv(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32>) -> memref<3xf32> {
// CHECK:           "lccl.recv"(%[[VAL_0]]) <{source_index = 0 : i64, synchronous = true}> : (memref<3xf32>) -> ()
// CHECK:           return %[[VAL_0]] : memref<3xf32>
// CHECK:         }

// -----

func.func @recv_dynamic(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %target_index = arith.constant 0 : i64
    %0 = "ccl.recv"(%arg0, %target_index) { synchronous = true } : (tensor<3xf32>, i64) -> tensor<3xf32>
    return %0 : tensor<3xf32>
}

// CHECK-LABEL:   func.func @recv_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32>) -> memref<3xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           "lccl.recv"(%[[VAL_0]], %[[VAL_1]]) <{synchronous = true}> : (memref<3xf32>, i64) -> ()
// CHECK:           return %[[VAL_0]] : memref<3xf32>
// CHECK:         }

// -----

func.func @all_gather_0(%arg0: tensor<4x4xf32>) -> tensor<8x4xf32> {
    %0 = "ccl.all_gather"(%arg0) { replica_groups = [[0, 1] ,[2, 3]], axis = 0 : i64 , synchronous = true }: (tensor<4x4xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
}
// CHECK-LABEL:   func.func @all_gather_0(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32>) -> memref<8x4xf32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<8x4xf32>
// CHECK:           "lccl.all_gather"(%[[VAL_0]], %[[VAL_1]]) <{axis = 0 : i64, replica_groups = {{\[\[}}0, 1], [2, 3]], synchronous = true}> : (memref<4x4xf32>, memref<8x4xf32>) -> ()
// CHECK:           return %[[VAL_1]] : memref<8x4xf32>
// CHECK:         }

// -----

func.func @all_gather_1(%arg0: tensor<4x4xf32>) -> tensor<4x8xf32> {
    %0 = "ccl.all_gather"(%arg0) { replica_groups = [[0, 1] ,[2, 3]], axis = 1 : i64 , synchronous = true }: (tensor<4x4xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
}
// CHECK-LABEL:   func.func @all_gather_1(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32>) -> memref<4x8xf32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<4x8xf32>
// CHECK:           "lccl.all_gather"(%[[VAL_0]], %[[VAL_1]]) <{axis = 1 : i64, replica_groups = {{\[\[}}0, 1], [2, 3]], synchronous = true}> : (memref<4x4xf32>, memref<4x8xf32>) -> ()
// CHECK:           return %[[VAL_1]] : memref<4x8xf32>
// CHECK:         }

// -----

func.func @all_gather_dynamic_0(%arg0: tensor<4x4xf32>, %arg1: tensor<2x2xindex>) -> tensor<8x4xf32> {
    %0 = "ccl.all_gather"(%arg0, %arg1) {axis=0 : i64, synchronous=true}: (tensor<4x4xf32>, tensor<2x2xindex>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
}
// CHECK-LABEL:   func.func @all_gather_dynamic_0(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<2x2xindex>) -> memref<8x4xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<8x4xf32>
// CHECK:           "lccl.all_gather"(%[[VAL_0]], %[[VAL_2]], %[[VAL_1]]) <{axis = 0 : i64, synchronous = true}> : (memref<4x4xf32>, memref<8x4xf32>, memref<2x2xindex>) -> ()
// CHECK:           return %[[VAL_2]] : memref<8x4xf32>
// CHECK:         }

// -----

func.func @all_gather_dynamic_1(%arg0: tensor<4x4xf32>, %arg1: tensor<2x2xindex>) -> tensor<4x8xf32> {
    %0 = "ccl.all_gather"(%arg0, %arg1) {axis=1 : i64, synchronous=true}: (tensor<4x4xf32>, tensor<2x2xindex>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
}
// CHECK-LABEL:   func.func @all_gather_dynamic_1(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<2x2xindex>) -> memref<4x8xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<4x8xf32>
// CHECK:           "lccl.all_gather"(%[[VAL_0]], %[[VAL_2]], %[[VAL_1]]) <{axis = 1 : i64, synchronous = true}> : (memref<4x4xf32>, memref<4x8xf32>, memref<2x2xindex>) -> ()
// CHECK:           return %[[VAL_2]] : memref<4x8xf32>
// CHECK:         }

// -----

func.func @all_reduce(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "ccl.all_reduce"(%arg0) {reduction = "sum", synchronous=true, replica_groups = [[0, 1] ,[2, 3]]}: (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
// CHECK-LABEL:   func.func @all_reduce(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4xf32>) -> memref<4xf32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           "lccl.all_reduce"(%[[VAL_0]], %[[VAL_1]]) <{reduction = "sum", replica_groups = {{\[\[}}0, 1], [2, 3]], synchronous = true}> : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK:           return %[[VAL_1]] : memref<4xf32>
// CHECK:         }

// -----

func.func @all_reduce_dynamic(%arg0: tensor<4xf32>, %arg1:tensor<1x4xi64>) -> tensor<4xf32> {
    %0 = "ccl.all_reduce"(%arg0, %arg1) {reduction = "sum", synchronous=true}: (tensor<4xf32>, tensor<1x4xi64>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
// CHECK-LABEL:   func.func @all_reduce_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x4xi64>) -> memref<4xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           "lccl.all_reduce"(%[[VAL_0]], %[[VAL_2]], %[[VAL_1]]) <{reduction = "sum", synchronous = true}> : (memref<4xf32>, memref<4xf32>, memref<1x4xi64>) -> ()
// CHECK:           return %[[VAL_2]] : memref<4xf32>
// CHECK:         }

// -----

func.func @reduce_scatter_0(%arg0: tensor<4x4xf32>) -> tensor<1x4xf32> {
    %0 = "ccl.reduce_scatter"(%arg0) { reduction="sum", replica_groups = [[0, 1, 2, 3]], axis = 0 : i64 , synchronous=true } : (tensor<4x4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
}

// CHECK-LABEL:   func.func @reduce_scatter_0(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32>) -> memref<1x4xf32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<1x4xf32>
// CHECK:           "lccl.reduce_scatter"(%[[VAL_0]], %[[VAL_1]]) <{axis = 0 : i64, reduction = "sum", replica_groups = {{\[\[}}0, 1, 2, 3]], synchronous = true}> : (memref<4x4xf32>, memref<1x4xf32>) -> ()
// CHECK:           return %[[VAL_1]] : memref<1x4xf32>
// CHECK:         }

// -----

func.func @reduce_scatter_1(%arg0: tensor<4x4xf32>) -> tensor<4x1xf32> {
    %0 = "ccl.reduce_scatter"(%arg0) { reduction="sum", replica_groups = [[0, 1, 2, 3]], axis = 1 : i64 , synchronous=true } : (tensor<4x4xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
}

// CHECK-LABEL:   func.func @reduce_scatter_1(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32>) -> memref<4x1xf32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<4x1xf32>
// CHECK:           "lccl.reduce_scatter"(%[[VAL_0]], %[[VAL_1]]) <{axis = 1 : i64, reduction = "sum", replica_groups = {{\[\[}}0, 1, 2, 3]], synchronous = true}> : (memref<4x4xf32>, memref<4x1xf32>) -> ()
// CHECK:           return %[[VAL_1]] : memref<4x1xf32>
// CHECK:         }

// -----

func.func @reduce_scatter_dynamic_0(%arg0: tensor<4x4xf32>, %arg1: tensor<2x2xindex>) -> tensor<2x4xf32> {
    %0 = "ccl.reduce_scatter"(%arg0, %arg1) { axis = 0 : i64, synchronous = true, reduction = "sum" }: (tensor<4x4xf32>, tensor<2x2xindex>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
}
// CHECK-LABEL:   func.func @reduce_scatter_dynamic_0(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<2x2xindex>) -> memref<2x4xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           "lccl.reduce_scatter"(%[[VAL_0]], %[[VAL_2]], %[[VAL_1]]) <{axis = 0 : i64, reduction = "sum", synchronous = true}> : (memref<4x4xf32>, memref<2x4xf32>, memref<2x2xindex>) -> ()
// CHECK:           return %[[VAL_2]] : memref<2x4xf32>
// CHECK:         }

// -----

func.func @reduce_scatter_dynamic_1(%arg0: tensor<4x4xf32>, %arg1: tensor<2x2xindex>) -> tensor<4x2xf32> {
    %0 = "ccl.reduce_scatter"(%arg0, %arg1) { axis=1 : i64, synchronous=true, reduction= "sum" } : (tensor<4x4xf32>, tensor<2x2xindex>) -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
}
// CHECK-LABEL:   func.func @reduce_scatter_dynamic_1(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<2x2xindex>) -> memref<4x2xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<4x2xf32>
// CHECK:           "lccl.reduce_scatter"(%[[VAL_0]], %[[VAL_2]], %[[VAL_1]]) <{axis = 1 : i64, reduction = "sum", synchronous = true}> : (memref<4x4xf32>, memref<4x2xf32>, memref<2x2xindex>) -> ()
// CHECK:           return %[[VAL_2]] : memref<4x2xf32>
// CHECK:         }
