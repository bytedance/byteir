// RUN: byteir-opt %s  -lccl-to-byre -split-input-file | FileCheck %s

module attributes {byre.container_module} {
  func.func @send(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}) attributes {byre.entry_point}  {
    "lccl.send"(%arg0) <{synchronous = true, target_index = 0 : i64}> : (memref<3xf32>) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @send(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32> {byre.argname = "in0", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.Send(%[[VAL_0]]) {rank = 0 : i64, synchronous = true} : memref<3xf32>
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @send_dynamic(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}, %target: i64 {byre.argname = "in1", byre.argtype = 1 : i32}) attributes {byre.entry_point}  {
    "lccl.send"(%arg0, %target) <{synchronous = true}> : (memref<3xf32>, i64) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @send_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32> {byre.argname = "in0", byre.argtype = 1 : i32},
// CHECK-SAME:      %[[VAL_1:.*]]: i64 {byre.argname = "in1", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.Send(%[[VAL_0]], %[[VAL_1]]) {synchronous = true} : memref<3xf32>, i64
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @recv(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}) attributes {byre.entry_point}  {
    "lccl.recv"(%arg0) <{synchronous = true, source_index = 0 : i64}> : (memref<3xf32>) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @recv(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32> {byre.argname = "in0", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.Recv(%[[VAL_0]]) {rank = 0 : i64, synchronous = true} : memref<3xf32>
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @recv_dynamic(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}, %target: i64 {byre.argname = "in1", byre.argtype = 1 : i32}) attributes {byre.entry_point}  {
    "lccl.recv"(%arg0, %target) <{synchronous = true}> : (memref<3xf32>, i64) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @recv_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32> {byre.argname = "in0", byre.argtype = 1 : i32},
// CHECK-SAME:      %[[VAL_1:.*]]: i64 {byre.argname = "in1", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.Recv(%[[VAL_0]], %[[VAL_1]]) {synchronous = true} : memref<3xf32>, i64
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @broadcast(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}) attributes {byre.entry_point}  {
    "lccl.broadcast"(%arg0) <{synchronous = true, replica_groups = [[1, 2], [3, 4]]}> : (memref<3xf32>) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @broadcast(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32> {byre.argname = "in0", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.Broadcast(%[[VAL_0]]) {replica_group = [1, 2], synchronous = true} : memref<3xf32>
// CHECK:           byre.compute @nccl.Broadcast(%[[VAL_0]]) {replica_group = [3, 4], synchronous = true} : memref<3xf32>
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @broadcast_dynamic(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}, %arg1: memref<1x4xindex> {byre.argname = "in1", byre.argtype = 1: i32}) attributes {byre.entry_point}  {
    "lccl.broadcast"(%arg0, %arg1) <{synchronous = true}> : (memref<3xf32>, memref<1x4xindex>) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @broadcast_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32> {byre.argname = "in0", byre.argtype = 1 : i32},
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x4xindex> {byre.argname = "in1", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.Broadcast(%[[VAL_0]], %[[VAL_1]]) {synchronous = true} : memref<3xf32>, memref<1x4xindex>
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @all_reduce(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}, %arg1: memref<3xf32> {byre.argname = "in1", byre.argtype = 2: i32}) attributes {byre.entry_point}  {
    "lccl.all_reduce"(%arg0, %arg1) <{synchronous = true, replica_groups = [[0, 1], [2, 3]], reduction = "sum"}> : (memref<3xf32>, memref<3xf32>) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @all_reduce(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32> {byre.argname = "in0", byre.argtype = 1 : i32},
// CHECK-SAME:      %[[VAL_1:.*]]: memref<3xf32> {byre.argname = "in1", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.AllReduce(%[[VAL_0]], %[[VAL_1]]) {reduction = "sum", replica_group = [0, 1], synchronous = true} : memref<3xf32>, memref<3xf32>
// CHECK:           byre.compute @nccl.AllReduce(%[[VAL_0]], %[[VAL_1]]) {reduction = "sum", replica_group = [2, 3], synchronous = true} : memref<3xf32>, memref<3xf32>
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @all_reduce_dynamic(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}, %arg1: memref<3xf32> {byre.argname = "in1", byre.argtype = 2: i32}, %arg2: memref<2x2xindex> {byre.argname = "in2", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    "lccl.all_reduce"(%arg0, %arg1, %arg2) <{synchronous = true, reduction = "sum"}> : (memref<3xf32>, memref<3xf32>, memref<2x2xindex>) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @all_reduce_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3xf32> {byre.argname = "in0", byre.argtype = 1 : i32},
// CHECK-SAME:      %[[VAL_1:.*]]: memref<3xf32> {byre.argname = "in1", byre.argtype = 2 : i32},
// CHECK-SAME:      %[[VAL_2:.*]]: memref<2x2xindex> {byre.argname = "in2", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.AllReduce(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {reduction = "sum", synchronous = true} : memref<3xf32>, memref<3xf32>, memref<2x2xindex>
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @all_gather(%arg0: memref<4x4xf32> {byre.argname = "in0", byre.argtype = 1: i32}, %arg1: memref<8x4xf32> {byre.argname = "in1", byre.argtype = 1: i32}) attributes {byre.entry_point}{
    "lccl.all_gather"(%arg0, %arg1) <{axis = 0 : i64, replica_groups = [[0, 1], [2, 3]], synchronous = true}> : (memref<4x4xf32>, memref<8x4xf32>) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @all_gather(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32> {byre.argname = "in0", byre.argtype = 1 : i32},
// CHECK-SAME:      %[[VAL_1:.*]]: memref<8x4xf32> {byre.argname = "in1", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.AllGather(%[[VAL_0]], %[[VAL_1]]) {axis = 0 : i64, replica_group = [0, 1], synchronous = true} : memref<4x4xf32>, memref<8x4xf32>
// CHECK:           byre.compute @nccl.AllGather(%[[VAL_0]], %[[VAL_1]]) {axis = 0 : i64, replica_group = [2, 3], synchronous = true} : memref<4x4xf32>, memref<8x4xf32>
// CHECK:           return
// CHECK:         }

// -----

module attributes {byre.container_module} {
  func.func @all_gather_dynamic(%arg0: memref<4x4xf32> {byre.argname = "in0", byre.argtype = 1 : i32}, %arg1: memref<8x4xf32> {byre.argname = "in1", byre.argtype = 2 : i32}, %arg2: memref<2x2xindex> {byre.argname = "in2", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
    "lccl.all_gather"(%arg0, %arg1, %arg2) <{axis = 0 : i64, synchronous = true}> : (memref<4x4xf32>, memref<8x4xf32>, memref<2x2xindex>) -> ()
    return
  }
}
// CHECK-LABEL:   func.func @all_gather_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<4x4xf32> {byre.argname = "in0", byre.argtype = 1 : i32},
// CHECK-SAME:      %[[VAL_1:.*]]: memref<8x4xf32> {byre.argname = "in1", byre.argtype = 2 : i32},
// CHECK-SAME:      %[[VAL_2:.*]]: memref<2x2xindex> {byre.argname = "in2", byre.argtype = 1 : i32}) attributes {byre.entry_point} {
// CHECK:           byre.compute @nccl.AllGather(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {axis = 0 : i64, synchronous = true} : memref<4x4xf32>, memref<8x4xf32>, memref<2x2xindex>
// CHECK:           return
// CHECK:         }
