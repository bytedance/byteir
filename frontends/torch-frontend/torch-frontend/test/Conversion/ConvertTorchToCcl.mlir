// RUN: torch-frontend-opt %s -convert-torch-to-ccl | FileCheck %s

func.func @torch.c10d_functional.all_reduce(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>) {
  %str = torch.constant.str "sum"
  %str_0 = torch.constant.str ""
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.all_reduce %arg0, %str, %str_0, %0, %int4 : !torch.vtensor<[4],f32>, !torch.str, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[4],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %2 : !torch.vtensor<[4],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.all_reduce
// CHECK: ccl.all_reduce
// CHECK-SAME{LITERAL}: {reduction = "sum", replica_groups = [[0, 1, 2, 3]], synchronous = false}
// CHECK: ccl.wait

func.func @torch.c10d_functional.all_gather_into_tensor(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[16],f32>) {
  %str_0 = torch.constant.str ""
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.all_gather_into_tensor %arg0, %str_0, %0, %int4 : !torch.vtensor<[4],f32>, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[16],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
  return %2 : !torch.vtensor<[16],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.all_gather_into_tensor
// CHECK: ccl.all_gather
// CHECK-SAME{LITERAL}: {axis = 0 : i64, replica_groups = [[0, 1, 2, 3]], synchronous = false}
// CHECK: ccl.wait

func.func @torch.c10d_functional.reduce_scatter_tensor(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[1],f32>) {
  %str = torch.constant.str "sum"
  %str_0 = torch.constant.str ""
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.reduce_scatter_tensor %arg0, %str, %str_0, %0, %int4 : !torch.vtensor<[4],f32>, !torch.str, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[1],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[1],f32> -> !torch.vtensor<[1],f32>
  return %2 : !torch.vtensor<[1],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.reduce_scatter_tensor
// CHECK: ccl.reduce_scatter
// CHECK-SAME{LITERAL}: {axis = 0 : i64, reduction = "sum", replica_groups = [[0, 1, 2, 3]], synchronous = false}
// CHECK: ccl.wait

func.func @torch.c10d_functional.send(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>) {
  %str_0 = torch.constant.str ""
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.send %arg0, %int1, %str_0, %0, %int4 : !torch.vtensor<[4],f32>, !torch.int, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[4],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %2 : !torch.vtensor<[4],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.send
// CHECK: ccl.send
// CHECK-SAME{LITERAL}: {synchronous = true, target_index = 1 : i64}
// CHECK: ccl.wait

func.func @torch.c10d_functional.recv(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>) {
  %str_0 = torch.constant.str ""
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.recv %arg0, %int0, %str_0, %0, %int4 : !torch.vtensor<[4],f32>, !torch.int, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[4],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %2 : !torch.vtensor<[4],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.recv
// CHECK: ccl.recv
// CHECK-SAME{LITERAL}: {source_index = 0 : i64, synchronous = true}
// CHECK: ccl.wait

func.func @torch.c10d_functional.isend(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>) {
  %str_0 = torch.constant.str ""
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.isend %arg0, %int1, %str_0, %0, %int4 : !torch.vtensor<[4],f32>, !torch.int, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[4],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %2 : !torch.vtensor<[4],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.isend
// CHECK: ccl.send
// CHECK-SAME{LITERAL}: {synchronous = false, target_index = 1 : i64}
// CHECK: ccl.wait

func.func @torch.c10d_functional.irecv(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>) {
  %str_0 = torch.constant.str ""
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.irecv %arg0, %int0, %str_0, %0, %int4 : !torch.vtensor<[4],f32>, !torch.int, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[4],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %2 : !torch.vtensor<[4],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.irecv
// CHECK: ccl.recv
// CHECK-SAME{LITERAL}: {source_index = 0 : i64, synchronous = false}
// CHECK: ccl.wait

func.func @torch.c10d_functional.irecv.dynamic_shape(%arg0: !torch.vtensor<[?],f32>) -> (!torch.vtensor<[?],f32>) {
  %str_0 = torch.constant.str ""
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.irecv %arg0, %int0, %str_0, %0, %int4 : !torch.vtensor<[?],f32>, !torch.int, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[?],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[?],f32> -> !torch.vtensor<[?],f32>
  return %2 : !torch.vtensor<[?],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.irecv.dynamic_shape
// CHECK: ccl.recv
// CHECK-SAME{LITERAL}: {source_index = 0 : i64, synchronous = false}
// CHECK: ccl.wait

func.func @torch.c10d_functional.broadcast(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>) {
  %str = torch.constant.str "ptd:4"
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %1 = torch.c10d_functional.broadcast %arg0, %int3, %str, %0, %int4 : !torch.vtensor<[4],f32>, !torch.int, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[4],f32>
  %2 = torch.c10d_functional.wait_tensor %1 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %2 : !torch.vtensor<[4],f32>
}
// CHECK-LABEL: func.func @torch.c10d_functional.broadcast
// CHECK: ccl.broadcast
// CHECK-SAME{LITERAL}: {replica_groups = [[3, 0, 1, 2]], synchronous = false}
// CHECK: ccl.wait
