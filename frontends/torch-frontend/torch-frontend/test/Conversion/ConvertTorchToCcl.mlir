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
// CHECK-SAME{LITERAL}: {reduction = "sum", replica_groups = [[0, 1, 2, 3]]}
// CHECK: ccl.wait_tensor
