// RUN: torch-frontend-opt -pass-pipeline='builtin.module(torch-function-to-torch-pipeline{backend-legal-ops=torch.aten.square,torch.aten.argmax})' -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @torch.custom.dynamic_partition_stitch
// CHECK:        torch.custom_op
// CHECK-SAME:   custom_op_name = "dynamic_partition"
// CHECK:        torch.custom_op
// CHECK-SAME:   custom_op_name = "dynamic_stitch"
// CHECK:        return
// CHECK-SAME:   : !torch.vtensor<[10,5],f32>
func.func @torch.custom.dynamic_partition_stitch(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[10,5],f32>}, %arg1: !torch.tensor {torch.type_bound = !torch.vtensor<[10],si64>}, %arg2: !torch.tensor {torch.type_bound = !torch.vtensor<[6],si64>}, %arg3: !torch.tensor {torch.type_bound = !torch.vtensor<[4],si64>}) -> !torch.tensor {
  %int2 = torch.constant.int 2
  %int10 = torch.constant.int 10
  %int5 = torch.constant.int 5
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.custom.dynamic_partition %arg0, %arg1, %int2 : !torch.tensor, !torch.tensor, !torch.int -> !torch.list<tensor>
  %1 = torch.prim.ListConstruct %int10, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %arg2, %arg3 : (!torch.tensor, !torch.tensor) -> !torch.list<tensor>
  %3 = torch.aten.__getitem__.t %0, %int0 : !torch.list<tensor>, !torch.int -> !torch.tensor
  %4 = torch.aten.__getitem__.t %0, %int1 : !torch.list<tensor>, !torch.int -> !torch.tensor
  %5 = torch.prim.ListConstruct %3, %4 : (!torch.tensor, !torch.tensor) -> !torch.list<tensor>
  %6 = torch.custom.dynamic_stitch %2, %5, %1 : !torch.list<tensor>, !torch.list<tensor>, !torch.list<int> -> !torch.tensor
  return %6 : !torch.tensor
}
