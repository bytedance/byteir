// RUN: torch-frontend-opt %s --decompose-on-torch --canonicalize | FileCheck %s

func.func @torch.aten.var.dim(%arg0: !torch.vtensor<[50,100],f32>) -> (!torch.vtensor<[50],f32>) {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.var.dim %arg0, %0, %false, %false : !torch.vtensor<[50,100],f32>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[50],f32>
    return %1 : !torch.vtensor<[50],f32>
}
// CHECK-LABEL: func.func @torch.aten.var.dim
// CHECK:  %[[VAL0:.*]] = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
// CHECK:  %[[VAL1:.*]] = torch.aten.mean.dim %arg0, %[[VAL0]], %true, %none : !torch.vtensor<[50,100],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[50,1],f32>
// CHECK:  %[[VAL2:.*]] = torch.aten.sub.Tensor %arg0, %[[VAL1]], %float1.000000e00 : !torch.vtensor<[50,100],f32>, !torch.vtensor<[50,1],f32>, !torch.float -> !torch.vtensor<[50,100],f32>
// CHECK:  %[[VAL3:.*]] = torch.aten.square %[[VAL2]] : !torch.vtensor<[50,100],f32> -> !torch.vtensor<[50,100],f32>
// CHECK:  %[[VAL4:.*]] = torch.aten.mean.dim %[[VAL3]], %[[VAL0]], %false, %none : !torch.vtensor<[50,100],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[50],f32>
// CHECK:  return %[[VAL4]] : !torch.vtensor<[50],f32>

func.func @torch.aten.var.dim$keepdim(%arg0: !torch.vtensor<[50,100],f32>) -> (!torch.vtensor<[50,1],f32>) {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.var.dim %arg0, %0, %false, %true : !torch.vtensor<[50,100],f32>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[50,1],f32>
    return %1 : !torch.vtensor<[50,1],f32>
}
// CHECK-LABEL: func.func @torch.aten.var.dim$keepdim
// CHECK:  %[[VAL0:.*]] = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
// CHECK:  %[[VAL1:.*]] = torch.aten.mean.dim %arg0, %[[VAL0]], %true, %none : !torch.vtensor<[50,100],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[50,1],f32>
// CHECK:  %[[VAL2:.*]] = torch.aten.sub.Tensor %arg0, %[[VAL1]], %float1.000000e00 : !torch.vtensor<[50,100],f32>, !torch.vtensor<[50,1],f32>, !torch.float -> !torch.vtensor<[50,100],f32>
// CHECK:  %[[VAL3:.*]] = torch.aten.square %[[VAL2]] : !torch.vtensor<[50,100],f32> -> !torch.vtensor<[50,100],f32>
// CHECK:  %[[VAL4:.*]] = torch.aten.mean.dim %[[VAL3]], %[[VAL0]], %true, %none : !torch.vtensor<[50,100],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[50,1],f32>
// CHECK:  return %[[VAL4]] : !torch.vtensor<[50,1],f32>
