// RUN: torch-frontend-opt %s --unpack-public-function-return --canonicalize | FileCheck %s

module {
  func.func @forward(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[3,4],f32>}) -> !torch.list<tensor> {
    %0 = torch.prim.ListConstruct %arg0, %arg0, %arg0 : (!torch.tensor, !torch.tensor, !torch.tensor) -> !torch.list<tensor>
    return %0 : !torch.list<tensor>
  }
}
// CHECK-LABEL: func.func @forward
// CHECK: %0 = torch.prim.TupleConstruct %arg0, %arg0, %arg0 : !torch.tensor, !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor, tensor>
// CHECK: return %0 : !torch.tuple<tensor, tensor, tensor>
