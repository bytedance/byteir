// RUN: torch-frontend-opt %s --rewrite-entry-func-name="target-name=main" | FileCheck %s

module {
  func.func @forward(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>{
    return %arg0 : !torch.vtensor<[3,4],f32>
  }
}
// CHECK-LABEL: func.func @main
