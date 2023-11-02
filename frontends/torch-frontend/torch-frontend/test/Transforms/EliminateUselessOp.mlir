// RUN: torch-frontend-opt %s --eliminate-useless-op --canonicalize | FileCheck %s

module {
  func.func @forward() -> () {
    %str_0 = torch.constant.str "profiler_name"
    %none = torch.constant.none
    %0 = torch.operator "profiler._record_function_enter"(%str_0, %none) : (!torch.str, !torch.none) -> !torch.tensor
    torch.operator "profiler._record_function_exit"(%0) : (!torch.tensor) -> ()
    return
  }
}
// CHECK-LABEL: func.func @forward
// CHECK-NOT: torch.operator "profiler._record_function_enter"
// CHECK-NOT: torch.operator "profiler._record_function_exit"
