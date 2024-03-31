// RUN: torch-frontend-opt %s --eliminate-useless-op --canonicalize | FileCheck %s

func.func @profiler() -> () {
  %str_0 = torch.constant.str "profiler_name"
  %none = torch.constant.none
  %0 = torch.operator "profiler._record_function_enter"(%str_0, %none) : (!torch.str, !torch.none) -> !torch.tensor
  torch.operator "profiler._record_function_exit"(%0) : (!torch.tensor) -> ()
  return
}
// CHECK-LABEL: func.func @profiler
// CHECK-NOT: torch.operator "profiler._record_function_enter"
// CHECK-NOT: torch.operator "profiler._record_function_exit"

func.func @aten_warn() -> () {
  %str = torch.constant.str "warn message"
  %int0 = torch.constant.int 0
  torch.aten.warn %str, %int0 : !torch.str, !torch.int
  return
}
// CHECK-LABEL: func.func @aten_warn
// CHECK-NOT: torch.aten.warn
