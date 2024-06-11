// RUN: byteir-opt -test-print-arg-side-effect -split-input-file %s | FileCheck %s

func.func @all_reduce(%arg0: memref<3xf32> {byre.argname = "in0", byre.argtype = 1: i32}, %arg1: memref<3xf32> {byre.argname = "in1", byre.argtype = 2: i32})  {
  "lccl.all_reduce"(%arg0, %arg1) <{synchronous = true, replica_groups = [[0, 1], [2, 3]], reduction = "sum"}> : (memref<3xf32>, memref<3xf32>) -> ()
  return
}
// CHECK-LABEL: ============= registry of arg side effect =============
// CHECK-LABEL: ============= Test Module =============
// CHECK-NEXT: Testing lccl.all_reduce:
// CHECK-NEXT: arg 0 ArgSideEffectType: kInput
// CHECK-NEXT: arg 1 ArgSideEffectType: kOutput

// -----

module attributes {byre.container_module} {
func.func @byre_compute(%arg0: memref<1024x64xf32, "cuda"> {byre.argname = "in0", byre.argtype = 1: i32}, %arg1: memref<1024x64xf32, "cuda"> {byre.argname = "in1", byre.argtype = 1: i32}, %arg2: memref<1024x64xf32, "cuda"> {byre.argname = "in2", byre.argtype = 2: i32}) attributes {byre.entry_point} {
  byre.compute @PTXOp(%arg0, %arg1, %arg2) {BlockSize.x = 256 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32], call_convention = "bare_ptr", device = "cuda", kernel_name = "Elementwise9", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1024x64xf32, "cuda">, memref<1024x64xf32, "cuda">, memref<1024x64xf32, "cuda">
  return
}
}
// CHECK-LABEL: ============= registry of arg side effect =============
// CHECK-LABEL: ============= Test Module =============
// CHECK-NEXT: Testing byre.compute:
// CHECK-NEXT: arg 0 ArgSideEffectType: kInput
// CHECK-NEXT: arg 1 ArgSideEffectType: kInput
// CHECK-NEXT: arg 2 ArgSideEffectType: kOutput
