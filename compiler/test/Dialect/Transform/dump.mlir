// RUN: byteir-opt %s --transform-dialect-interpreter --split-input-file | FileCheck %s
// RUN: byteir-opt %s --transform-dialect-interpreter --split-input-file -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUMP

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  dump "before cleanup"
  cleanup
}

// CHECK-LABEL: @cse_many
func.func @cse_many(f32, f32) -> (f32) {
^bb0(%a : f32, %b : f32):
  // CHECK-NEXT: %[[VAR_0:[0-9a-zA-Z_]+]] = arith.addf %{{.*}}, %{{.*}} : f32
  %c = arith.addf %a, %b : f32
  %d = arith.addf %a, %b : f32
  %e = arith.addf %a, %b : f32
  %f = arith.addf %a, %b : f32

  // CHECK-NEXT: %[[VAR_1:[0-9a-zA-Z_]+]] = arith.addf %[[VAR_0]], %[[VAR_0]] : f32
  %g = arith.addf %c, %d : f32
  %h = arith.addf %e, %f : f32
  %i = arith.addf %c, %e : f32

  // CHECK-NEXT: %[[VAR_2:[0-9a-zA-Z_]+]] = arith.addf %[[VAR_1]], %[[VAR_1]] : f32
  %j = arith.addf %g, %h : f32
  %k = arith.addf %h, %i : f32

  // CHECK-NEXT: %[[VAR_3:[0-9a-zA-Z_]+]] = arith.addf %[[VAR_2]], %[[VAR_2]] : f32
  %l = arith.addf %j, %k : f32

  // CHECK-NEXT: return %[[VAR_3]] : f32
  return %l : f32
}

func.func @canonicalize_eliminate_splat_constant_transpose() -> tensor<2x1x4x3xi32> {
  %0 = mhlo.constant dense<0> : tensor<1x2x3x4xi32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %1: tensor<2x1x4x3xi32>
}
// CHECK-LABEL: canonicalize_eliminate_splat_constant_transpose
// CHECK-NEXT: %0 = mhlo.constant dense<0> : tensor<2x1x4x3xi32>


// CHECK-LABEL: func @sccp_no_control_flow
func.func @sccp_no_control_flow(%arg0: i32) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: return %[[CST]] : i32

  %cond = arith.constant true
  %cst_1 = arith.constant 1 : i32
  %select = arith.select %cond, %cst_1, %arg0 : i32
  return %select : i32
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match attributes{"__test__"} in %arg0
    transform.dump(%0 : !pdl.operation) "Done"
}

func.func @foo(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) {
  %0 = mhlo.add %arg0, %arg1 {__test__} : tensor<?x?xf32>
  %1 = mhlo.multiply %arg0, %arg1 : tensor<?x?xf32>
  return
}
// CHECK-LABEL: foo

// DUMP-LABEL: Done
// DUMP-NOT: foo
// DUMP: mhlo.add
// DUMP-NOT: mhlo.multiply
