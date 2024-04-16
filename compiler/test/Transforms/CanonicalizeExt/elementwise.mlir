// RUN: byteir-opt %s -canonicalize-ext="blind-fold=true" | FileCheck %s

// FIXME: make constant really large or trigger canonicalize-ext anywhy.
func.func @fold_large_constant_binary_op() -> tensor<2xf32> {
  %0 = mhlo.constant dense<[0.00000e+0, 1.00000e+0]> : tensor<2xf32>
  %1 = mhlo.constant dense<[1.00000e+0, 1.00000e+0]> : tensor<2xf32>
  %2 = mhlo.add %0, %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}
// CHECK-LABEL: fold_large_constant_binary_op
// CHECK-NOT: mhlo.add
// CHECK: mhlo.constant dense<[1.000000e+00, 2.000000e+00]>

func.func @eliminate_redundant_convert(%arg0: tensor<12xi1>) -> (tensor<12xi4>) {
  %1 = mhlo.convert %arg0 : (tensor<12xi1>) -> tensor<12xf32>
  %result = mhlo.convert %1 : (tensor<12xf32>) -> tensor<12xi4>
  return %result : tensor<12xi4>
}
// CHECK-LABEL: eliminate_redundant_convert
// CHECK: mhlo.convert
// CHECK-NEXT: return

func.func @fold_clamp$case0() -> tensor<5xi64> {
  %1 = mhlo.constant dense<[-1, 100, 200, 0, 149]> : tensor<5xi64>
  %2 = mhlo.constant dense<149> : tensor<i64>
  %3 = mhlo.constant dense<0> : tensor<i64>
  %4 = mhlo.clamp %3, %1, %2 : (tensor<i64>, tensor<5xi64>, tensor<i64>) -> tensor<5xi64>
  return %4 : tensor<5xi64>
}
// CHECK-LABEL: fold_clamp$case0
// CHECK: mhlo.constant dense<[0, 100, 149, 0, 149]> : tensor<5xi64>
// CHECK-NOT: mhlo.clamp

func.func @fold_clamp$case1() -> tensor<5xi64> {
  %0 = mhlo.constant dense<[149, 101, -1,  30, 50]> : tensor<5xi64>
  %1 = mhlo.constant dense<[-1,  100, 200, 0,  149]> : tensor<5xi64>
  %2 = mhlo.constant dense<[0,   10,  -10, 10, -100]> : tensor<5xi64>
  %3 = mhlo.clamp %2, %1, %0 : (tensor<5xi64>, tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
  return %3 : tensor<5xi64>
}
// CHECK-LABEL: fold_clamp$case1
// CHECK{LITERAL}: mhlo.constant dense<[0, 100, -1, 10, 50]>
// CHECK-NOT: mhlo.clamp

func.func @fold_clamp$case2() -> tensor<6xf32> {
  %0 = mhlo.constant dense<[5.0, 66.0, 0xFFFFFFFF, -2.0,       0xFFFFFFFF, 6.0]> : tensor<6xf32>
  %1 = mhlo.constant dense<[5.0, 3.0,  2.0,        0xFFFFFFFF, 0xFFFFFFFF, 4.0]> : tensor<6xf32>
  %2 = mhlo.constant dense<[5.0, 1.0,  1.0,        0xFFFFFFFF, 0xFFFFFFFF, 5.0]> : tensor<6xf32>
  %3 = mhlo.clamp %2, %1, %0 : (tensor<6xf32>, tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %3 : tensor<6xf32>
}
// CHECK-LABEL: fold_clamp$case2
// CHECK{LITERAL}: mhlo.constant dense<[5.000000e+00, 3.000000e+00, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 5.000000e+00]
// CHECK-NOT: mhlo.clamp

func.func @simplify_byteir_addn(%arg0: tensor<150x768xf16>, %arg1: tensor<150x768xf16>) -> tensor<150x768xf16> {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {api_version = 1 : i32, backend_config = "", byteir_attrs = {_grappler_ArithmeticOptimizer_AddOpsRewriteStage = true}, call_target_name = "byteir.addn", called_computations = [], has_side_effect = false, output_operand_aliases = []} : (tensor<150x768xf16>, tensor<150x768xf16>) -> tensor<150x768xf16>
  return %0 : tensor<150x768xf16>
}
// CHECK-LABEL: simplify_byteir_addn
// CHECK-NOT: mhlo.custom_call
// CHECK: mhlo.add

func.func @multiply_zero(%arg0: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> {
  %c0 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
  %1 = "mhlo.multiply"(%arg0, %c0) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
  return %1: tensor<2x128x128xf32>
}
// CHECK-LABEL: multiply_zero
// CHECK: mhlo.constant
// CHECK-NOT: mhlo.multiply
