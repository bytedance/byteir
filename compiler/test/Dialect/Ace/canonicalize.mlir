// RUN: byteir-opt --canonicalize %s | FileCheck %s

func.func @test_ace_constant_case0() -> tensor<!ace.string> {
  %0 = "ace.constant"() {value = dense<"fork_active_pay"> : tensor<!ace.string>} : () -> tensor<!ace.string>
  return %0 : tensor<!ace.string>
}
// CHECK: ace.constant
