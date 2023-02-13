// RUN: byteir-opt %s -test-byre-op-interface | FileCheck %s

module attributes {byre.container_module} {
  func.func @main(%arg0: memref<100x32xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x32xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.compute @some_kernel_0(%arg0, %arg1) {memory_effects = [1 : i32, 2 : i32]} : memref<100x32xf32>, memref<100x32xf32>
// CHECK-LABEL: some_kernel_0
// CHECK-NEXT: 1 Inputs:
// CHECK-NEXT:    <block argument> {{.*}} index: 0
// CHECK-NEXT: 1 Outputs:
// CHECK-NEXT:    <block argument> {{.*}} index: 1
    byre.compute @some_kernel_1(%arg0, %arg1) : memref<100x32xf32>, memref<100x32xf32>
// CHECK-LABEL: some_kernel_1
// CHECK-NEXT: 2 Inputs:
// CHECK-NEXT:    <block argument> {{.*}} index: 0
// CHECK-NEXT:    <block argument> {{.*}} index: 1
// CHECK-NEXT: 2 Outputs:
// CHECK-NEXT:    <block argument> {{.*}} index: 0
// CHECK-NEXT:    <block argument> {{.*}} index: 1
    byre.copy(%arg0, %arg1)  {callee = "some_copy"} : memref<100x32xf32>, memref<100x32xf32>
// CHECK-LABEL: some_copy
// CHECK-NEXT: 1 Inputs:
// CHECK-NEXT:    <block argument> {{.*}} index: 0
// CHECK-NEXT: 1 Outputs:
// CHECK-NEXT:    <block argument> {{.*}} index: 1
    %0 = "byre.alias"(%arg0) {offset = 0: i64} : (memref<100x32xf32>) -> memref<100x32xf32>
// CHECK-LABEL: AliasOp
// CHECK-NEXT: 1 Inputs:
// CHECK-NEXT:    <block argument> {{.*}} index: 0
// CHECK-NEXT: 1 Outputs:
// CHECK-NEXT:    "byre.alias"
    return
  }
}
