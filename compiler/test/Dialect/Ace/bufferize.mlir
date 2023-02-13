// RUN: byteir-opt -ace-bufferize -canonicalize -split-input-file %s | FileCheck %s

func.func @constant() -> tensor<!ace.string> {
  %0 = "ace.constant"() {value = dense<"foo"> : tensor<!ace.string>} : () -> tensor<!ace.string>
  return %0 : tensor<!ace.string>
}

// CHECK-LABEL: func.func @constant
// CHECK-NEXT: %[[BUFFER:.*]] = memref.alloc
// CHECK-NEXT: "lace.constant"(%[[BUFFER]])
// CHECK-NEXT: %[[Tensor:.*]] = bufferization.to_tensor %[[BUFFER]]
// CHECK-NEXT: return %[[Tensor]]

// -----

func.func @custom_call(%arg0: tensor<!ace.string>) -> tensor<!ace.string> {
  %0 = "ace.custom_call"(%arg0) {byteir_attrs = {some_attr}, call_target_name = "some_op"} : (tensor<!ace.string>) -> tensor<!ace.string>
  return %0 : tensor<!ace.string>
}
// CHECK-LABEL: func.func @custom_call
// CHECK-NEXT: %[[ARG_BUFFER:.*]] = bufferization.to_memref %arg0
// CHECK-NEXT: %[[BUFFER:.*]] = memref.alloc
// CHECK-NEXT: "lace.custom_call"(%[[ARG_BUFFER]], %[[BUFFER]])
//   CHECK-DAG: byteir_attrs = {some_attr}
//   CHECK-DAG: call_target_name = "some_op"
// CHECK-NEXT: %[[Tensor:.*]] = bufferization.to_tensor %[[BUFFER]]
// CHECK-NEXT: return %[[Tensor]]
