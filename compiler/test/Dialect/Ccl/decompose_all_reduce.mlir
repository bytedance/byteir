// RUN: byteir-opt %s --transform-dialect-interpreter --split-input-file | FileCheck %s


func.func @all_reduce(%arg0: tensor<10x32xf32>) -> tensor<10x32xf32> {
  %0 = "ccl.all_reduce"(%arg0)
    { reduction = "sum", replica_groups = [[0, 1, 2, 3, 4, 5, 6, 7]], unique_id = 0 : i64} : (tensor<10x32xf32>) -> tensor<10x32xf32>
  func.return %0 : tensor<10x32xf32>
}
// CHECK-LABEL: func.func @all_reduce
// CHECK-NEXT: ccl.reduce_scatter
// CHECK-NEXT: ccl.all_gather
// CHECK-NEXT: return

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["ccl.all_reduce"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %reduce_scatter, %all_gather = transform.decompose_all_reduce %0 {axis = 1}
}