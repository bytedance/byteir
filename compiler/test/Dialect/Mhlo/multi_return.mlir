// RUN: byteir-opt %s | FileCheck %s


func.func @mhlo_tuple_return(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %res = "mhlo.tuple"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>>
  return %res : tuple<tensor<4xf32>, tensor<4xf32>>
}
// CHECK-LABEL: func.func @mhlo_tuple_return

func.func @mhlo_mutiple_return_case_1(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_mutiple_return_case_1

func.func @mhlo_mutiple_return_case_2(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.tuple"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>>
  %3 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  %4 = "mhlo.get_tuple_element"(%2) {index = 1 : i32} : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  return %3, %4 : tensor<4xf32>, tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_mutiple_return_case_2




