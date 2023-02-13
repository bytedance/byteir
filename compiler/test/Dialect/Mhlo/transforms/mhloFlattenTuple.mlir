// RUN: byteir-opt -mhlo-flatten-tuple %s | FileCheck %s
// Note mhlo-flatten-tuple statement only. It won't apply to arg and return.

func.func @mhlo_tuple_statement(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.tuple"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>>
  %3 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  %4 = "mhlo.get_tuple_element"(%2) {index = 1 : i32} : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  return %3, %4 : tensor<4xf32>, tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_tuple_statement
// CHECK: %[[VAR_0:.*]] = mhlo.add %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK: %[[VAR_1:.*]] = mhlo.add %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK: return %[[VAR_0]], %[[VAR_1]] : tensor<4xf32>, tensor<4xf32>

func.func @mhlo_tuple_return(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %res = "mhlo.tuple"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>>
  return %res : tuple<tensor<4xf32>, tensor<4xf32>>
}
// CHECK-LABEL: func.func @mhlo_tuple_return
// CHECK: %[[VAR_0:.*]] = mhlo.add %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK: %[[VAR_1:.*]] = mhlo.add %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK: %[[VAR_2:.*]] = mhlo.tuple %[[VAR_0]], %[[VAR_1]] : tuple<tensor<4xf32>, tensor<4xf32>>
// CHECK: return %[[VAR_2]] : tuple<tensor<4xf32>, tensor<4xf32>>
