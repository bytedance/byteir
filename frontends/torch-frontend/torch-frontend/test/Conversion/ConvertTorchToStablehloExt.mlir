// RUN: torch-frontend-opt %s -convert-torch-to-stablehlo-ext --canonicalize-ext | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten._index_put_impl(
// CHECK-SAME:          %[[ARG0:.*]]: !torch.vtensor<[10,8],f32>, %[[ARG1:.*]]: !torch.vtensor<[2,5],si64>, %[[ARG2:.*]]: !torch.vtensor<[2,5,8],f32>) -> !torch.vtensor<[10,8],f32> {
// CHECK-DAG:         %[[V0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[10,8],f32> -> tensor<10x8xf32>
// CHECK-DAG:         %[[V1:.*]] = torch_c.to_builtin_tensor %[[ARG2]] : !torch.vtensor<[2,5,8],f32> -> tensor<2x5x8xf32>
// CHECK-DAG:         %[[V2:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[2,5],si64> -> tensor<2x5xi64>
// CHECK:         %[[V3:.*]] = stablehlo.reshape %[[V2]] : (tensor<2x5xi64>) -> tensor<10x1xi64>
// CHECK:         %[[V4:.*]] = stablehlo.reshape %[[V1]] : (tensor<2x5x8xf32>) -> tensor<10x8xf32>
// CHECK:         %[[V5:.*]] = "stablehlo.scatter"(%[[V0]], %[[V3]], %[[V4]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:         ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:           %[[V7:.*]] = stablehlo.add %[[ARG3]], %[[ARG4]] : tensor<f32>
// CHECK:           stablehlo.return %[[V7]] : tensor<f32>
// CHECK:         }) : (tensor<10x8xf32>, tensor<10x1xi64>, tensor<10x8xf32>) -> tensor<10x8xf32>
// CHECK:         %[[V6:.*]] = torch_c.from_builtin_tensor %[[V5]] : tensor<10x8xf32> -> !torch.vtensor<[10,8],f32>
// CHECK:         return %[[V6]] : !torch.vtensor<[10,8],f32>
func.func @torch.aten._index_put_impl(%arg0: !torch.vtensor<[10,8],f32>, %arg1: !torch.vtensor<[2,5],si64>, %arg2: !torch.vtensor<[2,5,8],f32>) -> !torch.vtensor<[10,8],f32> {
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %0 = torch.prim.ListConstruct %arg1 : (!torch.vtensor<[2,5],si64>) -> !torch.list<vtensor>
  %1 = torch.aten._index_put_impl %arg0, %0, %arg2, %true, %false : !torch.vtensor<[10,8],f32>, !torch.list<vtensor>, !torch.vtensor<[2,5,8],f32>, !torch.bool, !torch.bool -> !torch.vtensor<[10,8],f32>
  return %1 : !torch.vtensor<[10,8],f32>
}

// CHECK-LABEL:   func.func @torch.aten.pow.Scalar
// CHECK:  chlo.broadcast_power
// CHECK-SAME:  (tensor<f32>, tensor<3x4xf32>) -> tensor<3x4xf32>
func.func @torch.aten.pow.Scalar(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %0 = torch.aten.pow.Scalar %float2.000000e00, %arg0 : !torch.float, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}
