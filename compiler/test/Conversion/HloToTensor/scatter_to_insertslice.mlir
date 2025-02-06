// RUN: byteir-opt -convert-hlo-to-tensor --canonicalize --split-input-file %s | FileCheck %s

func.func @one_indices(%arg0: tensor<6x8x5xf32>, %arg1: tensor<6x1x5xf32>) -> tensor<6x8x5xf32> {
    %0 = mhlo.constant dense<0> : tensor<1x1xi64>
    %1 = "mhlo.scatter"(%arg0, %0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        mhlo.return %arg3 : tensor<f32>
    }) : (tensor<6x8x5xf32>, tensor<1x1xi64>, tensor<6x1x5xf32>) -> tensor<6x8x5xf32>
    return %1 : tensor<6x8x5xf32>
}

// CHECK: func.func
// CHECK-NEXT: %inserted_slice = tensor.insert_slice %arg1 into %arg0[0, 0, 0] [6, 1, 5] [1, 1, 1] : tensor<6x1x5xf32> into tensor<6x8x5xf32>
// CHECK-NEXT: return %inserted_slice : tensor<6x8x5xf32>

func.func @static_stride(%arg0: tensor<6x8x5xf32>, %arg1: tensor<6x3x5xf32>) -> tensor<6x8x5xf32> {
    %0 = mhlo.constant dense<[[0], [1], [2]]> : tensor<3x1xi64>
    %1 = "mhlo.scatter"(%arg0, %0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        mhlo.return %arg3 : tensor<f32>
    }) : (tensor<6x8x5xf32>, tensor<3x1xi64>, tensor<6x3x5xf32>) -> tensor<6x8x5xf32>
    return %1 : tensor<6x8x5xf32>
}

// CHECK: func.func
// CHECK-NEXT: %inserted_slice = tensor.insert_slice %arg1 into %arg0[0, 0, 0] [6, 3, 5] [1, 1, 1] : tensor<6x3x5xf32> into tensor<6x8x5xf32>
// CHECK-NEXT: return %inserted_slice : tensor<6x8x5xf32>

func.func @dynamic_stride(%arg0: tensor<6x8x5xf32>, %arg1: tensor<6x3x5xf32>) -> tensor<6x8x5xf32> {
    %0 = mhlo.constant dense<[[0], [1], [3]]> : tensor<3x1xi64>
    %1 = "mhlo.scatter"(%arg0, %0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        mhlo.return %arg3 : tensor<f32>
    }) : (tensor<6x8x5xf32>, tensor<3x1xi64>, tensor<6x3x5xf32>) -> tensor<6x8x5xf32>
    return %1 : tensor<6x8x5xf32>
}

// CHECK: func.func
// CHECK-NOT: tensor.insert_slice