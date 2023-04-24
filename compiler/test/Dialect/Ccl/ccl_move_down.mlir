// RUN: byteir-opt %s -ccl-move-down -split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 3)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @decomposed_all_reduce(%arg0: tensor<125x32x15xf32>) -> tensor<125x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c5 = arith.constant 5 : index
  %0 = tensor.empty() : tensor<125x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<125x32xf32>) -> tensor<125x32xf32>
  %2 = tensor.empty() : tensor<125x32xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<125x32xf32>) -> tensor<125x32xf32>
  %4 = scf.forall (%arg1) in (%c5) shared_outs(%arg2 = %3) -> (tensor<125x32xf32>) {
    %5 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %5] [125, 32, 3] [1, 1, 1] : tensor<125x32x15xf32> to tensor<125x32x3xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<125x32x3xf32>) outs(%arg2 : tensor<125x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.mulf %in, %in : f32
      %11 = arith.addf %10, %out : f32
      linalg.yield %11 : f32
    } -> tensor<125x32xf32>
    %7 = "ccl.reduce_scatter"(%6) {axis = 0 : i64, reduction = "sum", replica_groups = [[0, 1, 2, 3, 4]]} : (tensor<125x32xf32>) -> tensor<25x32xf32>
    %8 = "ccl.all_gather"(%7) {axis = 0 : i64, replica_groups = [[0, 1, 2, 3, 4]]} : (tensor<25x32xf32>) -> tensor<125x32xf32>
    %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<125x32xf32>) outs(%1 : tensor<125x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.addf %in, %out : f32
      linalg.yield %10 : f32
    } -> tensor<125x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg2[0, 0] [125, 32] [1, 1] : tensor<125x32xf32> into tensor<125x32xf32>
    }
  }
  return %4 : tensor<125x32xf32>
}
// CHECK-LABEL: func.func @decomposed_all_reduce
// CHECK: ccl.reduce_scatter
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
// CHECK: ccl.all_gather
// CHECK-NOT: linalg.generic
// CHECK-NEXT: scf.forall.in_parallel

// -----

#map = affine_map<(d0) -> (d0 * 3)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @decomposed_all_reduce_with_tensor_empty_init_op(%arg0: tensor<125x32x15xf32>) -> tensor<125x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c5 = arith.constant 5 : index
  %0 = tensor.empty() : tensor<125x32xf32>
  %2 = tensor.empty() : tensor<125x32xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<125x32xf32>) -> tensor<125x32xf32>
  %4 = scf.forall (%arg1) in (%c5) shared_outs(%arg2 = %3) -> (tensor<125x32xf32>) {
    %5 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %5] [125, 32, 3] [1, 1, 1] : tensor<125x32x15xf32> to tensor<125x32x3xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<125x32x3xf32>) outs(%arg2 : tensor<125x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.mulf %in, %in : f32
      %11 = arith.addf %10, %out : f32
      linalg.yield %11 : f32
    } -> tensor<125x32xf32>
    %7 = "ccl.reduce_scatter"(%6) {axis = 0 : i64, reduction = "sum", replica_groups = [[0, 1, 2, 3, 4]]} : (tensor<125x32xf32>) -> tensor<25x32xf32>
    %8 = "ccl.all_gather"(%7) {axis = 0 : i64, replica_groups = [[0, 1, 2, 3, 4]]} : (tensor<25x32xf32>) -> tensor<125x32xf32>
    %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<125x32xf32>) outs(%0 : tensor<125x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.addf %in, %out : f32
      linalg.yield %10 : f32
    } -> tensor<125x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg2[0, 0] [125, 32] [1, 1] : tensor<125x32xf32> into tensor<125x32xf32>
    }
  }
  return %4 : tensor<125x32xf32>
}
// CHECK-LABEL: func.func @decomposed_all_reduce_with_tensor_empty_init_op
// CHECK: ccl.reduce_scatter
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
// CHECK: ccl.all_gather
// CHECK-NOT: linalg.generic
// CHECK-NEXT: scf.forall.in_parallel

// -----

#map = affine_map<(d0) -> (d0 * 3)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @user_outside_region_of_all_gather(%arg0: tensor<125x32x15xf32>) -> tensor<125x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c5 = arith.constant 5 : index
  %0 = tensor.empty() : tensor<125x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<125x32xf32>) -> tensor<125x32xf32>
  %2 = tensor.empty() : tensor<125x32xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<125x32xf32>) -> tensor<125x32xf32>
  %4 = scf.forall (%arg1) in (%c5) shared_outs(%arg2 = %3) -> (tensor<125x32xf32>) {
    %5 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %5] [125, 32, 3] [1, 1, 1] : tensor<125x32x15xf32> to tensor<125x32x3xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<125x32x3xf32>) outs(%arg2 : tensor<125x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.mulf %in, %in : f32
      %11 = arith.addf %10, %out : f32
      linalg.yield %11 : f32
    } -> tensor<125x32xf32>
    %7 = "ccl.reduce_scatter"(%6) {axis = 0 : i64, reduction = "sum", replica_groups = [[0, 1, 2, 3, 4]]} : (tensor<125x32xf32>) -> tensor<25x32xf32>
    %8 = "ccl.all_gather"(%7) {axis = 0 : i64, replica_groups = [[0, 1, 2, 3, 4]]} : (tensor<25x32xf32>) -> tensor<125x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg2[0, 0] [125, 32] [1, 1] : tensor<125x32xf32> into tensor<125x32xf32>
    }
  }
  %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<125x32xf32>) outs(%1 : tensor<125x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.addf %in, %out : f32
      linalg.yield %10 : f32
    } -> tensor<125x32xf32>

  return %9 : tensor<125x32xf32>
}
// CHECK-LABEL: func.func @user_outside_region_of_all_gather
// CHECK: ccl.reduce_scatter
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
// CHECK: ccl.all_gather
// CHECK-NOT: linalg.generic
// CHECK-NEXT: scf.forall.in_parallel
