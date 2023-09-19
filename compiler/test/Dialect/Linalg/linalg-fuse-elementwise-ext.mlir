// RUN: byteir-opt %s -linalg-fuse-elementwise-ext="shared-input" -cse -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @one_output
func.func @one_output(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: linalg.yield [[T2]]
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @two_outputs
func.func @two_outputs(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: linalg.yield [[T1]], [[T2]]   
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %3, %4 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @two_use_two_outputs
func.func @two_use_two_outputs(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %4 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %2, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK: linalg.generic {
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):  
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: [[T3:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[T2]]
      // CHECK: linalg.yield [[T1]], [[T3]]    
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %1#0, %6 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @one_use_0_two_outputs
func.func @one_use_0_two_outputs(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %4 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %2, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK: linalg.generic {
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%1#0, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):  
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: linalg.yield [[T1]], [[T2]]    
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %1#0, %6 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @one_use_1_two_outputs
func.func @one_use_1_two_outputs(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %4 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %2, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK: linalg.generic {
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%1#1, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG3:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):  
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: [[T3:%[a-zA-Z0-9_]*]] = arith.mulf [[T2]], [[ARG3]]
      // CHECK: linalg.yield [[T1]], [[T3]]    
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %1#0, %6 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @two_consumers
func.func @two_consumers(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):       
      %4 = arith.addf %arg4, %arg5 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG3:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: [[T3:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG3]]
      // CHECK: linalg.yield [[T2]], [[T3]]   
      %6 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %6 : f32
    } -> tensor<?x?xf32>
  return %4, %5 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @shared_input
func.func @shared_input(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      %5 = arith.addf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
// CHECK: linalg.generic {
  %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
    // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
    // CHECK-NOT: linalg.yield
    // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[ARG0]], [[ARG2]]
    // CHECK: linalg.yield [[T1]], [[T2]]  
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4, %5 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @generic_with_map
func.func @generic_with_map(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
// CHECK: linalg.generic {
  %4 = linalg.map
      ins(%3, %arg2: tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2:tensor<?x?xf32>)
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
      (%lhs_elem: f32, %rhs_elem: f32) {
    // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
    // CHECK-NOT: linalg.yield
    // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
    // CHECK: linalg.yield [[T2]]  
        %6 = arith.mulf %lhs_elem, %rhs_elem: f32
        linalg.yield %6: f32
      }
  return %4 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @single_map
func.func @single_map(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
// CHECK-NOT: linalg.map
// CHECK: linalg.generic
  %4 = linalg.map
      ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2:tensor<?x?xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
  
        %6 = arith.mulf %lhs_elem, %rhs_elem: f32
        linalg.yield %6: f32
      }
  return %4 : tensor<?x?xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#trait = {
  indexing_maps = [#map, #map],
  iterator_types = ["parallel", "parallel"]
}
func.func @no_more_break_outs_dependency(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.generic #trait ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
       ^bb0(%arg1 : f32, %arg2 : f32) :
         %1 = arith.addf %arg1, %arg1 : f32
         linalg.yield %1 : f32
       } -> tensor<?x?xf32>
  %2 = linalg.generic #trait ins(%0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
       ^bb0(%arg1 : f32, %arg2 : f32) :
         %3 = arith.mulf %arg1, %arg1 : f32
         linalg.yield %3 : f32
       } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func @no_more_break_outs_dependency(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]])
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     outs(%[[INIT]] : tensor<?x?xf32>)
// CHECK: return %[[RESULT]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>

// CHECK-LABEL: @broadcast_output
func.func @broadcast_output(%arg0: tensor<96x1024x1024xf16>, %arg1: tensor<1x1x1024x1024xf32>) -> (tensor<1x1x1024x1024xi1>, tensor<8x12x1024x1024xf16>) attributes {__byteir_elementwise_fusion__} {
  %cst = arith.constant dense<1.250000e-01> : tensor<96x1024x1024xf16>
  %cst_0 = arith.constant dense<0xFC00> : tensor<8x12x1024x1024xf16>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1x1024x1024xf32>
  %0 = tensor.empty() : tensor<96x1024x1024xf16>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst : tensor<96x1024x1024xf16>, tensor<96x1024x1024xf16>) outs(%0 : tensor<96x1024x1024xf16>) {
  ^bb0(%in: f16, %in_2: f16, %out: f16):
    %8 = arith.mulf %in, %in_2 : f16
    linalg.yield %8 : f16
  } -> tensor<96x1024x1024xf16>
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] : tensor<96x1024x1024xf16> into tensor<8x12x1024x1024xf16>
  %2 = tensor.empty() : tensor<1x1x1024x1024xi1>
// CHECK: linalg.generic {
// CHECK: arith.cmpf
// CHECK: linalg.yield
  %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %cst_1 : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32>) outs(%2 : tensor<1x1x1024x1024xi1>) {
  ^bb0(%in: f32, %in_2: f32, %out: i1):
    %8 = arith.cmpf oeq, %in, %in_2 : f32
    linalg.yield %8 : i1
  } -> tensor<1x1x1024x1024xi1>
  %4 = tensor.empty() : tensor<8x12x1024x1024xi1>
// CHECK: linalg.generic {
// CHECK: arith.mulf
// CHECK: arith.select
// CHECK: linalg.yield
  %5 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x1x1024x1024xi1>) outs(%4 : tensor<8x12x1024x1024xi1>) {
  ^bb0(%in: i1, %out: i1):
    linalg.yield %in : i1
  } -> tensor<8x12x1024x1024xi1>
  %6 = tensor.empty() : tensor<8x12x1024x1024xf16>
  %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst_0, %expanded : tensor<8x12x1024x1024xi1>, tensor<8x12x1024x1024xf16>, tensor<8x12x1024x1024xf16>) outs(%6 : tensor<8x12x1024x1024xf16>) {
  ^bb0(%in: i1, %in_2: f16, %in_3: f16, %out: f16):
    %8 = arith.select %in, %in_2, %in_3 : f16
    linalg.yield %8 : f16
  } -> tensor<8x12x1024x1024xf16>
  return %3, %7 : tensor<1x1x1024x1024xi1>, tensor<8x12x1024x1024xf16>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @check_collapse_expand
func.func @check_collapse_expand(%arg0: tensor<3x4xf32>, %arg1 : tensor<3x4xf32>, %arg2 : tensor<12xf32>) -> (tensor<3x4xf32>, tensor<12xf32>, tensor<4x3xf32>, tensor<12xf32>)
{
  %2 = tensor.empty() : tensor<3x4xf32>
  %1 = tensor.empty() : tensor<3x4xf32>
  %0 = tensor.empty() : tensor<3x4xf32>
  %3:3 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<3x4xf32>, tensor<3x4xf32>)
      outs(%0, %1, %2 : tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      %5 = arith.mulf %arg3, %arg4 : f32
      %6 = arith.subf %arg3, %arg4 : f32
      linalg.yield %4, %5, %6  : f32, f32, f32
  } -> (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>)
  %collapsed = tensor.collapse_shape %3#1 [[0, 1]] : tensor<3x4xf32> into tensor<12xf32>
  %11 = tensor.empty() : tensor<12xf32>
  %collapsed_1 = tensor.collapse_shape %3#1 [[0, 1]] : tensor<3x4xf32> into tensor<12xf32>
  %expanded = tensor.expand_shape %collapsed_1 [[0, 1]] : tensor<12xf32> into tensor<4x3xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]}
// CHECK: linalg.generic {
// CHECK: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: linalg.yield
      ins(%collapsed  : tensor<12xf32>)
      outs(%11 : tensor<12xf32>) {
    ^bb0(%arg5: f32, %arg6: f32):       
      %5 = arith.mulf %arg5, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<12xf32>
  return %3#0, %collapsed, %expanded, %4 : tensor<3x4xf32>, tensor<12xf32>, tensor<4x3xf32>, tensor<12xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @consumer_multi_result_from_producer(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  %0 = tensor.empty() : tensor<2x4xf32>
  %1 = tensor.empty() : tensor<2x4xf32>
  %2 = tensor.empty() : tensor<2x4xf32>
  %3 = tensor.empty() : tensor<2x4xf32>
  %4:3 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%0, %1, %3 : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32, %out_1: f32, %out_2: f32):
    %6 = arith.maxf %in, %in_0 : f32
    %7 = arith.minf %in, %in_0 : f32
    %8 = arith.addf %7, %in_0 : f32
    linalg.yield %7, %8, %6 : f32, f32, f32
  } -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>)
  %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4#1, %4#0 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%2 : tensor<2x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.mulf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<2x4xf32>
  return %4#1, %4#2, %5 : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}
// CHECK-LABEL: @consumer_multi_result_from_producer
// CHECK: %[[RES:.*]]:3 = linalg.generic
// CHECK-DAG: %[[V0:.*]] = arith.maxf
// CHECK-DAG: %[[V1:.*]] = arith.minf
// CHECK-DAG: %[[V2:.*]] = arith.addf
// CHECK-DAG: %[[V3:.*]] = arith.mulf
// CHECK:     linalg.yield %[[V2]], %[[V0]], %[[V3]]
// CHECK-NOT: linalg.generic
// CHECK:     return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @consumer_multi_result_from_producer_with_different_indexing_map(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  %0 = tensor.empty() : tensor<2x4xf32>
  %1 = tensor.empty() : tensor<2x4xf32>
  %2 = tensor.empty() : tensor<2x4xf32>
  %3 = tensor.empty() : tensor<2x4xf32>
  %4:3 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%0, %1, %3 : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32, %out_1: f32, %out_2: f32):
    %6 = arith.maxf %in, %in_0 : f32
    %7 = arith.minf %in, %in_0 : f32
    %8 = arith.addf %7, %in_0 : f32
    linalg.yield %7, %8, %6 : f32, f32, f32
  } -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>)
  %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4#1, %4#0 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%2 : tensor<2x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.mulf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<2x4xf32>
  return %4#1, %4#2, %5 : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}
// CHECK-LABEL: @consumer_multi_result_from_producer_with_different_indexing_map
// CHECK: linalg.generic
// CHECK: linalg.generic

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @constant_in_affine_map_with_collapse_shape(%arg0: tensor<1x256x1024xf32>, %arg1: tensor<256x1024xf16>, %arg2: tensor<256x1xf32>, %arg3: tensor<256x1xf32>) -> tensor<256x1024xf32> {
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x1024xf16> into tensor<1x256x1024xf16>
  %0 = tensor.empty() : tensor<1x256x1024xf32>
  %1 = tensor.empty() : tensor<256x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0 : tensor<1x256x1024xf16>, tensor<1x256x1024xf32>) outs(%0 : tensor<1x256x1024xf32>) {
  ^bb0(%in: f16, %in_0: f32, %out: f32):
    %4 = arith.extf %in : f16 to f32
    %5 = arith.addf %in_0, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x256x1024xf32>
  %collapsed = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<1x256x1024xf32> into tensor<256x1024xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2, %collapsed : tensor<256x1xf32>, tensor<256x1xf32>, tensor<256x1024xf32>) outs(%1 : tensor<256x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %4 = arith.subf %in_1, %in_0 : f32
    %5 = arith.mulf %4, %in : f32
    linalg.yield %5 : f32
  } -> tensor<256x1024xf32>
  return %3 : tensor<256x1024xf32>
}

// CHECK-LABEL: @constant_in_affine_map_with_collapse_shape
// CHECK: linalg.generic
// CHECK-NOT: linalg.generic
