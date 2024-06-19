// RUN: byteir-opt %s --linalg-tensor-opt --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func  @DynamicParallelReduction(%arg0: tensor<?x?xf32>) -> tensor<?xf32> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %0 = tensor.empty(%dim0) : tensor<?xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    return %2 : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @DynamicParallelReduction
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x?xf32>
// CHECK-DAG: %[[EMPTY:.*]] = tensor.empty(%[[DIM0]]) : tensor<?xf32>
// CHECK-DAG: %[[DIM1:.*]] = tensor.dim %arg0, %[[C1]] : tensor<?x?xf32>
// CHECK: %[[RESULT:.*]] = scf.forall (%[[BLOCK_ID:.*]]) in (%[[DIM0]]) shared_outs(%[[OUTS:.*]] = %[[EMPTY]])
	// CHECK-NEXT: %[[EXTRACT_SLICE0:.*]] = tensor.extract_slice %[[OUTS]][%[[BLOCK_ID]]] [1] [1] : tensor<?xf32> to tensor<f32>
	// CHECK-NEXT: %[[THREAD_LOC_SUM:.*]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>}
	// CHECK-NEXT: scf.forall (%[[THREAD_ID:.*]]) in ({{.*}}) shared_outs(%{{.*}} = %[[THREAD_LOC_SUM]])
		// CHECK-NOT: linalg.fill
		// CHECK-NEXT: scf.for %{{.*}} = %[[THREAD_ID]] to %[[DIM1:.*]] step %{{.*}} iter_args(%{{.*}} = %[[CST0]])
			// CHECK-NOT: linalg.fill
	// CHECK: } {mapping = [#gpu.thread<x>]}

	// CHECK: %[[WARP_SUM:.*]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} 
	// CHECK-NEXT: scf.forall (%[[WARP_ID:.*]]) in ({{.*}}) shared_outs(%{{.*}} = %[[WARP_SUM]])
		//CHECK-NOT: linalg.fill
	// CHECK: } {mapping = [#gpu.warp<linear_dim_0>]}

	// CHECK: scf.forall (%{{.*}}) in ({{.*}}) shared_outs(%{{.*}} = %[[EXTRACT_SLICE0]]) -> (tensor<f32>)
		// CHECK-NOT: linalg.fill
	// CHECK: } {mapping = [#gpu.warp<linear_dim_0>]}

// CHECK: } {mapping = [#gpu.block<linear_dim_0>]}
// CHECK: return %[[RESULT]] : tensor<?xf32>

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @ReduceParallel(%arg0: tensor<?x?xf32>) -> tensor<?xf32> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim1) : tensor<?xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    return %2 : tensor<?xf32>
  }
}

// CHECK-DAG: #[[$MAP_BLOCK_NUM:.*]] = affine_map<()[s0] -> (s0 ceildiv {{.*}})>
// CHECK-DAG: #[[$MAP_TILE_SIZE:.*]] = affine_map<(d0)[s0] -> (d0 * {{.*}} + s0, {{.*}})>
// CHECK-DAG: #[[$MAP_BLOCK_OFFSET:.*]] = affine_map<(d0) -> (d0 * {{.*}})>

// CHECK-LABEL: func.func @ReduceParallel
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM1:.*]] = tensor.dim %arg0, %[[C1]] : tensor<?x?xf32>
// CHECK-DAG: %[[EMPTY:.*]] = tensor.empty(%[[DIM1]]) : tensor<?xf32>
// CHECK-DAG: %[[BLOCK_NUM:.*]] = affine.apply #[[$MAP_BLOCK_NUM]]()[%[[DIM1]]]
// CHECK-DAG:  %[[DIM0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x?xf32>
// CHECK: %[[RESULT:.*]] = scf.forall (%[[BLOCK_ID:.*]]) in (%[[BLOCK_NUM]]) shared_outs(%[[OUTS:.*]] = %[[EMPTY]])
	// CHECK-NEXT: %[[TILE_SIZE:.*]] = affine.min #[[$MAP_TILE_SIZE]](%[[BLOCK_ID]])[%[[DIM1]]]
	// CHECK-NEXT: %[[BLOCK_OFFSET:.*]] = affine.apply #[[$MAP_BLOCK_OFFSET]](%[[BLOCK_ID]])
	// CHECK-NEXT: %[[BLOCK_TILE:.*]] = tensor.extract_slice %[[OUTS]][%[[BLOCK_OFFSET]]] [%[[TILE_SIZE]]] [1] : tensor<?xf32> to tensor<?xf32>
	// CHECK-NEXT: scf.forall (%[[THREAD_ID:.*]]) in (%[[TILE_SIZE]]) shared_outs(%[[BLOCK_OUTS:.*]] = %[[BLOCK_TILE]])
		// CHECK-NEXT: scf.for %[[IV:.*]] = %[[C0]] to %[[DIM0]] step %[[C1]] iter_args(%[[LOCAL_SUM:.*]] = %[[CST]]) -> (f32) {
			// CHECK-NOT: linalg.fill
	// CHECK: } {mapping = [#gpu.thread<x>]}
// CHECK: mapping = [#gpu.block<linear_dim_0>]}
// CHECK: return %[[RESULT:.*]] : tensor<?xf32>

// -----

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @SplitReduce(%arg0: tensor<9000xf32>) -> tensor<f32> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<9000xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<f32>
    return %2 : tensor<f32>
  }
}

// CHECK-LABEL: func.func @SplitReduce
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<f32>
// CHECK-DAG: %[[SPLIT_EMPTY:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[TMP_RESULT:.*]] = scf.forall (%arg1) in ({{.*}}) shared_outs(%[[TMP_OUTS:.*]] = %[[SPLIT_EMPTY]])
	// CHECK-NOT: linalg.fill

// CHECK: %[[RESULT:.*]] = scf.forall (%arg1) in (1) shared_outs(%[[OUTS:.*]] = %[[EMPTY]])
	// CHECK-NOT: linalg.fill
// CHECK: return %[[RESULT:.*]] : tensor<f32>