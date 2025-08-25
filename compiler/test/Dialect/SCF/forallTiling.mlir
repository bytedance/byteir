// RUN: byteir-opt %s --forall-tiling="tile-sizes=256" --split-input-file --canonicalize --cse | FileCheck %s

func.func @Copy(%arg0: memref<32x64xf32>, %arg1: memref<32x64xf32>) attributes {__byteir_reduction_fusion__} {
	%c64 = arith.constant 64 : index
	scf.forall (%arg2) in (2048) {
		%0 = arith.remsi %arg2, %c64 : index
		%1 = arith.divsi %arg2, %c64 : index
		%2 = memref.load %arg0[%1, %0] : memref<32x64xf32>
		memref.store %2, %arg1[%1, %0] : memref<32x64xf32>
	}
	return
}

// CHECK-LABEL: func.func @Copy
// CHECK-NEXT: %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT: scf.forall (%[[ARG2:.*]]) = (0) to (2048) step (256) {
	// CHECK-NEXT: scf.forall (%[[ARG3:.*]]) in (256) {
		// CHECK-NEXT: %[[V0:.*]] = arith.addi %[[ARG3]], %[[ARG2]] : index
    // CHECK-NEXT: %[[V1:.*]] = arith.remsi %[[V0]], %[[C64]] : index
    // CHECK-NEXT: %[[V2:.*]] = arith.divsi %[[V0]], %[[C64]] : index
    // CHECK-NEXT: %[[V3:.*]] = memref.load %arg0[%[[V2]], %[[V1]]] : memref<32x64xf32>
    // CHECK-NEXT: memref.store %[[V3]], %arg1[%[[V2]], %[[V1]]] : memref<32x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @Elementwise(%arg0: memref<32x1024x?x30xf32>) -> memref<32768x?x30xf32> attributes {__byteir_elementwise_fusion__} {
	%c983040 = arith.constant 983040 : index
	%c30 = arith.constant 30 : index
	%c2 = arith.constant 2 : index
	%collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2], [3]] : memref<32x1024x?x30xf32> into memref<32768x?x30xf32>
	%dim = memref.dim %arg0, %c2 : memref<32x1024x?x30xf32>
	%alloc = memref.alloc(%dim) : memref<32768x?x30xf32>
	%0 = arith.muli %dim, %c983040 : index
	scf.forall (%arg1) in (%0) {
		%1 = arith.remsi %arg1, %c30 : index
		%2 = arith.divsi %arg1, %c30 : index
		%3 = arith.remsi %2, %dim : index
		%4 = arith.divsi %2, %dim : index
		%subview = memref.subview %collapse_shape[%4, %3, %1] [1, 1, 1] [1, 1, 1] : memref<32768x?x30xf32> to memref<1x1x1xf32, strided<[300, 30, 1], offset: ?>>
		%subview_0 = memref.subview %alloc[%4, %3, %1] [1, 1, 1] [1, 1, 1] : memref<32768x?x30xf32> to memref<1x1x1xf32, strided<[300, 30, 1], offset: ?>>
		linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<1x1x1xf32, strided<[300, 30, 1], offset: ?>>) outs(%subview_0 : memref<1x1x1xf32, strided<[300, 30, 1], offset: ?>>) attrs =  {__byteir_gpu_tile_elementwise_0} {
		^bb0(%in: f32, %out: f32):
			%5 = arith.mulf %in, %in : f32
			linalg.yield %5 : f32
		}
	}
	return %alloc : memref<32768x?x30xf32>
}

// CHECK: #[[$MAP_LOOP_SIZE:.*]]  = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
// CHECK-LABEL: func.func @Elementwise
// CHECK-DAG: %[[C983040:.*]] = arith.constant 983040 : index
// CHECK-DAG: %[[C30:.*]] = arith.constant 30 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[COLLAPSE:.*]] = memref.collapse_shape %arg0
// CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[C2]] : memref<32x1024x?x30xf32>
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<32768x?x30xf32>
// CHECK-NEXT: %[[LB:.*]] = arith.muli %dim, %[[C983040]] : index
// CHECK-NEXT: scf.forall (%[[ARG1:.*]]) = (0) to (%[[LB]]) step (256) {
	// CHECK-NEXT: %[[V1:.*]] = affine.min #[[$MAP_LOOP_SIZE]](%[[ARG1]])[%[[LB]]]
	// CHECK-NEXT: scf.forall (%[[ARG2:.*]]) in (%[[V1:.*]]) {
	// CHECK-NEXT: %[[V2:.*]] = arith.addi %[[ARG2]], %[[ARG1]] : index
  // CHECK-NEXT: %[[V3:.*]] = arith.remsi %[[V2]], %[[C30]] : index
  // CHECK-NEXT: %[[V4:.*]] = arith.divsi %[[V2]], %[[C30]] : index
  // CHECK-NEXT: %[[V5:.*]] = arith.remsi %[[V4]], %[[DIM]] : index
  // CHECK-NEXT: %[[V6:.*]] = arith.divsi %[[V4]], %[[DIM]] : index
	// CHECK-NEXT: %[[SUBVIEW:.*]] = memref.subview %[[COLLAPSE]][%[[V6]], %[[V5]], %[[V3]]] [1, 1, 1] [1, 1, 1] : memref<32768x?x30xf32> to memref<1x1x1xf32, strided<[300, 30, 1], offset: ?>>
  // CHECK-NEXT: %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC]][%[[V6]], %[[V5]], %[[V3]]] [1, 1, 1] [1, 1, 1] : memref<32768x?x30xf32> to memref<1x1x1xf32, strided<[300, 30, 1], offset: ?>>
