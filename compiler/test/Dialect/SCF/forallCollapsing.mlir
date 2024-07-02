// RUN: byteir-opt %s --forall-collapsing --split-input-file --canonicalize --cse | FileCheck %s

func.func @Copy(%arg0: memref<32x64xf32>, %arg1: memref<32x64xf32>) attributes {__byteir_reduction_fusion__} {
	scf.forall (%arg2, %arg3) in (32, 64) {
		%0 = memref.load %arg0[%arg2, %arg3] : memref<32x64xf32>
		memref.store %0, %arg1[%arg2, %arg3] : memref<32x64xf32>
	}
	return
}

// CHECK-LABEL: func.func @Copy
// CHECK-NEXT: %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT: scf.forall (%[[ARG2:.*]]) in (2048) {
// CHECK-NEXT: %[[V0:.*]] = arith.remsi %[[ARG2]], %[[C64]] : index
// CHECK-NEXT: %[[V1:.*]] = arith.divsi %[[ARG2]], %[[C64]] : index
// CHECK-NEXT: %[[V2:.*]] = memref.load %arg0[%[[V1]], %[[V0]]] : memref<32x64xf32>
// CHECK-NEXT: memref.store %[[V2]], %arg1[%[[V1]], %[[V0]]] : memref<32x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @Elementwise(%arg0: memref<32x1024x?x30xf32>) -> memref<32768x?x30xf32> attributes {__byteir_elementwise_fusion__} {
	%collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2], [3]] : memref<32x1024x?x30xf32> into memref<32768x?x30xf32>
	%c2 = arith.constant 2 : index
	%dim = memref.dim %arg0, %c2 : memref<32x1024x?x30xf32>
	%alloc = memref.alloc(%dim) : memref<32768x?x30xf32>
	scf.forall (%arg1, %arg2, %arg3) in (32768, %dim, 30) {
		%subview = memref.subview %collapse_shape[%arg1, %arg2, %arg3] [1, 1, 1] [1, 1, 1] : memref<32768x?x30xf32> to memref<1x1x1xf32, strided<[?, 30, 1], offset: ?>>
		%subview_0 = memref.subview %alloc[%arg1, %arg2, %arg3] [1, 1, 1] [1, 1, 1] : memref<32768x?x30xf32> to memref<1x1x1xf32, strided<[?, 30, 1], offset: ?>>
		linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<1x1x1xf32, strided<[?, 30, 1], offset: ?>>) outs(%subview_0 : memref<1x1x1xf32, strided<[?, 30, 1], offset: ?>>) attrs =  {__byteir_gpu_tile_elementwise_0} {
		^bb0(%in: f32, %out: f32):
			%0 = arith.mulf %in, %in : f32
			linalg.yield %0 : f32
		}
	}
	return %alloc : memref<32768x?x30xf32>
}

// CHECK-LABEL: func.func @Elementwise
// CHECK-DAG: %[[C983040:.*]] = arith.constant 983040 : index
// CHECK-DAG: %[[C30:.*]] = arith.constant 30 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[COLLAPSE:.*]] = memref.collapse_shape %arg0
// CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[C2]] : memref<32x1024x?x30xf32>
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<32768x?x30xf32>
// CHECK-NEXT: %[[CNT:.*]] = arith.muli %dim, %[[C983040]] : index
// CHECK-NEXT: scf.forall (%[[ARG1:.*]]) in (%[[CNT]]) {
// CHECK-NEXT: %[[V1:.*]] = arith.remsi %arg1, %[[C30]] : index
// CHECK-NEXT: %[[V2:.*]] = arith.divsi %arg1, %[[C30]] : index
// CHECK-NEXT: %[[V3:.*]] = arith.remsi %[[V2]], %[[DIM]] : index
// CHECK-NEXT: %[[V4:.*]] = arith.divsi %[[V2]], %[[DIM]] : index
// CHECK-NEXT: %[[SUBVIEW:.*]] = memref.subview %[[COLLAPSE]][%[[V4]], %[[V3]], %[[V1]]] [1, 1, 1] [1, 1, 1] : memref<32768x?x30xf32> to memref<1x1x1xf32, strided<[?, 30, 1], offset: ?>>
// CHECK-NEXT: %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC]][%[[V4]], %[[V3]], %[[V1]]] [1, 1, 1] [1, 1, 1] : memref<32768x?x30xf32> to memref<1x1x1xf32, strided<[?, 30, 1], offset: ?>>
// CHCCK-NEXT: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[SUBVIEW]] : memref<1x1x1xf32, strided<[?, 30, 1], offset: ?>>) outs(%[[SUBVIEW_0]] : memref<1x1x1xf32, strided<[?, 30, 1], offset: ?>>) attrs =  {__byteir_gpu_tile_elementwise_0} {
