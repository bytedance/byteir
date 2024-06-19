// RUN: byteir-opt %s -fuse-nested-forall | FileCheck %s

func.func @Copy(%arg0: memref<32x64xf32>, %arg1: memref<32x64xf32>) attributes {__byteir_reduction_fusion__} {
	scf.forall (%arg2) in (32) {
		scf.forall (%arg3) in (64) {
		%0 = memref.load %arg0[%arg2, %arg3] : memref<32x64xf32>
		memref.store %0, %arg1[%arg2, %arg3] : memref<32x64xf32>
		} {mapping = [#gpu.block<linear_dim_0>]}
	} {mapping = [#gpu.block<linear_dim_1>]}
	return
}

// CHECK-LABEL: func.func @Copy
	// CHECK: scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (32, 64) {
		// CHECK-NEXT: %[[VAL:.*]] = memref.load %arg0[%[[IV0]], %[[IV1]]] : memref<32x64xf32>
		// CHECK-NEXT: memref.store %[[VAL]], %arg1[%[[IV0]], %[[IV1]]] : memref<32x64xf32>
	// CHECK-NEXT: } {mapping = [#gpu.block<linear_dim_1>, #gpu.block<linear_dim_0>]}
