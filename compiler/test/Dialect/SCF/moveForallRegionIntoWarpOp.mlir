// RUN: byteir-opt %s -move-forall-region-into-warp-op | FileCheck %s

func.func @Reduction2D(%arg0: memref<1024x1024xf32>) -> memref<1024xf32> attributes {__byteir_reduction_fusion__} {
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<1024xf32>
  scf.forall (%arg1) in (1024) {
    %alloca = memref.alloca() : memref<256xf32, #gpu.address_space<workgroup>>
    scf.forall (%arg2) in (256) {
      %0 = scf.for %arg3 = %arg2 to %c1024 step %c256 iter_args(%arg4 = %cst_0) -> (f32) {
        %1 = memref.load %arg0[%arg1, %arg3] : memref<1024x1024xf32>
        %2 = arith.addf %arg4, %1 : f32
        scf.yield %2 : f32
      }
      memref.store %0, %alloca[%arg2] : memref<256xf32, #gpu.address_space<workgroup>>
    } {mapping = [#gpu.thread<x>]}
    %alloca_1 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
    scf.forall (%arg2) in (8) {
      %0 = arith.muli %arg2, %c32 : index
      %1 = vector.transfer_read %alloca[%0], %cst_0 {in_bounds = [true]} : memref<256xf32, #gpu.address_space<workgroup>>, vector<32xf32>
      %2 = vector.reduction <add>, %1, %cst_0 : vector<32xf32> into f32
      %3 = vector.insertelement %2, %cst[%c0 : index] : vector<1xf32>
      %4 = vector.extract %3[0] : f32 from vector<1xf32>
      %5 = vector.broadcast %4 : f32 to vector<f32>
      vector.transfer_write %5, %alloca_1[%arg2] : vector<f32>, memref<8xf32, #gpu.address_space<workgroup>>
    } {mapping = [#gpu.warp<linear_dim_0>]}
    scf.forall (%arg2) in (1) {
      %0 = arith.muli %arg2, %c8 : index
      %1 = vector.transfer_read %alloca_1[%0], %cst_0 {in_bounds = [true]} : memref<8xf32, #gpu.address_space<workgroup>>, vector<8xf32>
      %2 = vector.reduction <add>, %1, %cst_0 : vector<8xf32> into f32
      %3 = vector.insertelement %2, %cst[%c0 : index] : vector<1xf32>
      %4 = vector.extract %3[0] : f32 from vector<1xf32>
      %5 = vector.broadcast %4 : f32 to vector<f32>
      vector.transfer_write %5, %alloc[%arg1] : vector<f32>, memref<1024xf32>
    } {mapping = [#gpu.warp<linear_dim_0>]}
  } {mapping = [#gpu.block<linear_dim_0>]}
  return %alloc : memref<1024xf32>
}


// CHECK-LABEL: func.func @Reduction2D
	// CHECK: scf.forall (%{{.*}}) in (1024) {
    // CHECK: scf.forall (%{{.*}}) in (256) {
      // CHECK-NOT: vector.warp_execute_on_lane_0
    // CHECK: } {mapping = [#gpu.thread<x>]}

		// CHECK: scf.forall (%{{.*}}) in (8) {
      // CHECK-NEXT: %[[LANE_ID0:.*]] = gpu.lane_id
      // CHECK-NEXT: vector.warp_execute_on_lane_0(%[[LANE_ID0]])[32] {
    // CHECK: } {mapping = [#gpu.warp<linear_dim_0>]}

    // CHECK: scf.forall (%{{.*}}) in (1) {
      // CHECK-NEXT: %[[LANE_ID1:.*]] = gpu.lane_id
      // CHECK-NEXT: %[[C8_1:.*]] = arith.constant 8 : index
      // CHECK-NEXT: %[[CMP:.*]] = arith.cmpi ult, %[[LANE_ID1]], %[[C8_1]] : index
      // CHECK-NEXT: scf.if %[[CMP]] {
        // CHECK-NEXT: vector.warp_execute_on_lane_0(%[[LANE_ID1]])[8] {
         
    // CHECK: } {mapping = [#gpu.warp<linear_dim_0>]}
  // CHECK: } {mapping = [#gpu.block<linear_dim_0>]}
