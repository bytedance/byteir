// RUN: byteir-opt %s --horizontal-fusion-on-scf  --cse --canonicalize | FileCheck %s
#map = affine_map<(d0, d1) -> (d0)>
module {
  func.func private @Unknown0(%arg0: memref<32x16xf32>, %arg1: memref<32x16xf32>, %arg2: memref<32x16xf32>) -> memref<32x16xf32> attributes {__byteir_elementwise_fusion__} {
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<32x16xf32>
    scf.forall (%arg3) in (2) {
      %0 = arith.muli %arg3, %c256 : index
      scf.forall (%arg4) in (256) {
        %1 = arith.addi %arg4, %0 : index
        %2 = arith.remsi %1, %c16 : index
        %3 = arith.divsi %1, %c16 : index
        %4 = memref.load %arg0[%3, %2] : memref<32x16xf32>
        %5 = memref.load %arg1[%3, %2] : memref<32x16xf32>
        %6 = memref.load %arg2[%3, %2] : memref<32x16xf32>
        %7 = arith.mulf %4, %5 : f32
        %8 = arith.divf %7, %6 : f32
        memref.store %8, %alloc[%3, %2] : memref<32x16xf32>
      } {mapping = [#gpu.thread<linear_dim_0>]}
    } {mapping = [#gpu.block<linear_dim_0>]}
    return %alloc : memref<32x16xf32>
  }
  func.func private @Unknown1(%arg0: memref<32x16xf32>, %arg1: memref<32x16xf32>) -> memref<16xf32> attributes {__byteir_reduction_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<16xf32>
    scf.forall (%arg2) in (16) {
      scf.forall (%arg3) in (1) {
        %0 = vector.transfer_read %arg0[%c0, %arg2], %cst {in_bounds = [true], permutation_map = #map} : memref<32x16xf32>, vector<32xf32>
        %1 = vector.transfer_read %arg1[%c0, %arg2], %cst {in_bounds = [true], permutation_map = #map} : memref<32x16xf32>, vector<32xf32>
        %2 = arith.mulf %0, %0 : vector<32xf32>
        %3 = arith.subf %2, %1 : vector<32xf32>
        %4 = vector.reduction <add>, %3, %cst : vector<32xf32> into f32
        %5 = vector.insertelement %4, %cst_0[%c0 : index] : vector<1xf32>
        %6 = vector.extract %5[0] : f32 from vector<1xf32>
        %7 = vector.broadcast %6 : f32 to vector<f32>
        vector.transfer_write %7, %alloc[%arg2] : vector<f32>, memref<16xf32>
      } {mapping = [#gpu.warp<linear_dim_0>]}
    } {mapping = [#gpu.block<linear_dim_0>]}
    return %alloc : memref<16xf32>
  }
  func.func private @Unknown2(%arg0: memref<16xf32>) -> memref<16xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<16xf32>
    scf.forall (%arg1) in (1) {
      %0 = arith.muli %arg1, %c256 : index
      %1 = arith.subi %c16, %0 : index
      %2 = arith.cmpi sgt, %1, %c256 : index
      %3 = arith.select %2, %c256, %1 : index
      scf.forall (%arg2) in (%3) {
        %4 = arith.addi %arg2, %0 : index
        %5 = memref.load %arg0[%4] : memref<16xf32>
        %6 = arith.maximumf %5, %cst : f32
        memref.store %6, %alloc[%4] : memref<16xf32>
      } {mapping = [#gpu.thread<linear_dim_0>]}
    } {mapping = [#gpu.block<linear_dim_0>]}
    return %alloc : memref<16xf32>
  }
  func.func private @Unknown3(%arg0: memref<16xf32>, %arg1: memref<32x16xf32>) -> memref<32x16xf32> attributes {__byteir_elementwise_fusion__} {
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<32x16xf32>
    scf.forall (%arg2) in (2) {
      %0 = arith.muli %arg2, %c256 : index
      scf.forall (%arg3) in (256) {
        %1 = arith.addi %arg3, %0 : index
        %2 = arith.remsi %1, %c16 : index
        %3 = arith.divsi %1, %c16 : index
        %4 = memref.load %arg0[%2] : memref<16xf32>
        %5 = memref.load %arg1[%3, %2] : memref<32x16xf32>
        %6 = arith.addf %4, %5 : f32
        memref.store %6, %alloc[%3, %2] : memref<32x16xf32>
      } {mapping = [#gpu.thread<linear_dim_0>]}
    } {mapping = [#gpu.block<linear_dim_0>]}
    return %alloc : memref<32x16xf32>
  }
  func.func @main(%arg0: memref<32x16xf32>, %arg1: memref<32x16xf32>, %arg2: memref<32x16xf32>) -> (memref<32x16xf32>, memref<32x16xf32>) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg0, %arg1, %arg2) : (memref<32x16xf32>, memref<32x16xf32>, memref<32x16xf32>) -> memref<32x16xf32>
    %1 = call @Unknown1(%0, %arg0) : (memref<32x16xf32>, memref<32x16xf32>) -> memref<16xf32>
    %2 = call @Unknown2(%1) : (memref<16xf32>) -> memref<16xf32>
    %3 = call @Unknown3(%2, %0) : (memref<16xf32>, memref<32x16xf32>) -> memref<32x16xf32>
    return %0, %3 : memref<32x16xf32>, memref<32x16xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: scf.forall
// CHECK: scf.forall
// CHECK-not: scf.forall
