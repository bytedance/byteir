// RUN: byteir-opt %s --horizontal-fusion-on-scf  --cse --canonicalize | FileCheck %s
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
module {
  func.func private @Unknown0(%arg0: memref<32x16xf32>, %arg1: memref<32x16xf32>, %arg2: memref<32x16xf32>) -> memref<32x16xf32> attributes {__byteir_elementwise_fusion__} {
    %c16 = arith.constant 16 : index
    %alloc = memref.alloc() : memref<32x16xf32>
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %0 = arith.muli %c1, %c256 : index
    %c0_1 = arith.constant 0 : index
    %c512_2 = arith.constant 512 : index
    %1 = arith.subi %c512_2, %c0_1 : index
    %2 = arith.ceildivsi %1, %0 : index
    %c1_3 = arith.constant 1 : index
    scf.forall (%arg3) in (%2) {
      %3 = arith.muli %arg3, %0 : index
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      scf.forall (%arg4) in (%0) {
        %4 = arith.addi %arg4, %3 : index
        %5 = arith.remsi %4, %c16 : index
        %6 = arith.divsi %4, %c16 : index
        %7 = memref.load %arg0[%6, %5] : memref<32x16xf32>
        %8 = memref.load %arg1[%6, %5] : memref<32x16xf32>
        %9 = memref.load %arg2[%6, %5] : memref<32x16xf32>
        %10 = arith.mulf %7, %8 : f32
        %11 = arith.divf %10, %9 : f32
        memref.store %11, %alloc[%6, %5] : memref<32x16xf32>
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
    %alloc = memref.alloc() : memref<16xf32>
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %0 = arith.muli %c1, %c256 : index
    %c0_1 = arith.constant 0 : index
    %c16_2 = arith.constant 16 : index
    %1 = arith.subi %c16_2, %c0_1 : index
    %2 = arith.ceildivsi %1, %0 : index
    %c1_3 = arith.constant 1 : index
    scf.forall (%arg1) in (%2) {
      %3 = arith.muli %arg1, %0 : index
      %4 = affine.min #map1(%0, %c16, %3)
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      scf.forall (%arg2) in (%4) {
        %5 = arith.addi %arg2, %3 : index
        %6 = memref.load %arg0[%5] : memref<16xf32>
        %7 = arith.maximumf %6, %cst : f32
        memref.store %7, %alloc[%5] : memref<16xf32>
      } {mapping = [#gpu.thread<linear_dim_0>]}
    } {mapping = [#gpu.block<linear_dim_0>]}
    return %alloc : memref<16xf32>
  }
  func.func private @Unknown3(%arg0: memref<16xf32>, %arg1: memref<32x16xf32>) -> memref<32x16xf32> attributes {__byteir_elementwise_fusion__} {
    %c16 = arith.constant 16 : index
    %alloc = memref.alloc() : memref<32x16xf32>
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %0 = arith.muli %c1, %c256 : index
    %c0_1 = arith.constant 0 : index
    %c512_2 = arith.constant 512 : index
    %1 = arith.subi %c512_2, %c0_1 : index
    %2 = arith.ceildivsi %1, %0 : index
    %c1_3 = arith.constant 1 : index
    scf.forall (%arg2) in (%2) {
      %3 = arith.muli %arg2, %0 : index
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      scf.forall (%arg3) in (%0) {
        %4 = arith.addi %arg3, %3 : index
        %5 = arith.remsi %4, %c16 : index
        %6 = arith.divsi %4, %c16 : index
        %7 = memref.load %arg0[%5] : memref<16xf32>
        %8 = memref.load %arg1[%6, %5] : memref<32x16xf32>
        %9 = arith.addf %7, %8 : f32
        memref.store %9, %alloc[%6, %5] : memref<32x16xf32>
      } {mapping = [#gpu.thread<linear_dim_0>]}
    } {mapping = [#gpu.block<linear_dim_0>]}
    return %alloc : memref<32x16xf32>
  }
  func.func private @Unknown4(%arg0: memref<32x16xf32>) -> memref<32x16xf32> attributes {__byteir_elementwise_fusion__} {
    %c16 = arith.constant 16 : index
    %alloc = memref.alloc() : memref<32x16xf32>
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %0 = arith.muli %c1, %c256 : index
    %c0_1 = arith.constant 0 : index
    %c512_2 = arith.constant 512 : index
    %1 = arith.subi %c512_2, %c0_1 : index
    %2 = arith.ceildivsi %1, %0 : index
    %c1_3 = arith.constant 1 : index
    scf.forall (%arg1) in (%2) {
      %3 = arith.muli %arg1, %0 : index
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      scf.forall (%arg2) in (%0) {
        %4 = arith.addi %arg2, %3 : index
        %5 = arith.remsi %4, %c16 : index
        %6 = arith.divsi %4, %c16 : index
        %7 = memref.load %arg0[%6, %5] : memref<32x16xf32>
        %8 = arith.mulf %7, %7 : f32
        memref.store %8, %alloc[%6, %5] : memref<32x16xf32>
      } {mapping = [#gpu.thread<linear_dim_0>]}
    } {mapping = [#gpu.block<linear_dim_0>]}
    return %alloc : memref<32x16xf32>
  }
  func.func @main(%arg0: memref<32x16xf32>, %arg1: memref<32x16xf32>, %arg2: memref<32x16xf32>) -> (memref<32x16xf32>, memref<32x16xf32>) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg0, %arg1, %arg2) : (memref<32x16xf32>, memref<32x16xf32>, memref<32x16xf32>) -> memref<32x16xf32>
    %1 = call @Unknown1(%0, %arg0) : (memref<32x16xf32>, memref<32x16xf32>) -> memref<16xf32>
    %2 = call @Unknown2(%1) : (memref<16xf32>) -> memref<16xf32>
    %3 = call @Unknown3(%2, %0) : (memref<16xf32>, memref<32x16xf32>) -> memref<32x16xf32>
    %4 = call @Unknown4(%arg1) : (memref<32x16xf32>) -> memref<32x16xf32>
    return %3, %4 : memref<32x16xf32>, memref<32x16xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: scf.forall
  // CHECK: scf.forall
// CHECK: scf.forall
  // CHECK: scf.forall
// CHECK: scf.forall
  // CHECK: scf.forall
// CHECK: scf.forall
  // CHECK: case 0
    // CHECK: scf.forall
  // CHECK: case 1
    // CHECK: scf.forall
  // CHECK: default
