// RUN: byteir-opt -gpu-distribute-shared-memory-copy --cse --canonicalize --fold-memref-alias-ops --canonicalize --cse --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 128)>
module {
  func.func private @Unknown0(%arg0: memref<5376x2048xf16>, %arg1: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<5376x5376xf16>
    scf.forall (%arg2, %arg3) in (42, 42) {
      %alloca = memref.alloca() {__byteir_alloca_accumulator__} : memref<128x128xf16, #gpu.address_space<workgroup>>
      %alloca_0 = memref.alloca() {__byteir_alloca_matrix_b__} : memref<32x128xf16, #gpu.address_space<workgroup>>
      %alloca_1 = memref.alloca() {__byteir_alloca_matrix_a__} : memref<128x32xf16, #gpu.address_space<workgroup>>
      %0 = affine.apply #map(%arg2)
      %1 = affine.apply #map(%arg3)
      %subview = memref.subview %alloc[%0, %1] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
      linalg.fill ins(%cst : f16) outs(%alloca : memref<128x128xf16, #gpu.address_space<workgroup>>)
      scf.for %arg4 = %c0 to %c2048 step %c32 {
        %subview_2 = memref.subview %arg0[%0, %arg4] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>>
        %subview_3 = memref.subview %arg1[%arg4, %1] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>>
        linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%subview_2 : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%alloca_1 : memref<128x32xf16, #gpu.address_space<workgroup>>)
        linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%subview_3 : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%alloca_0 : memref<32x128xf16, #gpu.address_space<workgroup>>)
        linalg.matmul {__byteir_gpu_tile_gemm_0, __byteir_mma__, __byteir_mma_level__ = "Threadblock", __byteir_target__ = "nv_sm_80"} ins(%alloca_1, %alloca_0 : memref<128x32xf16, #gpu.address_space<workgroup>>, memref<32x128xf16, #gpu.address_space<workgroup>>) outs(%alloca : memref<128x128xf16, #gpu.address_space<workgroup>>)
      }
      linalg.copy {__byteir_store_matrix_c__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%alloca : memref<128x128xf16, #gpu.address_space<workgroup>>) outs(%subview : memref<128x128xf16, strided<[5376, 1], offset: ?>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %alloc : memref<5376x5376xf16>
  }
}
// CHECK-LABEL:      scf.for %arg4 = %c0 to %c2048 step %c32 {

// CHECK:            %[[READ0:.*]] = vector.transfer_read %arg0[%{{.*}}, %{{.*}}], %cst {in_bounds = [true, true]} : memref<5376x2048xf16>, vector<1x8xf16>
// CHECK:            vector.transfer_write %[[READ0]], %alloca_1[%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<1x8xf16>, memref<128x32xf16, #gpu.address_space<workgroup>>
// CHECK:            %[[READ1:.*]] = vector.transfer_read %arg0[%{{.*}}, %{{.*}}], %cst {in_bounds = [true, true]} : memref<5376x2048xf16>, vector<1x8xf16>
// CHECK:            vector.transfer_write %[[READ1]], %alloca_1[%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<1x8xf16>, memref<128x32xf16, #gpu.address_space<workgroup>>
// CHECK:            %[[READ2:.*]] = vector.transfer_read %arg0[%{{.*}}, %{{.*}}], %cst {in_bounds = [true, true]} : memref<5376x2048xf16>, vector<1x8xf16>
// CHECK:            vector.transfer_write %[[READ2]], %alloca_1[%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<1x8xf16>, memref<128x32xf16, #gpu.address_space<workgroup>>
// CHECK:            %[[READ3:.*]] = vector.transfer_read %arg0[%{{.*}}, %{{.*}}], %cst {in_bounds = [true, true]} : memref<5376x2048xf16>, vector<1x8xf16>
// CHECK:            vector.transfer_write %[[READ3]], %alloca_1[%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<1x8xf16>, memref<128x32xf16, #gpu.address_space<workgroup>>

// CHECK:            %[[READ4:.*]] = vector.transfer_read %arg1[%{{.*}}, %{{.*}}], %cst {in_bounds = [true, true]} : memref<2048x5376xf16>, vector<1x8xf16>
// CHECK:            vector.transfer_write %[[READ4]], %alloca_0[%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<1x8xf16>, memref<32x128xf16, #gpu.address_space<workgroup>>
// CHECK:            %[[READ5:.*]] = vector.transfer_read %arg1[%{{.*}}, %{{.*}}], %cst {in_bounds = [true, true]} : memref<2048x5376xf16>, vector<1x8xf16>
// CHECK:            vector.transfer_write %[[READ5]], %alloca_0[%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<1x8xf16>, memref<32x128xf16, #gpu.address_space<workgroup>>
// CHECK:            %[[READ6:.*]] = vector.transfer_read %arg1[%{{.*}}, %{{.*}}], %cst {in_bounds = [true, true]} : memref<2048x5376xf16>, vector<1x8xf16>
// CHECK:            vector.transfer_write %[[READ6]], %alloca_0[%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<1x8xf16>, memref<32x128xf16, #gpu.address_space<workgroup>>
// CHECK:            %[[READ7:.*]] = vector.transfer_read %arg1[%{{.*}}, %{{.*}}], %cst {in_bounds = [true, true]} : memref<2048x5376xf16>, vector<1x8xf16>
// CHECK:            vector.transfer_write %[[READ7]], %alloca_0[%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<1x8xf16>, memref<32x128xf16, #gpu.address_space<workgroup>>

// CHECK:            linalg.matmul {__byteir_gpu_tile_gemm_0, __byteir_mma__, __byteir_mma_level__ = "Threadblock", __byteir_target__ = "nv_sm_80"} ins(%alloca_1, %alloca_0 : memref<128x32xf16, #gpu.address_space<workgroup>>, memref<32x128xf16, #gpu.address_space<workgroup>>) outs(%alloca : memref<128x128xf16, #gpu.address_space<workgroup>>)
// CHECK-COUNT-8:    vector.transfer_read %alloca{{.*}} : memref<{{.*}}>, vector<1x8xf16>
// CHECK-COUNT-8:    vector.transfer_write %{{.*}}, %alloc[{{.*}}] {{.*}} : vector<1x8xf16>, memref<{{.*}}>
