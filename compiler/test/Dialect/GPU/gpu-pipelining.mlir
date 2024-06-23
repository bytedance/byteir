// RUN: byteir-opt -gpu-pipelining="stages=3" -canonicalize --cse --verify-diagnostics %s | FileCheck %s

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

// CHECK-LABEL: scf.forall (%arg2, %arg3) in (42, 42) {

// init:
// CHECK: %[[ALLOCA:.*]] = memref.alloca() {__byteir_alloca_accumulator__} : memref<128x128xf16, #gpu.address_space<workgroup>>
// CHECK: %[[ALLOCA0:.*]] = memref.alloca() {__byteir_alloca_matrix_b__} : memref<3x32x128xf16, #gpu.address_space<workgroup>
// CHECK: %[[ALLOCA1:.*]] = memref.alloca() {__byteir_alloca_matrix_a__} : memref<3x128x32xf16, #gpu.address_space<workgroup>
// CHECK: %[[IDX0:.*]] = affine.apply #map(%{{.*}})
// CHECK: %[[IDX1:.*]] = affine.apply #map(%{{.*}})
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC:.*]][%[[IDX0]], %[[IDX1]]] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
// CHECK: linalg.fill ins(%[[CST:.*]] : f16) outs(%[[ALLOCA]] : memref<128x128xf16, #gpu.address_space<workgroup>>)

// prelogue0:
// CHECK: %[[SUBVIEW2:.*]] = memref.subview %[[ALLOCA1]][0, 0, 0] [1, 128, 32] [1, 1, 1] : memref<3x128x32xf16, #gpu.address_space<workgroup>> to memref<128x32xf16, strided<[32, 1]>, #gpu.address_space<workgroup>
// CHECK: %[[CAST2:.*]] = memref.cast %[[SUBVIEW2]] : memref<128x32xf16, strided<[32, 1]>, #gpu.address_space<workgroup>> to memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: %[[SUBVIEW3:.*]] = memref.subview %[[ALLOCA0]][0, 0, 0] [1, 32, 128] [1, 1, 1] : memref<3x32x128xf16, #gpu.address_space<workgroup>> to memref<32x128xf16, strided<[128, 1]>, #gpu.address_space<workgroup>
// CHECK: %[[CAST3:.*]] = memref.cast %[[SUBVIEW3]] : memref<32x128xf16, strided<[128, 1]>, #gpu.address_space<workgroup>> to memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: %[[SUBVIEW5:.*]] = memref.subview %arg0[%[[IDX0]], 0] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>
// CHECK: %[[SUBVIEW6:.*]] = memref.subview %arg1[0, %[[IDX1]]] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>
// CHECK: linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW5]] : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%[[SUBVIEW2]] : memref<128x32xf16, strided<[32, 1]>, #gpu.address_space<workgroup>>)
// CHECK: linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW6]] : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%[[SUBVIEW3]] : memref<32x128xf16, strided<[128, 1]>, #gpu.address_space<workgroup>>)
// CHECK: nvvm.cp.async.commit.group

// prelogue1:
// CHECK: %[[SUBVIEW7:.*]] = memref.subview %[[ALLOCA1]][1, 0, 0] [1, 128, 32] [1, 1, 1] : memref<3x128x32xf16, #gpu.address_space<workgroup>> to memref<128x32xf16, strided<[32, 1], offset: 4096>, #gpu.address_space<workgroup>
// CHECK: %[[CAST4:.*]] = memref.cast %[[SUBVIEW7]] : memref<128x32xf16, strided<[32, 1], offset: 4096>, #gpu.address_space<workgroup>> to memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: %[[SUBVIEW9:.*]] = memref.subview %[[ALLOCA0]][1, 0, 0] [1, 32, 128] [1, 1, 1] : memref<3x32x128xf16, #gpu.address_space<workgroup>> to memref<32x128xf16, strided<[128, 1], offset: 4096>, #gpu.address_space<workgroup>
// CHECK: %[[CAST5:.*]] = memref.cast %[[SUBVIEW9]] : memref<32x128xf16, strided<[128, 1], offset: 4096>, #gpu.address_space<workgroup>> to memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: %[[SUBVIEW11:.*]] = memref.subview %arg0[%[[IDX0]], 32] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>
// CHECK: %[[SUBVIEW12:.*]] = memref.subview %arg1[32, %[[IDX1]]] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>
// CHECK: linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW11]] : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%[[SUBVIEW7]] : memref<128x32xf16, strided<[32, 1], offset: 4096>, #gpu.address_space<workgroup>>)
// CHECK: linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW12]] : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%[[SUBVIEW9]] : memref<32x128xf16, strided<[128, 1], offset: 4096>, #gpu.address_space<workgroup>>)
// CHECK: nvvm.cp.async.commit.group

// prelogue2:
// CHECK: %[[SUBVIEW13:.*]] = memref.subview %[[ALLOCA1]][2, 0, 0] [1, 128, 32] [1, 1, 1] : memref<3x128x32xf16, #gpu.address_space<workgroup>> to memref<128x32xf16, strided<[32, 1], offset: 8192>, #gpu.address_space<workgroup>
// CHECK: %[[CAST6:.*]] = memref.cast %[[SUBVIEW13]] : memref<128x32xf16, strided<[32, 1], offset: 8192>, #gpu.address_space<workgroup>> to memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: %[[SUBVIEW15:.*]] = memref.subview %[[ALLOCA0]][2, 0, 0] [1, 32, 128] [1, 1, 1] : memref<3x32x128xf16, #gpu.address_space<workgroup>> to memref<32x128xf16, strided<[128, 1], offset: 8192>, #gpu.address_space<workgroup>
// CHECK: %[[CAST7:.*]] = memref.cast %[[SUBVIEW15]] : memref<32x128xf16, strided<[128, 1], offset: 8192>, #gpu.address_space<workgroup>> to memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: %[[SUBVIEW17:.*]] = memref.subview %arg0[%[[IDX0]], 64] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>
// CHECK: %[[SUBVIEW18:.*]] = memref.subview %arg1[64, %[[IDX1]]] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>
// CHECK: linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW17]] : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%[[SUBVIEW13]] : memref<128x32xf16, strided<[32, 1], offset: 8192>, #gpu.address_space<workgroup>>)
// CHECK: linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW18]] : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%[[SUBVIEW15]] : memref<32x128xf16, strided<[128, 1], offset: 8192>, #gpu.address_space<workgroup>>)
// CHECK: nvvm.cp.async.commit.group

// kernel:
// CHECK: %[[CAST:.*]] = scf.for %arg4 = %c0 to %c2048 step %c32 iter_args(%arg5 = %[[CAST2]], %arg6 = %[[CAST4]], %arg7 = %[[CAST6]], %arg8 = %[[CAST3]], %arg9 = %[[CAST5]], %arg10 = %[[CAST7]]) -> (memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>) {
// CHECK: nvvm.cp.async.wait.group 2
// CHECK: linalg.matmul {__byteir_gpu_tile_gemm_0, __byteir_mma__, __byteir_mma_level__ = "Threadblock", __byteir_target__ = "nv_sm_80"} ins(%arg5, %arg8 : memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>) outs(%[[ALLOCA]] : memref<128x128xf16, #gpu.address_space<workgroup>>)

// CHECK: %[[IDX5:.*]] = affine.apply #map1(%[[IDX4:.*]])
// CHECK: %[[SUBVIEW19:.*]] = memref.subview %[[ALLOCA1]][%[[IDX5]], 0, 0] [1, 128, 32] [1, 1, 1] : memref<3x128x32xf16, #gpu.address_space<workgroup>> to memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: %[[SUBVIEW20:.*]] = memref.subview %[[ALLOCA0]][%[[IDX5]], 0, 0] [1, 32, 128] [1, 1, 1] : memref<3x32x128xf16, #gpu.address_space<workgroup>> to memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: %[[SUBVIEW21:.*]] = memref.subview %arg0[%[[IDX0]], %[[IDX8:.*]]] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>
// CHECK: %[[SUBVIEW22:.*]] = memref.subview %arg1[%[[IDX9:.*]], %[[IDX1]]] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>
// CHECK: scf.if %[[CMP:.*]] {
// CHECK: linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW21]] : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%[[SUBVIEW19]] : memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>)
// CHECK: linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW22]] : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%[[SUBVIEW20]] : memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>)
// CHECK: nvvm.cp.async.commit.group
// CHECK: scf.yield %arg6, %arg7, %[[SUBVIEW19]], %arg9, %arg10, %[[SUBVIEW20]] : memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<128x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<32x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>
// CHECK: }

// copy back to global memory:
// CHECK: linalg.copy {__byteir_store_matrix_c__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[ALLOCA]] : memref<128x128xf16, #gpu.address_space<workgroup>>) outs(%[[SUBVIEW]] : memref<128x128xf16, strided<[5376, 1], offset: ?>>)
