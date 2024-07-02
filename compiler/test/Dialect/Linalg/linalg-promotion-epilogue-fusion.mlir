// RUN: byteir-opt -linalg-promotion --cse --canonicalize %s | FileCheck %s
#map = affine_map<(d0) -> (d0 * 128)>
module {
  func.func private @Unknown0(%arg0: memref<5376x2048xf16>, %arg1: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<5376x5376xf16>
    scf.forall (%arg2, %arg3) in (42, 42) {
      %0 = affine.apply #map(%arg2)
      %1 = affine.apply #map(%arg3)
      %subview = memref.subview %alloc[%0, %1] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
      linalg.fill ins(%cst : f16) outs(%subview : memref<128x128xf16, strided<[5376, 1], offset: ?>>)
      scf.for %arg4 = %c0 to %c2048 step %c32 {
        %subview_0 = memref.subview %arg0[%0, %arg4] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>>
        %subview_1 = memref.subview %arg1[%arg4, %1] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>>
        linalg.matmul {__byteir_gpu_tile_gemm_0, __byteir_mma__, __byteir_mma_level__ = "Threadblock", __byteir_target__ = "nv_sm_80"} ins(%subview_0, %subview_1 : memref<128x32xf16, strided<[2048, 1], offset: ?>>, memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%subview : memref<128x128xf16, strided<[5376, 1], offset: ?>>)
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<128x128xf16, strided<[5376, 1], offset: ?>>) {
      ^bb0(%out: f16):
        %6 = arith.maximumf %out, %cst : f16
        linalg.yield %6 : f16
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %alloc : memref<5376x5376xf16>
  }
}
// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0 * 128)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func private @Unknown0(%[[ARG0:.*]]: memref<5376x2048xf16>, %[[ARG1:.*]]: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
// CHECK-NEXT:     %[[CST:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C2048:.*]] = arith.constant 2048 : index
// CHECK-NEXT:     %[[C32:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[ALLOC:.*]] = memref.alloc() : memref<5376x5376xf16>
// CHECK-NEXT:     scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (42, 42) {
// CHECK-NEXT:       %[[ALLOC_2:.*]] = memref.alloc() {__byteir_alloca_accumulator__} : memref<128x128xf16, #gpu.address_space<workgroup>>
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() {__byteir_alloca_matrix_b__} : memref<32x128xf16, #gpu.address_space<workgroup>>
// CHECK-NEXT:       %[[ALLOC_1:.*]] = memref.alloc() {__byteir_alloca_matrix_a__} : memref<128x32xf16, #gpu.address_space<workgroup>>
// CHECK-NEXT:       %[[APPLY_MAP0:.*]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK-NEXT:       %[[APPLY_MAP1:.*]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK-NEXT:       %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]][%[[APPLY_MAP0]], %[[APPLY_MAP1]]] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
// CHECK-NEXT:       linalg.fill ins(%[[CST]] : f16) outs(%[[ALLOC_2]] : memref<128x128xf16, #gpu.address_space<workgroup>>)
// CHECK-NEXT:       scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2048]] step %[[C32]] {
// CHECK-NEXT:         %[[SUBVIEW_2:.*]] = memref.subview %[[ARG0]][%[[APPLY_MAP0]], %[[ARG4]]] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>>
// CHECK-NEXT:         %[[SUBVIEW_3:.*]] = memref.subview %[[ARG1]][%[[ARG4]], %[[APPLY_MAP1]]] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>>
// CHECK-NEXT:         linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW_2]] : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%[[ALLOC_1]] : memref<128x32xf16, #gpu.address_space<workgroup>>)
// CHECK-NEXT:         linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%[[SUBVIEW_3]] : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%[[ALLOC_0]] : memref<32x128xf16, #gpu.address_space<workgroup>>)
// CHECK-NEXT:         gpu.barrier
// CHECK-NEXT:         linalg.matmul {__byteir_gpu_tile_gemm_0, __byteir_mma__, __byteir_mma_level__ = "Threadblock", __byteir_target__ = "nv_sm_80"} ins(%[[ALLOC_1]], %[[ALLOC_0]] : memref<128x32xf16, #gpu.address_space<workgroup>>, memref<32x128xf16, #gpu.address_space<workgroup>>) outs(%[[ALLOC_2]] : memref<128x128xf16, #gpu.address_space<workgroup>>)
// CHECK-NEXT:       }
// CHECK-NEXT:       gpu.barrier
// CHECK-NEXT:       linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ALLOC_2]] : memref<128x128xf16, #gpu.address_space<workgroup>>) outs(%[[SUBVIEW]] : memref<128x128xf16, strided<[5376, 1], offset: ?>>) attrs =  {__internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} {
// CHECK-NEXT:       ^bb0(%in: f16, %out: f16):
// CHECK-NEXT:         %2 = arith.maximumf %in, %cst : f16
// CHECK-NEXT:         linalg.yield %2 : f16
// CHECK-NEXT:       }
// CHECK-NEXT:     } {mapping = [#gpu.block<y>, #gpu.block<x>]}
// CHECK-NEXT:     return %[[ALLOC]] : memref<5376x5376xf16>
// CHECK-NEXT:   }
// CHECK-NEXT: }