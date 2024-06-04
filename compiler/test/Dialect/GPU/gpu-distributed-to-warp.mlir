// RUN: byteir-opt --gpu-distributed-to-warp  -canonicalize -cse --verify-diagnostics %s | FileCheck %s

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
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %alloc : memref<5376x5376xf16>
  }
}

// CHECK: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: #[[MAP2:.*]] = affine_map<(d0) -> ((d0 floordiv 32) * 64)>
// CHECK: %[[THREAD_ID_X:.*]] = gpu.thread_id x
// CHECK-NEXT: %[[THREAD_ID_Y:.*]] = gpu.thread_id y
// CHECK-NEXT: %[[APPLY_MAP1:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_Y]]]
// CHECK-NEXT: scf.for %[[ARG4:.*]] = %[[APPLY_MAP1]] to %c128 step %c128 {
// CHECK-NEXT:   %[[APPLY_MAP2:.*]] = affine.apply #[[MAP2]](%[[THREAD_ID_X]])
// CHECK-NEXT:   scf.for %[[ARG5:.*]] = %[[APPLY_MAP2]] to %c128 step %c128 {
// CHECK-NEXT:     %[[SUBVIEW_0:.*]] = memref.subview %subview[%[[ARG4]], %[[ARG5]]] [64, 64] [1, 1] : memref<128x128xf16, strided<[5376, 1], offset: ?>> to memref<64x64xf16, strided<[5376, 1], offset: ?>>
// CHECK-NEXT:     linalg.fill {__internal_linalg_transform__ = "vectorize"} ins(%{{.*}} : f16) outs(%[[SUBVIEW_0]] : memref<64x64xf16, strided<[5376, 1], offset: ?>>)
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: scf.for %[[ARG4]] = %c0 to %c2048 step %c32 {
// CHECK-NEXT:   %[[SUBVIEW_0:.*]] = memref.subview %arg0[%0, %[[ARG4]]] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>>
// CHECK-NEXT:   %[[SUBVIEW_1:.*]] = memref.subview %arg1[%[[ARG4]], %1] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>>
// CHECK-NEXT:   scf.for %[[ARG5:.*]] = %[[APPLY_MAP1]] to %c128 step %c128 {
// CHECK-NEXT:     %[[APPLY_MAP2]] = affine.apply #[[MAP2]](%[[THREAD_ID_X]])
// CHECK-NEXT:     scf.for %[[ARG6:.*]] = %[[APPLY_MAP2]] to %c128 step %c128 {
// CHECK-NEXT:       %[[SUBVIEW_2:.*]] = memref.subview %[[SUBVIEW_0]][%[[ARG5]], 0] [64, 32] [1, 1] : memref<128x32xf16, strided<[2048, 1], offset: ?>> to memref<64x32xf16, strided<[2048, 1], offset: ?>>
// CHECK-NEXT:       %[[SUBVIEW_3:.*]] = memref.subview %[[SUBVIEW_1]][0, %[[ARG6]]] [32, 64] [1, 1] : memref<32x128xf16, strided<[5376, 1], offset: ?>> to memref<32x64xf16, strided<[5376, 1], offset: ?>>
// CHECK-NEXT:       %[[SUBVIEW_4:.*]] = memref.subview %subview[%[[ARG5]], %[[ARG6]]] [64, 64] [1, 1] : memref<128x128xf16, strided<[5376, 1], offset: ?>> to memref<64x64xf16, strided<[5376, 1], offset: ?>>
// CHECK-NEXT:       linalg.matmul {{.*}} ins(%[[SUBVIEW_2]], %[[SUBVIEW_3]] : {{.*}}) outs(%[[SUBVIEW_4]] : memref<64x64xf16, strided<[5376, 1], offset: ?>>)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
