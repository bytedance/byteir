// RUN: byteir-opt -remove-trivial-loops -canonicalize -cse --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<(d0) -> ((d0 floordiv 32) * 64)>
module {
  func.func private @Unknown0(%arg0: memref<5376x2048xf16>, %arg1: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<5376x5376xf16>
    scf.forall (%arg2, %arg3) in (42, 42) {
      %0 = affine.apply #map(%arg2)
      %1 = affine.apply #map(%arg3)
      %subview = memref.subview %alloc[%0, %1] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
      %2 = gpu.thread_id  x
      %3 = gpu.thread_id  y
      %4 = affine.apply #map1()[%3]
      scf.for %arg4 = %4 to %c128 step %c128 {
        %5 = affine.apply #map2(%2)
        scf.for %arg5 = %5 to %c128 step %c128 {
          %subview_0 = memref.subview %subview[%arg4, %arg5] [64, 64] [1, 1] : memref<128x128xf16, strided<[5376, 1], offset: ?>> to memref<64x64xf16, strided<[5376, 1], offset: ?>>
          linalg.fill {__internal_linalg_transform__ = "vectorize"} ins(%cst : f16) outs(%subview_0 : memref<64x64xf16, strided<[5376, 1], offset: ?>>)
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %alloc : memref<5376x5376xf16>
  }
  func.func @main(%arg0: memref<5376x2048xf16>, %arg1: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg0, %arg1) : (memref<5376x2048xf16>, memref<2048x5376xf16>) -> memref<5376x5376xf16>
    return %0 : memref<5376x5376xf16>
  }
}

// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0 * 128)>
// CHECK-NEXT: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: #[[MAP2:.*]] = affine_map<(d0) -> ((d0 floordiv 32) * 64)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func private @Unknown0(%[[ARG0:.*]]: memref<5376x2048xf16>, %[[ARG1:.*]]: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
// CHECK-NEXT:     %[[CST:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:     %[[ALLOC:.*]] = memref.alloc() : memref<5376x5376xf16>
// CHECK-NEXT:     scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (42, 42) {
// CHECK-NEXT:       %[[APPLY_MAP:.*]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK-NEXT:       %[[APPLY_MAP1:.*]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK-NEXT:       %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]][%[[APPLY_MAP]], %[[APPLY_MAP1]]] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
// CHECK-NEXT:       %[[THREAD_ID_X:.*]] = gpu.thread_id x
// CHECK-NEXT:       %[[THREAD_ID_Y:.*]] = gpu.thread_id y
// CHECK-NEXT:       %[[APPLY_MAP1_Y:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_Y]]]
// CHECK-NEXT:       %[[APPLY_MAP2_X:.*]] = affine.apply #[[MAP2]](%[[THREAD_ID_X]])
// CHECK-NEXT:       %[[SUBVIEW_0:.*]] = memref.subview %[[SUBVIEW]][%[[APPLY_MAP1_Y]], %[[APPLY_MAP2_X]]] [64, 64] [1, 1] : memref<128x128xf16, strided<[5376, 1], offset: ?>> to memref<64x64xf16, strided<[5376, 1], offset: ?>>
// CHECK-NEXT:       linalg.fill {__internal_linalg_transform__ = "vectorize"} ins(%[[CST]] : f16) outs(%[[SUBVIEW_0]] : memref<64x64xf16, strided<[5376, 1], offset: ?>>)
// CHECK-NEXT:     } {mapping = [#gpu.block<y>, #gpu.block<x>]}
// CHECK-NEXT:     return %[[ALLOC]] : memref<5376x5376xf16>
// CHECK-NEXT:   }