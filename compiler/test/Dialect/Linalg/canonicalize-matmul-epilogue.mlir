// RUN: byteir-opt %s -canonicalize-matmul-epilogue --canonicalize -cse | FileCheck %s
#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func private @Unknown0(%arg0: tensor<5376x2048xf16>, %arg1: tensor<2048x5376xf16>, %arg2: tensor<5376x5376xf16>) -> tensor<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
    %c32 = arith.constant 32 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<5376x5376xf16>
    %1 = scf.forall (%arg3, %arg4) in (42, 42) shared_outs(%arg5 = %0) -> (tensor<5376x5376xf16>) {
      %2 = affine.apply #map(%arg3)
      %3 = affine.apply #map(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%2, 0] [128, 2048] [1, 1] : tensor<5376x2048xf16> to tensor<128x2048xf16>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %3] [2048, 128] [1, 1] : tensor<2048x5376xf16> to tensor<2048x128xf16>
      %extracted_slice_1 = tensor.extract_slice %0[%2, %3] [128, 128] [1, 1] : tensor<5376x5376xf16> to tensor<128x128xf16>
      %4 = linalg.fill ins(%cst : f16) outs(%extracted_slice_1 : tensor<128x128xf16>) -> tensor<128x128xf16>
      %5 = scf.for %arg6 = %c0 to %c2048 step %c32 iter_args(%arg7 = %4) -> (tensor<128x128xf16>) {
        %extracted_slice_4 = tensor.extract_slice %extracted_slice[0, %arg6] [128, 32] [1, 1] : tensor<128x2048xf16> to tensor<128x32xf16>
        %extracted_slice_5 = tensor.extract_slice %extracted_slice_0[%arg6, 0] [32, 128] [1, 1] : tensor<2048x128xf16> to tensor<32x128xf16>
        %7 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_4, %extracted_slice_5 : tensor<128x32xf16>, tensor<32x128xf16>) outs(%arg7 : tensor<128x128xf16>) attrs =  {__byteir_gpu_tile_gemm_0, __byteir_gpu_tile_gemm_1, __byteir_mma__, __byteir_mma_level__ = "Threadblock", __byteir_target__ = "nv_sm_80"} {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %8 = arith.mulf %in, %in_6 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<128x128xf16>
        scf.yield %7 : tensor<128x128xf16>
      }
      %extracted_slice_2 = tensor.extract_slice %arg2[%2, %3] [128, 128] [1, 1] : tensor<5376x5376xf16> to tensor<128x128xf16>
      %extracted_slice_3 = tensor.extract_slice %arg5[%2, %3] [128, 128] [1, 1] : tensor<5376x5376xf16> to tensor<128x128xf16>
      %6 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%5, %extracted_slice_2 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%extracted_slice_3 : tensor<128x128xf16>) attrs =  {__byteir_epilogue__} {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %7 = arith.addf %in, %in_4 : f16
        linalg.yield %7 : f16
      } -> tensor<128x128xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6 into %arg5[%2, %3] [128, 128] [1, 1] : tensor<128x128xf16> into tensor<5376x5376xf16>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %1 : tensor<5376x5376xf16>
  }
  func.func @main(%arg0: tensor<5376x2048xf16>, %arg1: tensor<2048x5376xf16>, %arg2: tensor<5376x5376xf16>) -> tensor<5376x5376xf16> {
    %0 = call @Unknown0(%arg0, %arg1, %arg2) : (tensor<5376x2048xf16>, tensor<2048x5376xf16>, tensor<5376x5376xf16>) -> tensor<5376x5376xf16>
    return %0 : tensor<5376x5376xf16>
  }
}

// CHECK: scf.forall (%{{.*}}, %{{.*}}) in (42, 42) shared_outs(%[[V0:.*]] = %{{.*}})
// CHECK: %[[V1:.*]] = tensor.extract_slice %[[V0]]
// CHECK: linalg.fill ins(%{{.*}}) outs(%[[V1]] : {{.*}})
// CHECK: %[[MATMUL_RESULT:.*]] = scf.for
// CHECK:           linalg.generic {{.*}} ins(%{{.*}} : tensor<128x128xf16>) outs(%[[MATMUL_RESULT]] : tensor<128x128xf16>) 
// CHECK-NEXT:      ^bb0(%in: f16, %out: f16):
// CHECK-NEXT:       %[[T1:.*]] = arith.addf %out, %in : f16
// CHECK-NEXT:       linalg.yield %[[T1]] : f16
// CHECK-NEXT:     } -> tensor<128x128xf16>
