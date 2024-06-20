// RUN: byteir-opt -optimize-vector-tranfer  -canonicalize -cse --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<(d0) -> ((d0 floordiv 32) * 64)>
#map3 = affine_map<(d0) -> ((d0 floordiv 32) * 64 + 8)>
#map4 = affine_map<(d0) -> ((d0 floordiv 32) * 64 + 16)>
#map5 = affine_map<(d0) -> ((d0 floordiv 32) * 64 + 24)>
#map6 = affine_map<(d0) -> ((d0 floordiv 32) * 64 + 32)>
#map7 = affine_map<(d0) -> ((d0 floordiv 32) * 64 + 40)>
#map8 = affine_map<(d0) -> ((d0 floordiv 32) * 64 + 48)>
#map9 = affine_map<(d0) -> ((d0 floordiv 32) * 64 + 56)>
#map10 = affine_map<()[s0] -> (s0 * 64 + 16)>
#map11 = affine_map<()[s0] -> (s0 * 64 + 32)>
#map12 = affine_map<()[s0] -> (s0 * 64 + 48)>
#map13 = affine_map<(d0, d1) -> (d1, d0)>
#map14 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map15 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map16 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func private @Unknown0(%arg0: memref<5376x2048xf16>, %arg1: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %c16 = arith.constant 16 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<5376x5376xf16>
    scf.forall (%arg2, %arg3) in (42, 42) {
      %alloca = memref.alloca() {__byteir_alloca_accumulator__} : memref<128x128xf16, #gpu.address_space<workgroup>>
      %alloca_1 = memref.alloca() {__byteir_alloca_matrix_b__} : memref<32x128xf16, #gpu.address_space<workgroup>>
      %alloca_2 = memref.alloca() {__byteir_alloca_matrix_a__} : memref<128x32xf16, #gpu.address_space<workgroup>>
      %0 = affine.apply #map(%arg2)
      %1 = affine.apply #map(%arg3)
      %subview = memref.subview %alloc[%0, %1] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
      %2 = gpu.thread_id  x
      %3 = gpu.thread_id  y
      %4 = affine.apply #map1()[%3]
      %5 = affine.apply #map2(%2)
      vector.transfer_write %cst, %alloca[%4, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %6 = affine.apply #map3(%2)
      vector.transfer_write %cst, %alloca[%4, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %7 = affine.apply #map4(%2)
      vector.transfer_write %cst, %alloca[%4, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %8 = affine.apply #map5(%2)
      vector.transfer_write %cst, %alloca[%4, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %9 = affine.apply #map6(%2)
      vector.transfer_write %cst, %alloca[%4, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %10 = affine.apply #map7(%2)
      vector.transfer_write %cst, %alloca[%4, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %11 = affine.apply #map8(%2)
      vector.transfer_write %cst, %alloca[%4, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %12 = affine.apply #map9(%2)
      vector.transfer_write %cst, %alloca[%4, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %13 = affine.apply #map10()[%3]
      vector.transfer_write %cst, %alloca[%13, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%13, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%13, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%13, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%13, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%13, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%13, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%13, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %14 = affine.apply #map11()[%3]
      vector.transfer_write %cst, %alloca[%14, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%14, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%14, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%14, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%14, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%14, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%14, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%14, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      %15 = affine.apply #map12()[%3]
      vector.transfer_write %cst, %alloca[%15, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%15, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%15, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%15, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%15, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%15, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%15, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %cst, %alloca[%15, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      scf.for %arg4 = %c0 to %c2048 step %c32 {
        %subview_3 = memref.subview %arg0[%0, %arg4] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>>
        %subview_4 = memref.subview %arg1[%arg4, %1] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>>
        linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%subview_3 : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%alloca_2 : memref<128x32xf16, #gpu.address_space<workgroup>>)
        linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%subview_4 : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%alloca_1 : memref<32x128xf16, #gpu.address_space<workgroup>>)
        %16 = vector.transfer_read %alloca_2[%4, %c0], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %17 = vector.transfer_read %alloca_2[%4, %c16], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %18 = vector.transfer_read %alloca_2[%13, %c0], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %19 = vector.transfer_read %alloca_2[%13, %c16], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %20 = vector.transfer_read %alloca_2[%14, %c0], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %21 = vector.transfer_read %alloca_2[%14, %c16], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %22 = vector.transfer_read %alloca_2[%15, %c0], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %23 = vector.transfer_read %alloca_2[%15, %c16], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %24 = vector.transfer_read %alloca[%4, %5], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %25 = vector.transfer_read %alloca[%4, %6], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %26 = vector.transfer_read %alloca[%4, %7], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %27 = vector.transfer_read %alloca[%4, %8], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %28 = vector.transfer_read %alloca[%4, %9], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %29 = vector.transfer_read %alloca[%4, %10], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %30 = vector.transfer_read %alloca[%4, %11], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %31 = vector.transfer_read %alloca[%4, %12], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %32 = vector.transfer_read %alloca[%13, %5], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %33 = vector.transfer_read %alloca[%13, %6], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %34 = vector.transfer_read %alloca[%13, %7], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %35 = vector.transfer_read %alloca[%13, %8], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %36 = vector.transfer_read %alloca[%13, %9], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %37 = vector.transfer_read %alloca[%13, %10], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %38 = vector.transfer_read %alloca[%13, %11], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %39 = vector.transfer_read %alloca[%13, %12], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %40 = vector.transfer_read %alloca[%14, %5], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %41 = vector.transfer_read %alloca[%14, %6], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %42 = vector.transfer_read %alloca[%14, %7], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %43 = vector.transfer_read %alloca[%14, %8], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %44 = vector.transfer_read %alloca[%14, %9], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %45 = vector.transfer_read %alloca[%14, %10], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %46 = vector.transfer_read %alloca[%14, %11], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %47 = vector.transfer_read %alloca[%14, %12], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %48 = vector.transfer_read %alloca[%15, %5], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %49 = vector.transfer_read %alloca[%15, %6], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %50 = vector.transfer_read %alloca[%15, %7], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %51 = vector.transfer_read %alloca[%15, %8], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %52 = vector.transfer_read %alloca[%15, %9], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %53 = vector.transfer_read %alloca[%15, %10], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %54 = vector.transfer_read %alloca[%15, %11], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %55 = vector.transfer_read %alloca[%15, %12], %cst_0 {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %56 = vector.transfer_read %alloca_1[%c0, %5], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %57 = vector.transfer_read %alloca_1[%c16, %5], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %58 = vector.transfer_read %alloca_1[%c0, %7], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %59 = vector.transfer_read %alloca_1[%c16, %7], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %60 = vector.transfer_read %alloca_1[%c0, %9], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %61 = vector.transfer_read %alloca_1[%c16, %9], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %62 = vector.transfer_read %alloca_1[%c0, %11], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %63 = vector.transfer_read %alloca_1[%c16, %11], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %64 = vector.extract_strided_slice %56 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %65 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %64, %24 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %66 = vector.extract_strided_slice %56 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %67 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %66, %25 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %68 = vector.extract_strided_slice %58 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %69 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %68, %26 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %70 = vector.extract_strided_slice %58 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %71 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %70, %27 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %72 = vector.extract_strided_slice %60 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %73 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %72, %28 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %74 = vector.extract_strided_slice %60 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %75 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %74, %29 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %76 = vector.extract_strided_slice %62 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %77 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %76, %30 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %78 = vector.extract_strided_slice %62 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %79 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %78, %31 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %80 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %64, %32 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %81 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %66, %33 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %82 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %68, %34 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %83 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %70, %35 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %84 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %72, %36 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %85 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %74, %37 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %86 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %76, %38 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %87 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %78, %39 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %88 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %64, %40 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %89 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %66, %41 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %90 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %68, %42 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %91 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %70, %43 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %92 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %72, %44 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %93 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %74, %45 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %94 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %76, %46 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %95 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %78, %47 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %96 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %64, %48 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %97 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %66, %49 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %98 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %68, %50 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %99 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %70, %51 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %100 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %72, %52 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %101 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %74, %53 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %102 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %76, %54 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %103 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %78, %55 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %104 = vector.extract_strided_slice %57 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %105 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %104, %65 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %106 = vector.extract_strided_slice %57 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %107 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %106, %67 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %108 = vector.extract_strided_slice %59 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %109 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %108, %69 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %110 = vector.extract_strided_slice %59 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %111 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %110, %71 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %112 = vector.extract_strided_slice %61 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %113 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %112, %73 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %114 = vector.extract_strided_slice %61 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %115 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %114, %75 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %116 = vector.extract_strided_slice %63 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %117 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %116, %77 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %118 = vector.extract_strided_slice %63 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %119 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %118, %79 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %120 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %104, %80 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %121 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %106, %81 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %122 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %108, %82 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %123 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %110, %83 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %124 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %112, %84 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %125 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %114, %85 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %126 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %116, %86 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %127 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %118, %87 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %128 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %104, %88 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %129 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %106, %89 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %130 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %108, %90 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %131 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %110, %91 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %132 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %112, %92 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %133 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %114, %93 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %134 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %116, %94 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %135 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %118, %95 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %136 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %104, %96 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %137 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %106, %97 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %138 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %108, %98 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %139 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %110, %99 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %140 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %112, %100 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %141 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %114, %101 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %142 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %116, %102 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %143 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %118, %103 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        vector.transfer_write %105, %alloca[%4, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %107, %alloca[%4, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %109, %alloca[%4, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %111, %alloca[%4, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %113, %alloca[%4, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %115, %alloca[%4, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %117, %alloca[%4, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %119, %alloca[%4, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %120, %alloca[%13, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %121, %alloca[%13, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %122, %alloca[%13, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %123, %alloca[%13, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %124, %alloca[%13, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %125, %alloca[%13, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %126, %alloca[%13, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %127, %alloca[%13, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %128, %alloca[%14, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %129, %alloca[%14, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %130, %alloca[%14, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %131, %alloca[%14, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %132, %alloca[%14, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %133, %alloca[%14, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %134, %alloca[%14, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %135, %alloca[%14, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %136, %alloca[%15, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %137, %alloca[%15, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %138, %alloca[%15, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %139, %alloca[%15, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %140, %alloca[%15, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %141, %alloca[%15, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %142, %alloca[%15, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %143, %alloca[%15, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      }
      linalg.copy {__byteir_store_matrix_c__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%alloca : memref<128x128xf16, #gpu.address_space<workgroup>>) outs(%subview : memref<128x128xf16, strided<[5376, 1], offset: ?>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %alloc : memref<5376x5376xf16>
  }
}

// CHECK:       scf.for 
// CHECK-COUNT-8:   {{.*}} = vector.transfer_read %alloca_2{{.*}}
// CHECK-COUNT-8:   {{.*}} = vector.transfer_read %alloca_1{{.*}}
// CHECK-COUNT-64:  {{.*}} = vector.contract {{.*}}
// CHECK:       scf.yield
// CHECK-NEXT: }
// CHECK-COUNT-32: vector.transfer_write {{.*}}, %alloca{{.*}}