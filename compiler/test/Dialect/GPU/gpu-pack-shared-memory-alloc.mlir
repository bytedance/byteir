//RUN: byteir-opt --gpu-pack-shared-memory-alloc %s | FileCheck %s

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
      %alloc_0 = memref.alloc() {__byteir_alloc_accumulator__} : memref<128x128xf16, #gpu.address_space<workgroup>>
      %alloc_1 = memref.alloc() {__byteir_alloc_matrix_b__} : memref<32x128xf16, #gpu.address_space<workgroup>>
      %alloc_2 = memref.alloc() {__byteir_alloc_matrix_a__} : memref<128x32xf16, #gpu.address_space<workgroup>>
      %0 = affine.apply #map(%arg2)
      %1 = affine.apply #map(%arg3)
      %subview = memref.subview %alloc[%0, %1] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
      %2 = gpu.thread_id  x
      %3 = gpu.thread_id  y
      %4 = affine.apply #map1()[%3]
      %5 = affine.apply #map2(%2)
      %6 = affine.apply #map3(%2)
      %7 = affine.apply #map4(%2)
      %8 = affine.apply #map5(%2)
      %9 = affine.apply #map6(%2)
      %10 = affine.apply #map7(%2)
      %11 = affine.apply #map8(%2)
      %12 = affine.apply #map9(%2)
      %13 = affine.apply #map10()[%3]
      %14 = affine.apply #map11()[%3]
      %15 = affine.apply #map12()[%3]
      %16:32 = scf.for %arg4 = %c0 to %c2048 step %c32 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst) -> (vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>) {
        %subview_3 = memref.subview %arg0[%0, %arg4] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>>
        %subview_4 = memref.subview %arg1[%arg4, %1] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>>
        linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%subview_3 : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%alloc_2 : memref<128x32xf16, #gpu.address_space<workgroup>>)
        linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%subview_4 : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%alloc_1 : memref<32x128xf16, #gpu.address_space<workgroup>>)
        %17 = vector.transfer_read %alloc_2[%4, %c0], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %18 = vector.transfer_read %alloc_2[%4, %c16], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %19 = vector.transfer_read %alloc_2[%13, %c0], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %20 = vector.transfer_read %alloc_2[%13, %c16], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %21 = vector.transfer_read %alloc_2[%14, %c0], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %22 = vector.transfer_read %alloc_2[%14, %c16], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %23 = vector.transfer_read %alloc_2[%15, %c0], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %24 = vector.transfer_read %alloc_2[%15, %c16], %cst_0 {in_bounds = [true, true]} : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %25 = vector.transfer_read %alloc_1[%c0, %5], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %26 = vector.transfer_read %alloc_1[%c16, %5], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %27 = vector.transfer_read %alloc_1[%c0, %7], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %28 = vector.transfer_read %alloc_1[%c16, %7], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %29 = vector.transfer_read %alloc_1[%c0, %9], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %30 = vector.transfer_read %alloc_1[%c16, %9], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %31 = vector.transfer_read %alloc_1[%c0, %11], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %32 = vector.transfer_read %alloc_1[%c16, %11], %cst_0 {in_bounds = [true, true], permutation_map = #map13} : memref<32x128xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %33 = vector.extract_strided_slice %25 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %34 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %33, %arg5 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %35 = vector.extract_strided_slice %25 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %36 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %35, %arg6 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %37 = vector.extract_strided_slice %27 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %38 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %37, %arg7 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %39 = vector.extract_strided_slice %27 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %40 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %39, %arg8 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %41 = vector.extract_strided_slice %29 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %42 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %41, %arg9 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %43 = vector.extract_strided_slice %29 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %44 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %43, %arg10 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %45 = vector.extract_strided_slice %31 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %46 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %45, %arg11 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %47 = vector.extract_strided_slice %31 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %48 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %47, %arg12 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %49 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %33, %arg13 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %50 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %35, %arg14 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %51 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %37, %arg15 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %52 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %39, %arg16 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %53 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %41, %arg17 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %54 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %43, %arg18 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %55 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %45, %arg19 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %56 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %47, %arg20 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %57 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %33, %arg21 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %58 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %35, %arg22 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %59 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %37, %arg23 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %60 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %39, %arg24 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %61 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %41, %arg25 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %62 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %43, %arg26 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %63 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %45, %arg27 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %64 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %47, %arg28 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %65 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %33, %arg29 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %66 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %35, %arg30 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %67 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %37, %arg31 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %68 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %39, %arg32 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %69 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %41, %arg33 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %70 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %43, %arg34 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %71 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %45, %arg35 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %72 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %47, %arg36 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %73 = vector.extract_strided_slice %26 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %74 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %73, %34 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %75 = vector.extract_strided_slice %26 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %76 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %75, %36 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %77 = vector.extract_strided_slice %28 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %78 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %77, %38 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %79 = vector.extract_strided_slice %28 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %80 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %79, %40 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %81 = vector.extract_strided_slice %30 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %82 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %81, %42 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %83 = vector.extract_strided_slice %30 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %84 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %83, %44 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %85 = vector.extract_strided_slice %32 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %86 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %85, %46 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %87 = vector.extract_strided_slice %32 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %88 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %87, %48 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %89 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %73, %49 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %90 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %75, %50 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %91 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %77, %51 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %92 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %79, %52 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %93 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %81, %53 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %94 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %83, %54 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %95 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %85, %55 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %96 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %87, %56 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %97 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %73, %57 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %98 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %75, %58 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %99 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %77, %59 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %100 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %79, %60 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %101 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %81, %61 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %102 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %83, %62 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %103 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %85, %63 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %104 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %87, %64 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %105 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %73, %65 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %106 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %75, %66 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %107 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %77, %67 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %108 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %79, %68 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %109 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %81, %69 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %110 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %83, %70 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %111 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %85, %71 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %112 = vector.contract {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %87, %72 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        scf.yield %74, %76, %78, %80, %82, %84, %86, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112 : vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>, vector<16x8xf16>
      }
      vector.transfer_write %16#31, %alloc_0[%15, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#30, %alloc_0[%15, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#29, %alloc_0[%15, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#28, %alloc_0[%15, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#27, %alloc_0[%15, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#26, %alloc_0[%15, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#25, %alloc_0[%15, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#24, %alloc_0[%15, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#23, %alloc_0[%14, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#22, %alloc_0[%14, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#21, %alloc_0[%14, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#20, %alloc_0[%14, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#19, %alloc_0[%14, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#18, %alloc_0[%14, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#17, %alloc_0[%14, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#16, %alloc_0[%14, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#15, %alloc_0[%13, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#14, %alloc_0[%13, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#13, %alloc_0[%13, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#12, %alloc_0[%13, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#11, %alloc_0[%13, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#10, %alloc_0[%13, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#9, %alloc_0[%13, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#8, %alloc_0[%13, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#7, %alloc_0[%4, %12] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#6, %alloc_0[%4, %11] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#5, %alloc_0[%4, %10] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#4, %alloc_0[%4, %9] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#3, %alloc_0[%4, %8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#2, %alloc_0[%4, %7] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#1, %alloc_0[%4, %6] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %16#0, %alloc_0[%4, %5] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
      linalg.copy {__byteir_store_matrix_c__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%alloc_0 : memref<128x128xf16, #gpu.address_space<workgroup>>) outs(%subview : memref<128x128xf16, strided<[5376, 1], offset: ?>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %alloc : memref<5376x5376xf16>
  }
}

// CHECK: %[[ALLOC_PACK:.*]] = memref.alloc() : memref<32768xi8, #gpu.address_space<workgroup>>
// CHECK: %{{.*}} = memref.view %[[ALLOC_PACK]][%c0{{.*}}][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<32x128xf16, #gpu.address_space<workgroup>>
// CHECK: %{{.*}} = memref.view %[[ALLOC_PACK]][%c8192{{.*}}][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<128x32xf16, #gpu.address_space<workgroup>>
// CHECK: %{{.*}} = memref.view %[[ALLOC_PACK]][%c0{{.*}}][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<128x128xf16, #gpu.address_space<workgroup>>