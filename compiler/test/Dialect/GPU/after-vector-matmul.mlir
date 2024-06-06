// RUN: byteir-opt -optimize-vector-tranfer  -canonicalize -cse --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<(d0) -> ((d0 floordiv 32) * 64)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func private @Unknown0(%arg0: memref<5376x2048xf16>, %arg1: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    %c40 = arith.constant 40 : index
    %c48 = arith.constant 48 : index
    %c56 = arith.constant 56 : index
    %cst = arith.constant dense<0.000000e+00> : vector<64x64xf16>
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
      %subview_3 = memref.subview %alloca[%4, %5] [64, 64] [1, 1] : memref<128x128xf16, #gpu.address_space<workgroup>> to memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %6 = vector.extract_strided_slice %cst {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %6, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %7 = vector.extract_strided_slice %cst {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %7, %subview_3[%c0, %c8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %8 = vector.extract_strided_slice %cst {offsets = [0, 16], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %8, %subview_3[%c0, %c16] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %9 = vector.extract_strided_slice %cst {offsets = [0, 24], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %9, %subview_3[%c0, %c24] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %10 = vector.extract_strided_slice %cst {offsets = [0, 32], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %10, %subview_3[%c0, %c32] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %11 = vector.extract_strided_slice %cst {offsets = [0, 40], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %11, %subview_3[%c0, %c40] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %12 = vector.extract_strided_slice %cst {offsets = [0, 48], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %12, %subview_3[%c0, %c48] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %13 = vector.extract_strided_slice %cst {offsets = [0, 56], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %13, %subview_3[%c0, %c56] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %14 = vector.extract_strided_slice %cst {offsets = [16, 0], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %14, %subview_3[%c16, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %15 = vector.extract_strided_slice %cst {offsets = [16, 8], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %15, %subview_3[%c16, %c8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %16 = vector.extract_strided_slice %cst {offsets = [16, 16], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %16, %subview_3[%c16, %c16] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %17 = vector.extract_strided_slice %cst {offsets = [16, 24], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %17, %subview_3[%c16, %c24] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %18 = vector.extract_strided_slice %cst {offsets = [16, 32], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %18, %subview_3[%c16, %c32] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %19 = vector.extract_strided_slice %cst {offsets = [16, 40], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %19, %subview_3[%c16, %c40] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %20 = vector.extract_strided_slice %cst {offsets = [16, 48], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %20, %subview_3[%c16, %c48] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %21 = vector.extract_strided_slice %cst {offsets = [16, 56], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %21, %subview_3[%c16, %c56] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %22 = vector.extract_strided_slice %cst {offsets = [32, 0], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %22, %subview_3[%c32, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %23 = vector.extract_strided_slice %cst {offsets = [32, 8], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %23, %subview_3[%c32, %c8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %24 = vector.extract_strided_slice %cst {offsets = [32, 16], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %24, %subview_3[%c32, %c16] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %25 = vector.extract_strided_slice %cst {offsets = [32, 24], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %25, %subview_3[%c32, %c24] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %26 = vector.extract_strided_slice %cst {offsets = [32, 32], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %26, %subview_3[%c32, %c32] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %27 = vector.extract_strided_slice %cst {offsets = [32, 40], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %27, %subview_3[%c32, %c40] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %28 = vector.extract_strided_slice %cst {offsets = [32, 48], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %28, %subview_3[%c32, %c48] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %29 = vector.extract_strided_slice %cst {offsets = [32, 56], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %29, %subview_3[%c32, %c56] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %30 = vector.extract_strided_slice %cst {offsets = [48, 0], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %30, %subview_3[%c48, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %31 = vector.extract_strided_slice %cst {offsets = [48, 8], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %31, %subview_3[%c48, %c8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %32 = vector.extract_strided_slice %cst {offsets = [48, 16], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %32, %subview_3[%c48, %c16] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %33 = vector.extract_strided_slice %cst {offsets = [48, 24], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %33, %subview_3[%c48, %c24] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %34 = vector.extract_strided_slice %cst {offsets = [48, 32], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %34, %subview_3[%c48, %c32] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %35 = vector.extract_strided_slice %cst {offsets = [48, 40], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %35, %subview_3[%c48, %c40] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %36 = vector.extract_strided_slice %cst {offsets = [48, 48], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %36, %subview_3[%c48, %c48] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      %37 = vector.extract_strided_slice %cst {offsets = [48, 56], sizes = [16, 8], strides = [1, 1]} : vector<64x64xf16> to vector<16x8xf16>
      vector.transfer_write %37, %subview_3[%c48, %c56] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      scf.for %arg4 = %c0 to %c2048 step %c32 {
        %subview_4 = memref.subview %arg0[%0, %arg4] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>>
        %subview_5 = memref.subview %arg1[%arg4, %1] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>>
        linalg.copy {__byteir_load_matrix_a__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%subview_4 : memref<128x32xf16, strided<[2048, 1], offset: ?>>) outs(%alloca_2 : memref<128x32xf16, #gpu.address_space<workgroup>>)
        linalg.copy {__byteir_load_matrix_b__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%subview_5 : memref<32x128xf16, strided<[5376, 1], offset: ?>>) outs(%alloca_1 : memref<32x128xf16, #gpu.address_space<workgroup>>)
        %subview_6 = memref.subview %alloca_2[%4, 0] [64, 32] [1, 1] : memref<128x32xf16, #gpu.address_space<workgroup>> to memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
        %subview_7 = memref.subview %alloca_1[0, %5] [32, 64] [1, 1] : memref<32x128xf16, #gpu.address_space<workgroup>> to memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        %38 = vector.transfer_read %subview_6[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %39 = vector.transfer_read %subview_6[%c0, %c16], %cst_0 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %40 = vector.transfer_read %subview_6[%c16, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %41 = vector.transfer_read %subview_6[%c16, %c16], %cst_0 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %42 = vector.transfer_read %subview_6[%c32, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %43 = vector.transfer_read %subview_6[%c32, %c16], %cst_0 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %44 = vector.transfer_read %subview_6[%c48, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %45 = vector.transfer_read %subview_6[%c48, %c16], %cst_0 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %46 = vector.transfer_read %subview_3[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %47 = vector.transfer_read %subview_3[%c0, %c8], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %48 = vector.transfer_read %subview_3[%c0, %c16], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %49 = vector.transfer_read %subview_3[%c0, %c24], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %50 = vector.transfer_read %subview_3[%c0, %c32], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %51 = vector.transfer_read %subview_3[%c0, %c40], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %52 = vector.transfer_read %subview_3[%c0, %c48], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %53 = vector.transfer_read %subview_3[%c0, %c56], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %54 = vector.transfer_read %subview_3[%c16, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %55 = vector.transfer_read %subview_3[%c16, %c8], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %56 = vector.transfer_read %subview_3[%c16, %c16], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %57 = vector.transfer_read %subview_3[%c16, %c24], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %58 = vector.transfer_read %subview_3[%c16, %c32], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %59 = vector.transfer_read %subview_3[%c16, %c40], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %60 = vector.transfer_read %subview_3[%c16, %c48], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %61 = vector.transfer_read %subview_3[%c16, %c56], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %62 = vector.transfer_read %subview_3[%c32, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %63 = vector.transfer_read %subview_3[%c32, %c8], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %64 = vector.transfer_read %subview_3[%c32, %c16], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %65 = vector.transfer_read %subview_3[%c32, %c24], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %66 = vector.transfer_read %subview_3[%c32, %c32], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %67 = vector.transfer_read %subview_3[%c32, %c40], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %68 = vector.transfer_read %subview_3[%c32, %c48], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %69 = vector.transfer_read %subview_3[%c32, %c56], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %70 = vector.transfer_read %subview_3[%c48, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %71 = vector.transfer_read %subview_3[%c48, %c8], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %72 = vector.transfer_read %subview_3[%c48, %c16], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %73 = vector.transfer_read %subview_3[%c48, %c24], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %74 = vector.transfer_read %subview_3[%c48, %c32], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %75 = vector.transfer_read %subview_3[%c48, %c40], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %76 = vector.transfer_read %subview_3[%c48, %c48], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %77 = vector.transfer_read %subview_3[%c48, %c56], %cst_0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x8xf16>
        %78 = vector.transfer_read %subview_7[%c0, %c0], %cst_0 {in_bounds = [true, true], permutation_map = #map3} : memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %79 = vector.transfer_read %subview_7[%c16, %c0], %cst_0 {in_bounds = [true, true], permutation_map = #map3} : memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %80 = vector.transfer_read %subview_7[%c0, %c16], %cst_0 {in_bounds = [true, true], permutation_map = #map3} : memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %81 = vector.transfer_read %subview_7[%c16, %c16], %cst_0 {in_bounds = [true, true], permutation_map = #map3} : memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %82 = vector.transfer_read %subview_7[%c0, %c32], %cst_0 {in_bounds = [true, true], permutation_map = #map3} : memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %83 = vector.transfer_read %subview_7[%c16, %c32], %cst_0 {in_bounds = [true, true], permutation_map = #map3} : memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %84 = vector.transfer_read %subview_7[%c0, %c48], %cst_0 {in_bounds = [true, true], permutation_map = #map3} : memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %85 = vector.transfer_read %subview_7[%c16, %c48], %cst_0 {in_bounds = [true, true], permutation_map = #map3} : memref<32x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<16x16xf16>
        %86 = vector.extract_strided_slice %78 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %87 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %86, %46 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %88 = vector.extract_strided_slice %78 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %89 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %88, %47 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %90 = vector.extract_strided_slice %80 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %91 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %90, %48 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %92 = vector.extract_strided_slice %80 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %93 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %92, %49 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %94 = vector.extract_strided_slice %82 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %95 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %94, %50 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %96 = vector.extract_strided_slice %82 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %97 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %96, %51 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %98 = vector.extract_strided_slice %84 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %99 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %98, %52 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %100 = vector.extract_strided_slice %84 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %101 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %100, %53 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %102 = vector.extract_strided_slice %78 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %103 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %102, %54 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %104 = vector.extract_strided_slice %78 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %105 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %104, %55 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %106 = vector.extract_strided_slice %80 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %107 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %106, %56 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %108 = vector.extract_strided_slice %80 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %109 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %108, %57 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %110 = vector.extract_strided_slice %82 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %111 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %110, %58 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %112 = vector.extract_strided_slice %82 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %113 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %112, %59 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %114 = vector.extract_strided_slice %84 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %115 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %114, %60 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %116 = vector.extract_strided_slice %84 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %117 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %116, %61 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %118 = vector.extract_strided_slice %78 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %119 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %118, %62 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %120 = vector.extract_strided_slice %78 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %121 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %120, %63 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %122 = vector.extract_strided_slice %80 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %123 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %122, %64 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %124 = vector.extract_strided_slice %80 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %125 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %124, %65 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %126 = vector.extract_strided_slice %82 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %127 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %126, %66 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %128 = vector.extract_strided_slice %82 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %129 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %128, %67 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %130 = vector.extract_strided_slice %84 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %131 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %130, %68 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %132 = vector.extract_strided_slice %84 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %133 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %42, %132, %69 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %134 = vector.extract_strided_slice %78 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %135 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %134, %70 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %136 = vector.extract_strided_slice %78 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %137 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %136, %71 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %138 = vector.extract_strided_slice %80 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %139 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %138, %72 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %140 = vector.extract_strided_slice %80 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %141 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %140, %73 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %142 = vector.extract_strided_slice %82 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %143 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %142, %74 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %144 = vector.extract_strided_slice %82 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %145 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %144, %75 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %146 = vector.extract_strided_slice %84 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %147 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %146, %76 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %148 = vector.extract_strided_slice %84 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %149 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %44, %148, %77 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %150 = vector.extract_strided_slice %79 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %151 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %39, %150, %87 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %152 = vector.extract_strided_slice %79 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %153 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %39, %152, %89 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %154 = vector.extract_strided_slice %81 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %155 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %39, %154, %91 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %156 = vector.extract_strided_slice %81 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %157 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %39, %156, %93 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %158 = vector.extract_strided_slice %83 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %159 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %39, %158, %95 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %160 = vector.extract_strided_slice %83 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %161 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %39, %160, %97 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %162 = vector.extract_strided_slice %85 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %163 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %39, %162, %99 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %164 = vector.extract_strided_slice %85 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %165 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %39, %164, %101 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %166 = vector.extract_strided_slice %79 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %167 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %166, %103 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %168 = vector.extract_strided_slice %79 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %169 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %168, %105 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %170 = vector.extract_strided_slice %81 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %171 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %170, %107 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %172 = vector.extract_strided_slice %81 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %173 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %172, %109 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %174 = vector.extract_strided_slice %83 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %175 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %174, %111 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %176 = vector.extract_strided_slice %83 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %177 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %176, %113 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %178 = vector.extract_strided_slice %85 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %179 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %178, %115 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %180 = vector.extract_strided_slice %85 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %181 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %180, %117 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %182 = vector.extract_strided_slice %79 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %183 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %182, %119 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %184 = vector.extract_strided_slice %79 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %185 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %184, %121 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %186 = vector.extract_strided_slice %81 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %187 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %186, %123 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %188 = vector.extract_strided_slice %81 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %189 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %188, %125 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %190 = vector.extract_strided_slice %83 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %191 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %190, %127 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %192 = vector.extract_strided_slice %83 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %193 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %192, %129 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %194 = vector.extract_strided_slice %85 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %195 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %194, %131 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %196 = vector.extract_strided_slice %85 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %197 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %196, %133 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %198 = vector.extract_strided_slice %79 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %199 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %45, %198, %135 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %200 = vector.extract_strided_slice %79 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %201 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %45, %200, %137 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %202 = vector.extract_strided_slice %81 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %203 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %45, %202, %139 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %204 = vector.extract_strided_slice %81 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %205 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %45, %204, %141 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %206 = vector.extract_strided_slice %83 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %207 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %45, %206, %143 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %208 = vector.extract_strided_slice %83 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %209 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %45, %208, %145 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %210 = vector.extract_strided_slice %85 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %211 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %45, %210, %147 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        %212 = vector.extract_strided_slice %85 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %213 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %45, %212, %149 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
        vector.transfer_write %151, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %153, %subview_3[%c0, %c8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %155, %subview_3[%c0, %c16] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %157, %subview_3[%c0, %c24] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %159, %subview_3[%c0, %c32] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %161, %subview_3[%c0, %c40] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %163, %subview_3[%c0, %c48] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %165, %subview_3[%c0, %c56] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %167, %subview_3[%c16, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %169, %subview_3[%c16, %c8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %171, %subview_3[%c16, %c16] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %173, %subview_3[%c16, %c24] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %175, %subview_3[%c16, %c32] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %177, %subview_3[%c16, %c40] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %179, %subview_3[%c16, %c48] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %181, %subview_3[%c16, %c56] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %183, %subview_3[%c32, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %185, %subview_3[%c32, %c8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %187, %subview_3[%c32, %c16] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %189, %subview_3[%c32, %c24] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %191, %subview_3[%c32, %c32] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %193, %subview_3[%c32, %c40] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %195, %subview_3[%c32, %c48] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %197, %subview_3[%c32, %c56] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %199, %subview_3[%c48, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %201, %subview_3[%c48, %c8] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %203, %subview_3[%c48, %c16] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %205, %subview_3[%c48, %c24] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %207, %subview_3[%c48, %c32] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %209, %subview_3[%c48, %c40] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %211, %subview_3[%c48, %c48] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        vector.transfer_write %213, %subview_3[%c48, %c56] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      }
      linalg.copy {__byteir_store_matrix_c__, __internal_linalg_transform__ = "__byteir_copy_related_to_workgroup_memory__"} ins(%alloca : memref<128x128xf16, #gpu.address_space<workgroup>>) outs(%subview : memref<128x128xf16, strided<[5376, 1], offset: ?>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %alloc : memref<5376x5376xf16>
  }
}

