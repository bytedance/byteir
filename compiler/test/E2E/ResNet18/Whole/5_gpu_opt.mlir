// RUN: byteir-opt %s -gpu-opt | FileCheck %s

// CHECK-LABEL: func.func @main

module @IrToMhlo.2452 {
  func.func private @Unknown0(%arg0: memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c602112 = arith.constant 602112 : index
    %c1 = arith.constant 1 : index
    %c224 = arith.constant 224 : index
    %c-1 = arith.constant -1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<4x3x224x224xf16>
    scf.for %arg1 = %c0 to %c602112 step %c1 {
      %0 = arith.remsi %arg1, %c224 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c224 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c224 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c224 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c224 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c224 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c3 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c3 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c3 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x3x224x224xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x3x224x224xf16>
    }
    return %alloc : memref<4x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c9408 = arith.constant 9408 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf16>
    scf.for %arg1 = %c0 to %c9408 step %c1 {
      %0 = arith.remsi %arg1, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c3 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c3 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c3 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x3x7x7xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x3x7x7xf16>
    }
    return %alloc : memref<64x3x7x7xf16>
  }
  func.func private @BatchNormTrainingOp2(%arg0: memref<4x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x112x112xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x64x112x112xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x112x112xf32>
    %alloc_1 = memref.alloc() : memref<64xf32>
    %alloc_2 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x112x112xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x64x112x112xf32>, memref<4x64x112x112xf16>) -> ()
    return %alloc_3 : memref<4x64x112x112xf16>
  }
  func.func private @Unknown3(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x64x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x64x3x3xf16>
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x64x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x64x3x3xf16>
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown5(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x64x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x64x3x3xf16>
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown6(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x64x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x64x3x3xf16>
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown7(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf16>
    scf.for %arg1 = %c0 to %c8192 step %c1 {
      %0 = arith.remsi %arg1, %c64 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c64 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c64 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3, %c0, %c0] : memref<128x64x1x1xf32>
      %11 = arith.truncf %10 : f32 to f16
      memref.store %11, %alloc[%9, %3, %c0, %c0] : memref<128x64x1x1xf16>
    }
    return %alloc : memref<128x64x1x1xf16>
  }
  func.func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c73728 = arith.constant 73728 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf16>
    scf.for %arg1 = %c0 to %c73728 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<128x64x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<128x64x3x3xf16>
    }
    return %alloc : memref<128x64x3x3xf16>
  }
  func.func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c147456 = arith.constant 147456 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    scf.for %arg1 = %c0 to %c147456 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<128x128x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<128x128x3x3xf16>
    }
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown10(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c147456 = arith.constant 147456 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    scf.for %arg1 = %c0 to %c147456 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<128x128x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<128x128x3x3xf16>
    }
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown11(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c147456 = arith.constant 147456 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    scf.for %arg1 = %c0 to %c147456 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<128x128x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<128x128x3x3xf16>
    }
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown12(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf16>
    scf.for %arg1 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg1, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3, %c0, %c0] : memref<256x128x1x1xf32>
      %11 = arith.truncf %10 : f32 to f16
      memref.store %11, %alloc[%9, %3, %c0, %c0] : memref<256x128x1x1xf16>
    }
    return %alloc : memref<256x128x1x1xf16>
  }
  func.func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c294912 = arith.constant 294912 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf16>
    scf.for %arg1 = %c0 to %c294912 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<256x128x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<256x128x3x3xf16>
    }
    return %alloc : memref<256x128x3x3xf16>
  }
  func.func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c589824 = arith.constant 589824 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    scf.for %arg1 = %c0 to %c589824 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<256x256x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<256x256x3x3xf16>
    }
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown15(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c589824 = arith.constant 589824 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    scf.for %arg1 = %c0 to %c589824 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<256x256x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<256x256x3x3xf16>
    }
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown16(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c589824 = arith.constant 589824 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    scf.for %arg1 = %c0 to %c589824 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<256x256x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<256x256x3x3xf16>
    }
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown17(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c131072 = arith.constant 131072 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf16>
    scf.for %arg1 = %c0 to %c131072 step %c1 {
      %0 = arith.remsi %arg1, %c256 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c256 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c256 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3, %c0, %c0] : memref<512x256x1x1xf32>
      %11 = arith.truncf %10 : f32 to f16
      memref.store %11, %alloc[%9, %3, %c0, %c0] : memref<512x256x1x1xf16>
    }
    return %alloc : memref<512x256x1x1xf16>
  }
  func.func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1179648 = arith.constant 1179648 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf16>
    scf.for %arg1 = %c0 to %c1179648 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<512x256x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<512x256x3x3xf16>
    }
    return %alloc : memref<512x256x3x3xf16>
  }
  func.func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c2359296 = arith.constant 2359296 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    scf.for %arg1 = %c0 to %c2359296 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<512x512x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<512x512x3x3xf16>
    }
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown20(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c2359296 = arith.constant 2359296 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    scf.for %arg1 = %c0 to %c2359296 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<512x512x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<512x512x3x3xf16>
    }
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown21(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c2359296 = arith.constant 2359296 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    scf.for %arg1 = %c0 to %c2359296 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<512x512x3x3xf32>
      %31 = arith.truncf %30 : f32 to f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<512x512x3x3xf16>
    }
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown22(%arg0: memref<4x1000xf32>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -2.500000e-01 : f32
    %c0 = arith.constant 0 : index
    %c4000 = arith.constant 4000 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    scf.for %arg1 = %c0 to %c4000 step %c1 {
      %0 = arith.remsi %arg1, %c1000 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c1000 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c1000 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3] : memref<4x1000xf32>
      %11 = arith.mulf %10, %cst : f32
      %12 = arith.truncf %11 : f32 to f16
      memref.store %12, %alloc[%9, %3] : memref<4x1000xf16>
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown23(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512000 = arith.constant 512000 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<1000x512xf16>
    scf.for %arg1 = %c0 to %c512000 step %c1 {
      %0 = arith.remsi %arg1, %c512 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c512 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c512 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3] : memref<1000x512xf32>
      %11 = arith.truncf %10 : f32 to f16
      memref.store %11, %alloc[%9, %3] : memref<1000x512xf16>
    }
    return %alloc : memref<1000x512xf16>
  }
  func.func private @Unknown24(%arg0: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c3211264 = arith.constant 3211264 : index
    %c1 = arith.constant 1 : index
    %c112 = arith.constant 112 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    %alloc_0 = memref.alloc() : memref<4x64x112x112xi1>
    scf.for %arg1 = %c0 to %c3211264 step %c1 {
      %0 = arith.remsi %arg1, %c112 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c112 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c112 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c112 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c112 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c112 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x112x112xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x64x112x112xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x64x112x112xi1>
    }
    return %alloc, %alloc_0 : memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
  }
  func.func private @BatchNormTrainingOp25(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32>
    %alloc_1 = memref.alloc() : memref<64xf32>
    %alloc_2 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %alloc_3 : memref<4x64x56x56xf16>
  }
  func.func private @Unknown26(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    scf.for %arg1 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg1, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x64x56x56xi1>
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp27(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32>
    %alloc_1 = memref.alloc() : memref<64xf32>
    %alloc_2 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %alloc_3 : memref<4x64x56x56xf16>
  }
  func.func private @Unknown28(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    scf.for %arg2 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg2, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %32 = arith.addf %30, %31 : f16
      %33 = arith.maxf %32, %cst : f16
      %34 = arith.cmpf ogt, %33, %cst : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      memref.store %34, %alloc_0[%29, %23, %13, %3] : memref<4x64x56x56xi1>
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp29(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32>
    %alloc_1 = memref.alloc() : memref<64xf32>
    %alloc_2 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %alloc_3 : memref<4x64x56x56xf16>
  }
  func.func private @Unknown30(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    scf.for %arg1 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg1, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x64x56x56xi1>
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp31(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32>
    %alloc_1 = memref.alloc() : memref<64xf32>
    %alloc_2 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %alloc_3 : memref<4x64x56x56xf16>
  }
  func.func private @Unknown32(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    scf.for %arg2 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg2, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %32 = arith.addf %30, %31 : f16
      %33 = arith.maxf %32, %cst : f16
      %34 = arith.cmpf ogt, %33, %cst : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      memref.store %34, %alloc_0[%29, %23, %13, %3] : memref<4x64x56x56xi1>
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @BatchNormTrainingOp33(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_1 = memref.alloc() : memref<128xf32>
    %alloc_2 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_3 : memref<4x128x28x28xf16>
  }
  func.func private @BatchNormTrainingOp34(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_1 = memref.alloc() : memref<128xf32>
    %alloc_2 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_3 : memref<4x128x28x28xf16>
  }
  func.func private @Unknown35(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c401408 = arith.constant 401408 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    scf.for %arg1 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg1, %c28 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c28 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c28 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c28 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c28 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c28 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x128x28x28xi1>
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp36(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_1 = memref.alloc() : memref<128xf32>
    %alloc_2 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_3 : memref<4x128x28x28xf16>
  }
  func.func private @Unknown37(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c401408 = arith.constant 401408 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    scf.for %arg2 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg2, %c28 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c28 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c28 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c28 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c28 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c28 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %32 = arith.addf %30, %31 : f16
      %33 = arith.maxf %32, %cst : f16
      %34 = arith.cmpf ogt, %33, %cst : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      memref.store %34, %alloc_0[%29, %23, %13, %3] : memref<4x128x28x28xi1>
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp38(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_1 = memref.alloc() : memref<128xf32>
    %alloc_2 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_3 : memref<4x128x28x28xf16>
  }
  func.func private @Unknown39(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c401408 = arith.constant 401408 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    scf.for %arg1 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg1, %c28 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c28 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c28 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c28 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c28 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c28 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x128x28x28xi1>
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp40(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_1 = memref.alloc() : memref<128xf32>
    %alloc_2 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_3 : memref<4x128x28x28xf16>
  }
  func.func private @Unknown41(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c401408 = arith.constant 401408 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    scf.for %arg2 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg2, %c28 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c28 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c28 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c28 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c28 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c28 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %32 = arith.addf %30, %31 : f16
      %33 = arith.maxf %32, %cst : f16
      %34 = arith.cmpf ogt, %33, %cst : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      memref.store %34, %alloc_0[%29, %23, %13, %3] : memref<4x128x28x28xi1>
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @BatchNormTrainingOp42(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_1 = memref.alloc() : memref<256xf32>
    %alloc_2 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_3 : memref<4x256x14x14xf16>
  }
  func.func private @BatchNormTrainingOp43(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_1 = memref.alloc() : memref<256xf32>
    %alloc_2 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_3 : memref<4x256x14x14xf16>
  }
  func.func private @Unknown44(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    scf.for %arg1 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg1, %c14 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c14 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c14 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c14 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c14 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c14 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x256x14x14xi1>
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp45(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_1 = memref.alloc() : memref<256xf32>
    %alloc_2 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_3 : memref<4x256x14x14xf16>
  }
  func.func private @Unknown46(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    scf.for %arg2 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg2, %c14 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c14 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c14 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c14 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c14 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c14 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %32 = arith.addf %30, %31 : f16
      %33 = arith.maxf %32, %cst : f16
      %34 = arith.cmpf ogt, %33, %cst : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      memref.store %34, %alloc_0[%29, %23, %13, %3] : memref<4x256x14x14xi1>
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp47(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_1 = memref.alloc() : memref<256xf32>
    %alloc_2 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_3 : memref<4x256x14x14xf16>
  }
  func.func private @Unknown48(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    scf.for %arg1 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg1, %c14 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c14 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c14 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c14 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c14 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c14 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x256x14x14xi1>
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp49(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_1 = memref.alloc() : memref<256xf32>
    %alloc_2 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_3 : memref<4x256x14x14xf16>
  }
  func.func private @Unknown50(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    scf.for %arg2 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg2, %c14 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c14 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c14 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c14 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c14 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c14 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %32 = arith.addf %30, %31 : f16
      %33 = arith.maxf %32, %cst : f16
      %34 = arith.cmpf ogt, %33, %cst : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      memref.store %34, %alloc_0[%29, %23, %13, %3] : memref<4x256x14x14xi1>
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @BatchNormTrainingOp51(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_1 = memref.alloc() : memref<512xf32>
    %alloc_2 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_3 : memref<4x512x7x7xf16>
  }
  func.func private @BatchNormTrainingOp52(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_1 = memref.alloc() : memref<512xf32>
    %alloc_2 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_3 : memref<4x512x7x7xf16>
  }
  func.func private @Unknown53(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    scf.for %arg1 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg1, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x512x7x7xi1>
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp54(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_1 = memref.alloc() : memref<512xf32>
    %alloc_2 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_3 : memref<4x512x7x7xf16>
  }
  func.func private @Unknown55(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    scf.for %arg2 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg2, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %32 = arith.addf %30, %31 : f16
      %33 = arith.maxf %32, %cst : f16
      %34 = arith.cmpf ogt, %33, %cst : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      memref.store %34, %alloc_0[%29, %23, %13, %3] : memref<4x512x7x7xi1>
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp56(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_1 = memref.alloc() : memref<512xf32>
    %alloc_2 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_3 : memref<4x512x7x7xf16>
  }
  func.func private @Unknown57(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    scf.for %arg1 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg1, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %31 = arith.maxf %30, %cst : f16
      %32 = arith.cmpf ogt, %31, %cst : f16
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      memref.store %32, %alloc_0[%29, %23, %13, %3] : memref<4x512x7x7xi1>
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @BatchNormTrainingOp58(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_1 = memref.alloc() : memref<512xf32>
    %alloc_2 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_0, %alloc_3) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_3 : memref<4x512x7x7xf16>
  }
  func.func private @Unknown59(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    scf.for %arg2 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg2, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %32 = arith.addf %30, %31 : f16
      %33 = arith.maxf %32, %cst : f16
      %34 = arith.cmpf ogt, %33, %cst : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      memref.store %34, %alloc_0[%29, %23, %13, %3] : memref<4x512x7x7xi1>
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown60(%arg0: memref<4x512xf16>) -> memref<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<4x512xf16>
    scf.for %arg1 = %c0 to %c2048 step %c1 {
      %0 = arith.remsi %arg1, %c512 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c512 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c512 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3] : memref<4x512xf16>
      %11 = arith.mulf %10, %cst : f16
      memref.store %11, %alloc[%9, %3] : memref<4x512xf16>
    }
    return %alloc : memref<4x512xf16>
  }
  func.func private @Unknown61(%arg0: memref<1000xf32>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4000 = arith.constant 4000 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    scf.for %arg2 = %c0 to %c4000 step %c1 {
      %0 = arith.remsi %arg2, %c1000 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c1000 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c1000 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg1[%9, %3] : memref<4x1000xf16>
      %11 = memref.load %arg0[%3] : memref<1000xf32>
      %12 = arith.truncf %11 : f32 to f16
      %13 = arith.addf %10, %12 : f16
      memref.store %13, %alloc[%9, %3] : memref<4x1000xf16>
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown62(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4000 = arith.constant 4000 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    %alloc_0 = memref.alloc() : memref<4x1000xf16>
    scf.for %arg2 = %c0 to %c4000 step %c1 {
      %0 = arith.remsi %arg2, %c1000 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c1000 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c1000 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg1[%9, %3] : memref<4x1000xf16>
      %11 = memref.load %arg0[%9] : memref<4xf16>
      %12 = arith.subf %10, %11 : f16
      %13 = math.exp %12 : f16
      memref.store %12, %alloc[%9, %3] : memref<4x1000xf16>
      memref.store %13, %alloc_0[%9, %3] : memref<4x1000xf16>
    }
    return %alloc, %alloc_0 : memref<4x1000xf16>, memref<4x1000xf16>
  }
  func.func private @Unknown63(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>, %arg4: memref<4x1000xf32>) -> (memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>) attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c4000 = arith.constant 4000 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    %alloc_0 = memref.alloc() : memref<4x1000xf32>
    %alloc_1 = memref.alloc() : memref<4x1000xf32>
    scf.for %arg5 = %c0 to %c4000 step %c1 {
      %0 = arith.remsi %arg5, %c1000 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c1000 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg5, %c0 : index
      %5 = arith.subi %c-1, %arg5 : index
      %6 = arith.select %4, %5, %arg5 : index
      %7 = arith.divsi %6, %c1000 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg3[%9, %3] : memref<4x1000xf16>
      %11 = memref.load %arg1[%9, %3] : memref<4x1000xf16>
      %12 = memref.load %arg0[%9] : memref<4xf16>
      %13 = memref.load %arg2[%9] : memref<4xf16>
      %14 = memref.load %arg4[%9, %3] : memref<4x1000xf32>
      %15 = math.log %12 : f16
      %16 = arith.subf %11, %15 : f16
      %17 = math.exp %16 : f16
      %18 = arith.mulf %17, %13 : f16
      %19 = arith.subf %10, %18 : f16
      %20 = arith.extf %16 : f16 to f32
      %21 = arith.mulf %20, %14 : f32
      %22 = arith.extf %19 : f16 to f32
      memref.store %19, %alloc[%9, %3] : memref<4x1000xf16>
      memref.store %21, %alloc_0[%9, %3] : memref<4x1000xf32>
      memref.store %22, %alloc_1[%9, %3] : memref<4x1000xf32>
    }
    return %alloc, %alloc_0, %alloc_1 : memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>
  }
  func.func private @Unknown64(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.900000e+01 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg2 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg2, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg1[%29, %23, %13, %3] : memref<4x512x7x7xi1>
      %31 = memref.load %arg0[%29, %23] : memref<4x512xf16>
      %32 = arith.divf %31, %cst : f16
      %33 = arith.select %30, %32, %cst_0 : f16
      memref.store %33, %alloc[%29, %23, %13, %3] : memref<4x512x7x7xf16>
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp65(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp66(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %alloc_1 : memref<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp67(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %alloc_0 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown68(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg2 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg2, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x512x7x7xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x512x7x7xf16>
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp69(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp70(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %alloc_1 : memref<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp71(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %alloc_0 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown72(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg3 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg3, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg2[%29, %23, %13, %3] : memref<4x512x7x7xi1>
      %31 = memref.load %arg0[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %32 = memref.load %arg1[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %33 = arith.addf %31, %32 : f16
      %34 = arith.select %30, %33, %cst : f16
      memref.store %34, %alloc[%29, %23, %13, %3] : memref<4x512x7x7xf16>
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp73(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp74(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %alloc_1 : memref<4x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp75(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %alloc_0 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown76(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg2 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg2, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x512x7x7xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x512x7x7xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x512x7x7xf16>
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @BatchNormGradOp77(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp78(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<3x3x256x512xf16>, memref<4x256x14x14xf16>) -> ()
    return %alloc_1 : memref<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp79(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x256x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x256x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %alloc_0 : memref<512x256x3x3xf16>
  }
  func.func private @BatchNormGradOp80(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp81(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<1x1x256x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<1x1x256x512xf16>, memref<4x256x14x14xf16>) -> ()
    return %alloc_0 : memref<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp82(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<1x1x256x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x256x1x1xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %alloc_0 : memref<512x256x1x1xf16>
  }
  func.func private @Unknown83(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    scf.for %arg3 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg3, %c14 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c14 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c14 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c14 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c14 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c14 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg2[%29, %23, %13, %3] : memref<4x256x14x14xi1>
      %31 = memref.load %arg0[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %32 = memref.load %arg1[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %33 = arith.addf %31, %32 : f16
      %34 = arith.select %30, %33, %cst : f16
      memref.store %34, %alloc[%29, %23, %13, %3] : memref<4x256x14x14xf16>
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp84(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp85(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %alloc_1 : memref<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp86(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %alloc_0 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown87(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    scf.for %arg2 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg2, %c14 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c14 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c14 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c14 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c14 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c14 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x256x14x14xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x256x14x14xf16>
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp88(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp89(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %alloc_1 : memref<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp90(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %alloc_0 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown91(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    scf.for %arg3 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg3, %c14 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c14 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c14 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c14 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c14 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c14 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg2[%29, %23, %13, %3] : memref<4x256x14x14xi1>
      %31 = memref.load %arg0[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %32 = memref.load %arg1[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %33 = arith.addf %31, %32 : f16
      %34 = arith.select %30, %33, %cst : f16
      memref.store %34, %alloc[%29, %23, %13, %3] : memref<4x256x14x14xf16>
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp92(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp93(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %alloc_1 : memref<4x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp94(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %alloc_0 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown95(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    scf.for %arg2 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg2, %c14 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c14 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c14 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c14 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c14 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c14 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x256x14x14xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x256x14x14xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x256x14x14xf16>
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @BatchNormGradOp96(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp97(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<3x3x128x256xf16>, memref<4x128x28x28xf16>) -> ()
    return %alloc_1 : memref<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp98(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x128x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x128x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %alloc_0 : memref<256x128x3x3xf16>
  }
  func.func private @BatchNormGradOp99(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp100(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<1x1x128x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<1x1x128x256xf16>, memref<4x128x28x28xf16>) -> ()
    return %alloc_0 : memref<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp101(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<1x1x128x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x128x1x1xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %alloc_0 : memref<256x128x1x1xf16>
  }
  func.func private @Unknown102(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c401408 = arith.constant 401408 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    scf.for %arg3 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg3, %c28 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c28 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c28 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c28 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c28 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c28 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg2[%29, %23, %13, %3] : memref<4x128x28x28xi1>
      %31 = memref.load %arg0[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %32 = memref.load %arg1[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %33 = arith.addf %31, %32 : f16
      %34 = arith.select %30, %33, %cst : f16
      memref.store %34, %alloc[%29, %23, %13, %3] : memref<4x128x28x28xf16>
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp103(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp104(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %alloc_1 : memref<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp105(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %alloc_0 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown106(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c401408 = arith.constant 401408 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    scf.for %arg2 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg2, %c28 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c28 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c28 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c28 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c28 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c28 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x128x28x28xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x128x28x28xf16>
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp107(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp108(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %alloc_1 : memref<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp109(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %alloc_0 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown110(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c401408 = arith.constant 401408 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    scf.for %arg3 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg3, %c28 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c28 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c28 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c28 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c28 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c28 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg2[%29, %23, %13, %3] : memref<4x128x28x28xi1>
      %31 = memref.load %arg0[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %32 = memref.load %arg1[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %33 = arith.addf %31, %32 : f16
      %34 = arith.select %30, %33, %cst : f16
      memref.store %34, %alloc[%29, %23, %13, %3] : memref<4x128x28x28xf16>
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp111(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp112(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %alloc_1 : memref<4x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp113(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %alloc_0 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown114(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c401408 = arith.constant 401408 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    scf.for %arg2 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg2, %c28 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c28 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c28 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c28 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c28 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c28 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x128x28x28xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x128x28x28xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x128x28x28xf16>
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @BatchNormGradOp115(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp116(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<3x3x64x128xf16>, memref<4x64x56x56xf16>) -> ()
    return %alloc_1 : memref<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp117(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %alloc_0 : memref<128x64x3x3xf16>
  }
  func.func private @BatchNormGradOp118(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp119(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<1x1x64x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<1x1x64x128xf16>, memref<4x64x56x56xf16>) -> ()
    return %alloc_0 : memref<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp120(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<1x1x64x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x64x1x1xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %alloc_0 : memref<128x64x1x1xf16>
  }
  func.func private @Unknown121(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg3 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg3, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg2[%29, %23, %13, %3] : memref<4x64x56x56xi1>
      %31 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %32 = memref.load %arg1[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %33 = arith.addf %31, %32 : f16
      %34 = arith.select %30, %33, %cst : f16
      memref.store %34, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp122(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp123(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %alloc_1 : memref<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp124(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %alloc_0 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown125(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg2 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg2, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp126(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp127(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %alloc_1 : memref<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp128(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %alloc_0 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown129(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg3 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg3, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg2[%29, %23, %13, %3] : memref<4x64x56x56xi1>
      %31 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %32 = memref.load %arg1[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %33 = arith.addf %31, %32 : f16
      %34 = arith.select %30, %33, %cst : f16
      memref.store %34, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp130(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp131(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %alloc_1 : memref<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp132(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %alloc_0 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown133(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg2 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg2, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @BatchNormGradOp134(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp135(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %alloc) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %alloc_1 : memref<4x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp136(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %alloc_0 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown137(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %c56 = arith.constant 56 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg2 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg2, %c56 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c56 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c56 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c56 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c56 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c56 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x64x56x56xf16>
      %32 = arith.addf %30, %31 : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x64x56x56xf16>
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown138(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c3211264 = arith.constant 3211264 : index
    %c1 = arith.constant 1 : index
    %c112 = arith.constant 112 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    scf.for %arg2 = %c0 to %c3211264 step %c1 {
      %0 = arith.remsi %arg2, %c112 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c112 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c112 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c112 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c112 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c112 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<4x64x112x112xi1>
      %31 = memref.load %arg1[%29, %23, %13, %3] : memref<4x64x112x112xf16>
      %32 = arith.select %30, %31, %cst : f16
      memref.store %32, %alloc[%29, %23, %13, %3] : memref<4x64x112x112xf16>
    }
    return %alloc : memref<4x64x112x112xf16>
  }
  func.func private @BatchNormGradOp139(%arg0: memref<4x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x112x112xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x112x112xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x112x112xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf32>, memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x112x112xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<4x64x112x112xf32>, memref<4x64x112x112xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardFilterOp140(%arg0: memref<4x3x224x224xf16>, %arg1: memref<4x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<7x7x3x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x3x224x224xf16>, memref<4x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x3x7x7xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %alloc_0 : memref<64x3x7x7xf16>
  }
  func.func private @Unknown141(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.000000e+00 : f32
    %alloc = memref.alloc() : memref<f32>
    %0 = memref.load %arg0[] : memref<f32>
    %1 = arith.negf %0 : f32
    %2 = arith.divf %1, %cst : f32
    memref.store %2, %alloc[] : memref<f32>
    return %alloc : memref<f32>
  }
  func.func private @Unknown142(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c9408 = arith.constant 9408 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf32>
    scf.for %arg1 = %c0 to %c9408 step %c1 {
      %0 = arith.remsi %arg1, %c7 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c7 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c7 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c7 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c7 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c7 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c3 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c3 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c3 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x3x7x7xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x3x7x7xf32>
    }
    return %alloc : memref<64x3x7x7xf32>
  }
  func.func private @Unknown143(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x64x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x64x3x3xf32>
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown144(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x64x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x64x3x3xf32>
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown145(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x64x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x64x3x3xf32>
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown146(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<64x64x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<64x64x3x3xf32>
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown147(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c73728 = arith.constant 73728 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf32>
    scf.for %arg1 = %c0 to %c73728 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c64 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c64 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c64 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<128x64x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<128x64x3x3xf32>
    }
    return %alloc : memref<128x64x3x3xf32>
  }
  func.func private @Unknown148(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c147456 = arith.constant 147456 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    scf.for %arg1 = %c0 to %c147456 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<128x128x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<128x128x3x3xf32>
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown149(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf32>
    scf.for %arg1 = %c0 to %c8192 step %c1 {
      %0 = arith.remsi %arg1, %c64 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c64 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c64 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3, %c0, %c0] : memref<128x64x1x1xf16>
      %11 = arith.extf %10 : f16 to f32
      memref.store %11, %alloc[%9, %3, %c0, %c0] : memref<128x64x1x1xf32>
    }
    return %alloc : memref<128x64x1x1xf32>
  }
  func.func private @Unknown150(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c147456 = arith.constant 147456 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    scf.for %arg1 = %c0 to %c147456 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<128x128x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<128x128x3x3xf32>
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown151(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c147456 = arith.constant 147456 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    scf.for %arg1 = %c0 to %c147456 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<128x128x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<128x128x3x3xf32>
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown152(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c294912 = arith.constant 294912 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf32>
    scf.for %arg1 = %c0 to %c294912 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c128 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c128 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c128 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<256x128x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<256x128x3x3xf32>
    }
    return %alloc : memref<256x128x3x3xf32>
  }
  func.func private @Unknown153(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c589824 = arith.constant 589824 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    scf.for %arg1 = %c0 to %c589824 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<256x256x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<256x256x3x3xf32>
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown154(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf32>
    scf.for %arg1 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg1, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3, %c0, %c0] : memref<256x128x1x1xf16>
      %11 = arith.extf %10 : f16 to f32
      memref.store %11, %alloc[%9, %3, %c0, %c0] : memref<256x128x1x1xf32>
    }
    return %alloc : memref<256x128x1x1xf32>
  }
  func.func private @Unknown155(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c589824 = arith.constant 589824 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    scf.for %arg1 = %c0 to %c589824 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<256x256x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<256x256x3x3xf32>
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown156(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c589824 = arith.constant 589824 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    scf.for %arg1 = %c0 to %c589824 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<256x256x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<256x256x3x3xf32>
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown157(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1179648 = arith.constant 1179648 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf32>
    scf.for %arg1 = %c0 to %c1179648 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c256 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c256 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c256 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<512x256x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<512x256x3x3xf32>
    }
    return %alloc : memref<512x256x3x3xf32>
  }
  func.func private @Unknown158(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c2359296 = arith.constant 2359296 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    scf.for %arg1 = %c0 to %c2359296 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<512x512x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<512x512x3x3xf32>
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown159(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c131072 = arith.constant 131072 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf32>
    scf.for %arg1 = %c0 to %c131072 step %c1 {
      %0 = arith.remsi %arg1, %c256 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c256 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c256 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3, %c0, %c0] : memref<512x256x1x1xf16>
      %11 = arith.extf %10 : f16 to f32
      memref.store %11, %alloc[%9, %3, %c0, %c0] : memref<512x256x1x1xf32>
    }
    return %alloc : memref<512x256x1x1xf32>
  }
  func.func private @Unknown160(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c2359296 = arith.constant 2359296 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    scf.for %arg1 = %c0 to %c2359296 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<512x512x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<512x512x3x3xf32>
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown161(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c2359296 = arith.constant 2359296 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    scf.for %arg1 = %c0 to %c2359296 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c3 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c3 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c3 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c3 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c3 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.remsi %19, %c512 : index
      %21 = arith.cmpi slt, %20, %c0 : index
      %22 = arith.addi %20, %c512 : index
      %23 = arith.select %21, %22, %20 : index
      %24 = arith.cmpi slt, %19, %c0 : index
      %25 = arith.subi %c-1, %19 : index
      %26 = arith.select %24, %25, %19 : index
      %27 = arith.divsi %26, %c512 : index
      %28 = arith.subi %c-1, %27 : index
      %29 = arith.select %24, %28, %27 : index
      %30 = memref.load %arg0[%29, %23, %13, %3] : memref<512x512x3x3xf16>
      %31 = arith.extf %30 : f16 to f32
      memref.store %31, %alloc[%29, %23, %13, %3] : memref<512x512x3x3xf32>
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @MatmulOp162(%arg0: memref<4x512xf16>, %arg1: memref<4x1000xf16>) -> memref<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %alloc = memref.alloc() : memref<512x1000xf16>
    "lmhlo.dot"(%arg0, %arg1, %alloc) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512xf16>, memref<4x1000xf16>, memref<512x1000xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<1000x512xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (memref<512x1000xf16>, memref<1000x512xf16>) -> ()
    return %alloc_0 : memref<1000x512xf16>
  }
  func.func private @Unknown163(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512000 = arith.constant 512000 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() : memref<1000x512xf32>
    scf.for %arg1 = %c0 to %c512000 step %c1 {
      %0 = arith.remsi %arg1, %c512 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c512 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c512 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9, %3] : memref<1000x512xf16>
      %11 = arith.extf %10 : f16 to f32
      memref.store %11, %alloc[%9, %3] : memref<1000x512xf32>
    }
    return %alloc : memref<1000x512xf32>
  }
  func.func private @Unknown164(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1000xf32>
    scf.for %arg1 = %c0 to %c1000 step %c1 {
      %0 = memref.load %arg0[%arg1] : memref<1000xf32>
      %1 = arith.truncf %0 : f32 to f16
      %2 = arith.extf %1 : f16 to f32
      memref.store %2, %alloc[%arg1] : memref<1000xf32>
    }
    return %alloc : memref<1000xf32>
  }
  func.func @main(%arg0: memref<4x3x224x224xf32>, %arg1: memref<4x1000xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64x64x3x3xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<64xf32>, %arg11: memref<64xf32>, %arg12: memref<64x64x3x3xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64xf32>, %arg16: memref<64xf32>, %arg17: memref<64x64x3x3xf32>, %arg18: memref<64xf32>, %arg19: memref<64xf32>, %arg20: memref<64xf32>, %arg21: memref<64xf32>, %arg22: memref<64x64x3x3xf32>, %arg23: memref<64xf32>, %arg24: memref<64xf32>, %arg25: memref<64xf32>, %arg26: memref<64xf32>, %arg27: memref<128x64x3x3xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128xf32>, %arg31: memref<128xf32>, %arg32: memref<128x128x3x3xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<128xf32>, %arg37: memref<128x64x1x1xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128x3x3xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<128xf32>, %arg47: memref<128x128x3x3xf32>, %arg48: memref<128xf32>, %arg49: memref<128xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<256x128x3x3xf32>, %arg53: memref<256xf32>, %arg54: memref<256xf32>, %arg55: memref<256xf32>, %arg56: memref<256xf32>, %arg57: memref<256x256x3x3xf32>, %arg58: memref<256xf32>, %arg59: memref<256xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256x128x1x1xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256x256x3x3xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<256xf32>, %arg71: memref<256xf32>, %arg72: memref<256x256x3x3xf32>, %arg73: memref<256xf32>, %arg74: memref<256xf32>, %arg75: memref<256xf32>, %arg76: memref<256xf32>, %arg77: memref<512x256x3x3xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<512xf32>, %arg81: memref<512xf32>, %arg82: memref<512x512x3x3xf32>, %arg83: memref<512xf32>, %arg84: memref<512xf32>, %arg85: memref<512xf32>, %arg86: memref<512xf32>, %arg87: memref<512x256x1x1xf32>, %arg88: memref<512xf32>, %arg89: memref<512xf32>, %arg90: memref<512xf32>, %arg91: memref<512xf32>, %arg92: memref<512x512x3x3xf32>, %arg93: memref<512xf32>, %arg94: memref<512xf32>, %arg95: memref<512xf32>, %arg96: memref<512xf32>, %arg97: memref<512x512x3x3xf32>, %arg98: memref<512xf32>, %arg99: memref<512xf32>, %arg100: memref<512xf32>, %arg101: memref<512xf32>, %arg102: memref<1000x512xf32>, %arg103: memref<1000xf32>) -> (memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>) {
    %alloc = memref.alloc() : memref<f32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %alloc_0 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%alloc_0) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    %alloc_1 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%alloc_1) {value = dense<0xFC00> : tensor<f16>} : (memref<f16>) -> ()
    %0 = call @Unknown0(%arg0) : (memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16>
    %1 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    %alloc_2 = memref.alloc() : memref<4x64x112x112xf16>
    lmhlo.convolution(%0, %1, %alloc_2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x3x224x224xf16>, memref<64x3x7x7xf16>, memref<4x64x112x112xf16>) -> ()
    %2 = call @BatchNormTrainingOp2(%alloc_2, %arg3, %arg4) : (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x112x112xf16>
    %3 = call @Unknown3(%arg7) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %4 = call @Unknown4(%arg12) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %5 = call @Unknown5(%arg17) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %6 = call @Unknown6(%arg22) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %7 = call @Unknown7(%arg37) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %8 = call @Unknown8(%arg27) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %9 = call @Unknown9(%arg32) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %10 = call @Unknown10(%arg42) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %11 = call @Unknown11(%arg47) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %12 = call @Unknown12(%arg62) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %13 = call @Unknown13(%arg52) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %14 = call @Unknown14(%arg57) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %15 = call @Unknown15(%arg67) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %16 = call @Unknown16(%arg72) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %17 = call @Unknown17(%arg87) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %18 = call @Unknown18(%arg77) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %19 = call @Unknown19(%arg82) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %20 = call @Unknown20(%arg92) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %21 = call @Unknown21(%arg97) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %22 = call @Unknown22(%arg1) : (memref<4x1000xf32>) -> memref<4x1000xf16>
    %23 = call @Unknown23(%arg102) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    %alloc_3 = memref.alloc() : memref<4xf16>
    "lmhlo.reduce"(%22, %alloc_0, %alloc_3) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %24:2 = call @Unknown24(%2) : (memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>)
    %alloc_4 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.reduce_window"(%24#0, %alloc_1, %alloc_4) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      %alloc_32 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg104, %arg105, %alloc_32) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%alloc_32, %arg106) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<4x64x112x112xf16>, memref<f16>, memref<4x64x56x56xf16>) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%alloc_4, %3, %alloc_5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %25 = call @BatchNormTrainingOp25(%alloc_5, %arg8, %arg9) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %26:2 = call @Unknown26(%25) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_6 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%26#0, %4, %alloc_6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %27 = call @BatchNormTrainingOp27(%alloc_6, %arg13, %arg14) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %28:2 = call @Unknown28(%27, %alloc_4) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_7 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%28#0, %5, %alloc_7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %29 = call @BatchNormTrainingOp29(%alloc_7, %arg18, %arg19) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %30:2 = call @Unknown30(%29) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_8 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%30#0, %6, %alloc_8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %31 = call @BatchNormTrainingOp31(%alloc_8, %arg23, %arg24) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %32:2 = call @Unknown32(%31, %28#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_9 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%32#0, %7, %alloc_9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<128x64x1x1xf16>, memref<4x128x28x28xf16>) -> ()
    %33 = call @BatchNormTrainingOp33(%alloc_9, %arg38, %arg39) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %alloc_10 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%32#0, %8, %alloc_10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16>, memref<128x64x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %34 = call @BatchNormTrainingOp34(%alloc_10, %arg28, %arg29) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %35:2 = call @Unknown35(%34) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_11 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%35#0, %9, %alloc_11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %36 = call @BatchNormTrainingOp36(%alloc_11, %arg33, %arg34) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %37:2 = call @Unknown37(%36, %33) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_12 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%37#0, %10, %alloc_12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %38 = call @BatchNormTrainingOp38(%alloc_12, %arg43, %arg44) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %39:2 = call @Unknown39(%38) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_13 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%39#0, %11, %alloc_13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %40 = call @BatchNormTrainingOp40(%alloc_13, %arg48, %arg49) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %41:2 = call @Unknown41(%40, %37#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_14 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%41#0, %12, %alloc_14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<256x128x1x1xf16>, memref<4x256x14x14xf16>) -> ()
    %42 = call @BatchNormTrainingOp42(%alloc_14, %arg63, %arg64) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %alloc_15 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%41#0, %13, %alloc_15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16>, memref<256x128x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %43 = call @BatchNormTrainingOp43(%alloc_15, %arg53, %arg54) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %44:2 = call @Unknown44(%43) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_16 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%44#0, %14, %alloc_16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %45 = call @BatchNormTrainingOp45(%alloc_16, %arg58, %arg59) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %46:2 = call @Unknown46(%45, %42) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_17 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%46#0, %15, %alloc_17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %47 = call @BatchNormTrainingOp47(%alloc_17, %arg68, %arg69) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %48:2 = call @Unknown48(%47) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_18 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%48#0, %16, %alloc_18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %49 = call @BatchNormTrainingOp49(%alloc_18, %arg73, %arg74) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %50:2 = call @Unknown50(%49, %46#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_19 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%50#0, %17, %alloc_19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<512x256x1x1xf16>, memref<4x512x7x7xf16>) -> ()
    %51 = call @BatchNormTrainingOp51(%alloc_19, %arg88, %arg89) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %alloc_20 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%50#0, %18, %alloc_20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16>, memref<512x256x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %52 = call @BatchNormTrainingOp52(%alloc_20, %arg78, %arg79) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %53:2 = call @Unknown53(%52) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_21 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%53#0, %19, %alloc_21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %54 = call @BatchNormTrainingOp54(%alloc_21, %arg83, %arg84) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %55:2 = call @Unknown55(%54, %51) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_22 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%55#0, %20, %alloc_22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %56 = call @BatchNormTrainingOp56(%alloc_22, %arg93, %arg94) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %57:2 = call @Unknown57(%56) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_23 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%57#0, %21, %alloc_23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %58 = call @BatchNormTrainingOp58(%alloc_23, %arg98, %arg99) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %59:2 = call @Unknown59(%58, %55#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_24 = memref.alloc() : memref<4x512xf16>
    "lmhlo.reduce"(%59#0, %alloc_0, %alloc_24) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<4x512x7x7xf16>, memref<f16>, memref<4x512xf16>) -> ()
    %60 = call @Unknown60(%alloc_24) : (memref<4x512xf16>) -> memref<4x512xf16>
    %alloc_25 = memref.alloc() : memref<4x1000xf16>
    "lmhlo.dot"(%60, %23, %alloc_25) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512xf16>, memref<1000x512xf16>, memref<4x1000xf16>) -> ()
    %61 = call @Unknown61(%arg103, %alloc_25) : (memref<1000xf32>, memref<4x1000xf16>) -> memref<4x1000xf16>
    %alloc_26 = memref.alloc() : memref<4xf16>
    "lmhlo.reduce"(%61, %alloc_1, %alloc_26) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.maximum"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %62:2 = call @Unknown62(%alloc_26, %61) : (memref<4xf16>, memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>)
    %alloc_27 = memref.alloc() : memref<4xf16>
    "lmhlo.reduce"(%62#1, %alloc_0, %alloc_27) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %63:3 = call @Unknown63(%alloc_27, %62#0, %alloc_3, %22, %arg1) : (memref<4xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf16>, memref<4x1000xf32>) -> (memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>)
    %alloc_28 = memref.alloc() : memref<4x512xf16>
    "lmhlo.dot"(%63#0, %23, %alloc_28) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x1000xf16>, memref<1000x512xf16>, memref<4x512xf16>) -> ()
    %64 = call @Unknown64(%alloc_28, %59#1) : (memref<4x512xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %65:3 = call @BatchNormGradOp65(%alloc_23, %arg98, %64) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %66 = call @ConvBackwardDataOp66(%65#0, %21) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %67 = call @ConvBackwardFilterOp67(%57#0, %65#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %68 = call @Unknown68(%57#1, %66) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %69:3 = call @BatchNormGradOp69(%alloc_22, %arg93, %68) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %70 = call @ConvBackwardDataOp70(%69#0, %20) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %71 = call @ConvBackwardFilterOp71(%55#0, %69#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %72 = call @Unknown72(%64, %70, %55#1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %73:3 = call @BatchNormGradOp73(%alloc_21, %arg83, %72) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %74 = call @ConvBackwardDataOp74(%73#0, %19) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %75 = call @ConvBackwardFilterOp75(%53#0, %73#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %76 = call @Unknown76(%53#1, %74) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %77:3 = call @BatchNormGradOp77(%alloc_20, %arg78, %76) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %78 = call @ConvBackwardDataOp78(%77#0, %18) : (memref<4x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %79 = call @ConvBackwardFilterOp79(%50#0, %77#0) : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %80:3 = call @BatchNormGradOp80(%alloc_19, %arg88, %72) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %81 = call @ConvBackwardDataOp81(%80#0, %17) : (memref<4x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<4x256x14x14xf16>
    %82 = call @ConvBackwardFilterOp82(%50#0, %80#0) : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %83 = call @Unknown83(%81, %78, %50#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %84:3 = call @BatchNormGradOp84(%alloc_18, %arg73, %83) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %85 = call @ConvBackwardDataOp85(%84#0, %16) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %86 = call @ConvBackwardFilterOp86(%48#0, %84#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %87 = call @Unknown87(%48#1, %85) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %88:3 = call @BatchNormGradOp88(%alloc_17, %arg68, %87) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %89 = call @ConvBackwardDataOp89(%88#0, %15) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %90 = call @ConvBackwardFilterOp90(%46#0, %88#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %91 = call @Unknown91(%83, %89, %46#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %92:3 = call @BatchNormGradOp92(%alloc_16, %arg58, %91) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %93 = call @ConvBackwardDataOp93(%92#0, %14) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %94 = call @ConvBackwardFilterOp94(%44#0, %92#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %95 = call @Unknown95(%44#1, %93) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %96:3 = call @BatchNormGradOp96(%alloc_15, %arg53, %95) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %97 = call @ConvBackwardDataOp97(%96#0, %13) : (memref<4x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %98 = call @ConvBackwardFilterOp98(%41#0, %96#0) : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %99:3 = call @BatchNormGradOp99(%alloc_14, %arg63, %91) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %100 = call @ConvBackwardDataOp100(%99#0, %12) : (memref<4x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<4x128x28x28xf16>
    %101 = call @ConvBackwardFilterOp101(%41#0, %99#0) : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %102 = call @Unknown102(%100, %97, %41#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %103:3 = call @BatchNormGradOp103(%alloc_13, %arg48, %102) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %104 = call @ConvBackwardDataOp104(%103#0, %11) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %105 = call @ConvBackwardFilterOp105(%39#0, %103#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %106 = call @Unknown106(%39#1, %104) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %107:3 = call @BatchNormGradOp107(%alloc_12, %arg43, %106) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %108 = call @ConvBackwardDataOp108(%107#0, %10) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %109 = call @ConvBackwardFilterOp109(%37#0, %107#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %110 = call @Unknown110(%102, %108, %37#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %111:3 = call @BatchNormGradOp111(%alloc_11, %arg33, %110) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %112 = call @ConvBackwardDataOp112(%111#0, %9) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %113 = call @ConvBackwardFilterOp113(%35#0, %111#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %114 = call @Unknown114(%35#1, %112) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %115:3 = call @BatchNormGradOp115(%alloc_10, %arg28, %114) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %116 = call @ConvBackwardDataOp116(%115#0, %8) : (memref<4x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %117 = call @ConvBackwardFilterOp117(%32#0, %115#0) : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %118:3 = call @BatchNormGradOp118(%alloc_9, %arg38, %110) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %119 = call @ConvBackwardDataOp119(%118#0, %7) : (memref<4x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<4x64x56x56xf16>
    %120 = call @ConvBackwardFilterOp120(%32#0, %118#0) : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %121 = call @Unknown121(%119, %116, %32#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %122:3 = call @BatchNormGradOp122(%alloc_8, %arg23, %121) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %123 = call @ConvBackwardDataOp123(%122#0, %6) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %124 = call @ConvBackwardFilterOp124(%30#0, %122#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %125 = call @Unknown125(%30#1, %123) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %126:3 = call @BatchNormGradOp126(%alloc_7, %arg18, %125) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %127 = call @ConvBackwardDataOp127(%126#0, %5) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %128 = call @ConvBackwardFilterOp128(%28#0, %126#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %129 = call @Unknown129(%121, %127, %28#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %130:3 = call @BatchNormGradOp130(%alloc_6, %arg13, %129) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %131 = call @ConvBackwardDataOp131(%130#0, %4) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %132 = call @ConvBackwardFilterOp132(%26#0, %130#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %133 = call @Unknown133(%26#1, %131) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %134:3 = call @BatchNormGradOp134(%alloc_5, %arg8, %133) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %135 = call @ConvBackwardDataOp135(%134#0, %3) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %136 = call @ConvBackwardFilterOp136(%alloc_4, %134#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %137 = call @Unknown137(%129, %135) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_29 = memref.alloc() : memref<4x64x112x112xf16>
    "lmhlo.select_and_scatter"(%24#0, %137, %alloc_0, %alloc_29) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %165 = mhlo.compare  GE, %arg104, %arg105 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %165 : tensor<i1>
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %165 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %165 : tensor<f16>
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<4x64x112x112xf16>, memref<4x64x56x56xf16>, memref<f16>, memref<4x64x112x112xf16>) -> ()
    %138 = call @Unknown138(%24#1, %alloc_29) : (memref<4x64x112x112xi1>, memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16>
    %139:3 = call @BatchNormGradOp139(%alloc_2, %arg3, %138) : (memref<4x64x112x112xf16>, memref<64xf32>, memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %140 = call @ConvBackwardFilterOp140(%0, %139#0) : (memref<4x3x224x224xf16>, memref<4x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %alloc_30 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%63#1, %alloc, %alloc_30) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<4x1000xf32>, memref<f32>, memref<f32>) -> ()
    %141 = call @Unknown141(%alloc_30) : (memref<f32>) -> memref<f32>
    %142 = call @Unknown142(%140) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %143 = call @Unknown143(%136) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %144 = call @Unknown144(%132) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %145 = call @Unknown145(%128) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %146 = call @Unknown146(%124) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %147 = call @Unknown147(%117) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %148 = call @Unknown148(%113) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %149 = call @Unknown149(%120) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %150 = call @Unknown150(%109) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %151 = call @Unknown151(%105) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %152 = call @Unknown152(%98) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %153 = call @Unknown153(%94) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %154 = call @Unknown154(%101) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %155 = call @Unknown155(%90) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %156 = call @Unknown156(%86) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %157 = call @Unknown157(%79) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %158 = call @Unknown158(%75) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %159 = call @Unknown159(%82) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %160 = call @Unknown160(%71) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %161 = call @Unknown161(%67) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %162 = call @MatmulOp162(%60, %63#0) : (memref<4x512xf16>, memref<4x1000xf16>) -> memref<1000x512xf16>
    %163 = call @Unknown163(%162) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %alloc_31 = memref.alloc() : memref<1000xf32>
    "lmhlo.reduce"(%63#2, %alloc, %alloc_31) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<4x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %164 = call @Unknown164(%alloc_31) : (memref<1000xf32>) -> memref<1000xf32>
    return %141, %142, %139#1, %139#2, %143, %134#1, %134#2, %144, %130#1, %130#2, %145, %126#1, %126#2, %146, %122#1, %122#2, %147, %115#1, %115#2, %148, %111#1, %111#2, %149, %118#1, %118#2, %150, %107#1, %107#2, %151, %103#1, %103#2, %152, %96#1, %96#2, %153, %92#1, %92#2, %154, %99#1, %99#2, %155, %88#1, %88#2, %156, %84#1, %84#2, %157, %77#1, %77#2, %158, %73#1, %73#2, %159, %80#1, %80#2, %160, %69#1, %69#2, %161, %65#1, %65#2, %163, %164 : memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>
  }
}