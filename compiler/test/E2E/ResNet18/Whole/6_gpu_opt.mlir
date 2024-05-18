// RUN: byteir-opt %s -gpu-opt | FileCheck %s

// CHECK-LABEL: func.func @main

module @IrToMhlo.2452 {
  func.func private @Unknown0(%arg0: memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %c224 = arith.constant 224 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c602112 = arith.constant 602112 : index
    %alloc = memref.alloc() : memref<4x3x224x224xf16>
    scf.for %arg1 = %c0 to %c602112 step %c1 {
      %0 = arith.remsi %arg1, %c224 : index
      %1 = arith.divsi %arg1, %c224 : index
      %2 = arith.remsi %1, %c224 : index
      %3 = arith.divsi %1, %c224 : index
      %4 = arith.remsi %3, %c3 : index
      %5 = arith.divsi %3, %c3 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x3x224x224xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<4x3x224x224xf16>
    }
    return %alloc : memref<4x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c9408 = arith.constant 9408 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf16>
    scf.for %arg1 = %c0 to %c9408 step %c1 {
      %0 = arith.remsi %arg1, %c7 : index
      %1 = arith.divsi %arg1, %c7 : index
      %2 = arith.remsi %1, %c7 : index
      %3 = arith.divsi %1, %c7 : index
      %4 = arith.remsi %3, %c3 : index
      %5 = arith.divsi %3, %c3 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<64x3x7x7xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<64x3x7x7xf16>
    }
    return %alloc : memref<64x3x7x7xf16>
  }
  func.func private @Unknown3(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<64x64x3x3xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<64x64x3x3xf16>
    }
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown7(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf16>
    scf.for %arg1 = %c0 to %c8192 step %c1 {
      %0 = arith.remsi %arg1, %c64 : index
      %1 = arith.divsi %arg1, %c64 : index
      %2 = memref.load %arg0[%1, %0, %c0, %c0] : memref<128x64x1x1xf32>
      %3 = arith.truncf %2 : f32 to f16
      memref.store %3, %alloc[%1, %0, %c0, %c0] : memref<128x64x1x1xf16>
    }
    return %alloc : memref<128x64x1x1xf16>
  }
  func.func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c73728 = arith.constant 73728 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf16>
    scf.for %arg1 = %c0 to %c73728 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<128x64x3x3xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<128x64x3x3xf16>
    }
    return %alloc : memref<128x64x3x3xf16>
  }
  func.func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c147456 = arith.constant 147456 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    scf.for %arg1 = %c0 to %c147456 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c128 : index
      %5 = arith.divsi %3, %c128 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<128x128x3x3xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<128x128x3x3xf16>
    }
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown12(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf16>
    scf.for %arg1 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg1, %c128 : index
      %1 = arith.divsi %arg1, %c128 : index
      %2 = memref.load %arg0[%1, %0, %c0, %c0] : memref<256x128x1x1xf32>
      %3 = arith.truncf %2 : f32 to f16
      memref.store %3, %alloc[%1, %0, %c0, %c0] : memref<256x128x1x1xf16>
    }
    return %alloc : memref<256x128x1x1xf16>
  }
  func.func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c294912 = arith.constant 294912 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf16>
    scf.for %arg1 = %c0 to %c294912 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c128 : index
      %5 = arith.divsi %3, %c128 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<256x128x3x3xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<256x128x3x3xf16>
    }
    return %alloc : memref<256x128x3x3xf16>
  }
  func.func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c589824 = arith.constant 589824 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    scf.for %arg1 = %c0 to %c589824 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c256 : index
      %5 = arith.divsi %3, %c256 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<256x256x3x3xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<256x256x3x3xf16>
    }
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown17(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c131072 = arith.constant 131072 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf16>
    scf.for %arg1 = %c0 to %c131072 step %c1 {
      %0 = arith.remsi %arg1, %c256 : index
      %1 = arith.divsi %arg1, %c256 : index
      %2 = memref.load %arg0[%1, %0, %c0, %c0] : memref<512x256x1x1xf32>
      %3 = arith.truncf %2 : f32 to f16
      memref.store %3, %alloc[%1, %0, %c0, %c0] : memref<512x256x1x1xf16>
    }
    return %alloc : memref<512x256x1x1xf16>
  }
  func.func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1179648 = arith.constant 1179648 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf16>
    scf.for %arg1 = %c0 to %c1179648 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c256 : index
      %5 = arith.divsi %3, %c256 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<512x256x3x3xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<512x256x3x3xf16>
    }
    return %alloc : memref<512x256x3x3xf16>
  }
  func.func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c2359296 = arith.constant 2359296 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    scf.for %arg1 = %c0 to %c2359296 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c512 : index
      %5 = arith.divsi %3, %c512 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<512x512x3x3xf32>
      %7 = arith.truncf %6 : f32 to f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<512x512x3x3xf16>
    }
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown22(%arg0: memref<4x1000xf32>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant -2.500000e-01 : f32
    %c4000 = arith.constant 4000 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    scf.for %arg1 = %c0 to %c4000 step %c1 {
      %0 = arith.remsi %arg1, %c1000 : index
      %1 = arith.divsi %arg1, %c1000 : index
      %2 = memref.load %arg0[%1, %0] : memref<4x1000xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.truncf %3 : f32 to f16
      memref.store %4, %alloc[%1, %0] : memref<4x1000xf16>
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown23(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c512000 = arith.constant 512000 : index
    %alloc = memref.alloc() : memref<1000x512xf16>
    scf.for %arg1 = %c0 to %c512000 step %c1 {
      %0 = arith.remsi %arg1, %c512 : index
      %1 = arith.divsi %arg1, %c512 : index
      %2 = memref.load %arg0[%1, %0] : memref<1000x512xf32>
      %3 = arith.truncf %2 : f32 to f16
      memref.store %3, %alloc[%1, %0] : memref<1000x512xf16>
    }
    return %alloc : memref<1000x512xf16>
  }
  func.func private @Unknown24(%arg0: memref<1000xf32>) -> memref<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1000xf16>
    scf.for %arg1 = %c0 to %c1000 step %c1 {
      %0 = memref.load %arg0[%arg1] : memref<1000xf32>
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %alloc[%arg1] : memref<1000xf16>
    }
    return %alloc : memref<1000xf16>
  }
  func.func private @Unknown25(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c-1 = arith.constant -1 : index
    %c-1024 = arith.constant -1024 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() : memref<4xf16>
    scf.forall (%arg1) in (4) {
      %subview = memref.subview %arg0[%arg1, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (512) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = arith.cmpi slt, %arg2, %c0 : index
        %2 = arith.subi %c-1, %arg2 : index
        %3 = arith.select %1, %2, %arg2 : index
        %4 = arith.divsi %3, %c512 : index
        %5 = arith.subi %c-1, %4 : index
        %6 = arith.select %1, %5, %4 : index
        %7 = arith.muli %6, %c-1024 : index
        %8 = arith.addi %0, %7 : index
        %9 = arith.cmpi slt, %8, %c1000 : index
        %10 = arith.select %9, %8, %c1000 : index
        %11 = arith.addi %8, %c2 : index
        %12 = arith.cmpi slt, %11, %c1000 : index
        %13 = arith.select %12, %11, %c1000 : index
        %14 = arith.subi %13, %10 : index
        %subview_8 = memref.subview %expand_shape[0, %10] [1, %14] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_9 = memref.expand_shape %subview_8 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %15 = arith.cmpi ugt, %14, %c0 : index
        %16 = scf.if %15 -> (f16) {
          %21 = memref.load %expand_shape_9[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %21 : f16
        } else {
          scf.yield %cst : f16
        }
        %17 = arith.addf %16, %cst : f16
        %18 = arith.cmpi ugt, %14, %c1 : index
        %19 = scf.if %18 -> (f16) {
          %21 = memref.load %expand_shape_9[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %21 : f16
        } else {
          scf.yield %cst : f16
        }
        %20 = arith.addf %17, %19 : f16
        memref.store %20, %alloca[%arg2] : memref<512xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_0 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (256) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca[%0] : memref<512xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca[%3] : memref<512xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_0[%arg2] : memref<256xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_1 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (128) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_0[%0] : memref<256xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_0[%3] : memref<256xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_1[%arg2] : memref<128xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_2 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_1[%0] : memref<128xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_1[%3] : memref<128xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_2[%arg2] : memref<64xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_3 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (32) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_2[%0] : memref<64xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_2[%3] : memref<64xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_3[%arg2] : memref<32xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (16) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_3[%0] : memref<32xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_3[%3] : memref<32xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_4[%arg2] : memref<16xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (8) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_4[%0] : memref<16xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_4[%3] : memref<16xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_5[%arg2] : memref<8xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_6 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (4) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_5[%0] : memref<8xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_5[%3] : memref<8xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_6[%arg2] : memref<4xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (2) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_6[%0] : memref<4xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_6[%3] : memref<4xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_7[%arg2] : memref<2xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg2) in (1) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_7[%0] : memref<2xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_7[%3] : memref<2xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloc[%arg1] : memref<4xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown26(%arg0: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %c112 = arith.constant 112 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c3211264 = arith.constant 3211264 : index
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    %alloc_0 = memref.alloc() : memref<4x64x112x112xi1>
    scf.for %arg1 = %c0 to %c3211264 step %c1 {
      %0 = arith.remsi %arg1, %c112 : index
      %1 = arith.divsi %arg1, %c112 : index
      %2 = arith.remsi %1, %c112 : index
      %3 = arith.divsi %1, %c112 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x64x112x112xf16>
      %7 = arith.maximumf %6, %cst : f16
      %8 = arith.cmpf ogt, %7, %cst : f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<4x64x112x112xf16>
      memref.store %8, %alloc_0[%5, %4, %2, %0] : memref<4x64x112x112xi1>
    }
    return %alloc, %alloc_0 : memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
  }
  func.func private @Unknown28(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c802816 = arith.constant 802816 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    scf.for %arg1 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg1, %c56 : index
      %1 = arith.divsi %arg1, %c56 : index
      %2 = arith.remsi %1, %c56 : index
      %3 = arith.divsi %1, %c56 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      %7 = arith.maximumf %6, %cst : f16
      %8 = arith.cmpf ogt, %7, %cst : f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      memref.store %8, %alloc_0[%5, %4, %2, %0] : memref<4x64x56x56xi1>
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown30(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c802816 = arith.constant 802816 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xi1>
    scf.for %arg2 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg2, %c56 : index
      %1 = arith.divsi %arg2, %c56 : index
      %2 = arith.remsi %1, %c56 : index
      %3 = arith.divsi %1, %c56 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maximumf %8, %cst : f16
      %10 = arith.cmpf ogt, %9, %cst : f16
      memref.store %9, %alloc[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      memref.store %10, %alloc_0[%5, %4, %2, %0] : memref<4x64x56x56xi1>
    }
    return %alloc, %alloc_0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown37(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c401408 = arith.constant 401408 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    scf.for %arg1 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg1, %c28 : index
      %1 = arith.divsi %arg1, %c28 : index
      %2 = arith.remsi %1, %c28 : index
      %3 = arith.divsi %1, %c28 : index
      %4 = arith.remsi %3, %c128 : index
      %5 = arith.divsi %3, %c128 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x128x28x28xf16>
      %7 = arith.maximumf %6, %cst : f16
      %8 = arith.cmpf ogt, %7, %cst : f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<4x128x28x28xf16>
      memref.store %8, %alloc_0[%5, %4, %2, %0] : memref<4x128x28x28xi1>
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown39(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c401408 = arith.constant 401408 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xi1>
    scf.for %arg2 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg2, %c28 : index
      %1 = arith.divsi %arg2, %c28 : index
      %2 = arith.remsi %1, %c28 : index
      %3 = arith.divsi %1, %c28 : index
      %4 = arith.remsi %3, %c128 : index
      %5 = arith.divsi %3, %c128 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x128x28x28xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maximumf %8, %cst : f16
      %10 = arith.cmpf ogt, %9, %cst : f16
      memref.store %9, %alloc[%5, %4, %2, %0] : memref<4x128x28x28xf16>
      memref.store %10, %alloc_0[%5, %4, %2, %0] : memref<4x128x28x28xi1>
    }
    return %alloc, %alloc_0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown46(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c200704 = arith.constant 200704 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    scf.for %arg1 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg1, %c14 : index
      %1 = arith.divsi %arg1, %c14 : index
      %2 = arith.remsi %1, %c14 : index
      %3 = arith.divsi %1, %c14 : index
      %4 = arith.remsi %3, %c256 : index
      %5 = arith.divsi %3, %c256 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x256x14x14xf16>
      %7 = arith.maximumf %6, %cst : f16
      %8 = arith.cmpf ogt, %7, %cst : f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<4x256x14x14xf16>
      memref.store %8, %alloc_0[%5, %4, %2, %0] : memref<4x256x14x14xi1>
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown48(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c200704 = arith.constant 200704 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xi1>
    scf.for %arg2 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg2, %c14 : index
      %1 = arith.divsi %arg2, %c14 : index
      %2 = arith.remsi %1, %c14 : index
      %3 = arith.divsi %1, %c14 : index
      %4 = arith.remsi %3, %c256 : index
      %5 = arith.divsi %3, %c256 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x256x14x14xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maximumf %8, %cst : f16
      %10 = arith.cmpf ogt, %9, %cst : f16
      memref.store %9, %alloc[%5, %4, %2, %0] : memref<4x256x14x14xf16>
      memref.store %10, %alloc_0[%5, %4, %2, %0] : memref<4x256x14x14xi1>
    }
    return %alloc, %alloc_0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown55(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c100352 = arith.constant 100352 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    scf.for %arg1 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg1, %c7 : index
      %1 = arith.divsi %arg1, %c7 : index
      %2 = arith.remsi %1, %c7 : index
      %3 = arith.divsi %1, %c7 : index
      %4 = arith.remsi %3, %c512 : index
      %5 = arith.divsi %3, %c512 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x512x7x7xf16>
      %7 = arith.maximumf %6, %cst : f16
      %8 = arith.cmpf ogt, %7, %cst : f16
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<4x512x7x7xf16>
      memref.store %8, %alloc_0[%5, %4, %2, %0] : memref<4x512x7x7xi1>
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown57(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c100352 = arith.constant 100352 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xi1>
    scf.for %arg2 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg2, %c7 : index
      %1 = arith.divsi %arg2, %c7 : index
      %2 = arith.remsi %1, %c7 : index
      %3 = arith.divsi %1, %c7 : index
      %4 = arith.remsi %3, %c512 : index
      %5 = arith.divsi %3, %c512 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x512x7x7xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x512x7x7xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maximumf %8, %cst : f16
      %10 = arith.cmpf ogt, %9, %cst : f16
      memref.store %9, %alloc[%5, %4, %2, %0] : memref<4x512x7x7xf16>
      memref.store %10, %alloc_0[%5, %4, %2, %0] : memref<4x512x7x7xi1>
    }
    return %alloc, %alloc_0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown62(%arg0: memref<4x512x7x7xf16>) -> memref<4x512xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c64 = arith.constant 64 : index
    %c49 = arith.constant 49 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2, 3]] : memref<4x512x7x7xf16> into memref<2048x49xf16>
    %alloc = memref.alloc() : memref<2048xf16>
    scf.forall (%arg1) in (2048) {
      %subview = memref.subview %collapse_shape[%arg1, 0] [1, 49] [1, 1] : memref<2048x49xf16> to memref<49xf16, strided<[1], offset: ?>>
      %expand_shape_0 = memref.expand_shape %subview [[0, 1]] : memref<49xf16, strided<[1], offset: ?>> into memref<1x49xf16, strided<[49, 1], offset: ?>>
      %alloca = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = arith.remsi %arg2, %c64 : index
        %1 = arith.cmpi slt, %0, %c0 : index
        %2 = arith.addi %0, %c64 : index
        %3 = arith.select %1, %2, %0 : index
        %4 = arith.cmpi slt, %3, %c49 : index
        %5 = arith.select %4, %3, %c49 : index
        %6 = arith.addi %3, %c1 : index
        %7 = arith.cmpi slt, %6, %c49 : index
        %8 = arith.select %7, %6, %c49 : index
        %9 = arith.subi %8, %5 : index
        %subview_6 = memref.subview %expand_shape_0[0, %5] [1, %9] [1, 1] : memref<1x49xf16, strided<[49, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_7 = memref.expand_shape %subview_6 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %10 = arith.cmpi ugt, %9, %c0 : index
        %11 = scf.if %10 -> (f16) {
          %13 = memref.load %expand_shape_7[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %13 : f16
        } else {
          scf.yield %cst : f16
        }
        %12 = arith.addf %11, %cst : f16
        memref.store %12, %alloca[%arg2] : memref<64xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_1 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (32) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca[%0] : memref<64xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca[%3] : memref<64xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_1[%arg2] : memref<32xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_2 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (16) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_1[%0] : memref<32xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_1[%3] : memref<32xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_2[%arg2] : memref<16xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_3 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (8) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_2[%0] : memref<16xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_2[%3] : memref<16xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_3[%arg2] : memref<8xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (4) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_3[%0] : memref<8xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_3[%3] : memref<8xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_4[%arg2] : memref<4xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (2) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_4[%0] : memref<4xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_4[%3] : memref<4xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_5[%arg2] : memref<2xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg2) in (1) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_5[%0] : memref<2xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_5[%3] : memref<2xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloc[%arg1] : memref<2048xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    %expand_shape = memref.expand_shape %alloc [[0, 1]] : memref<2048xf16> into memref<4x512xf16>
    return %expand_shape : memref<4x512xf16>
  }
  func.func private @Unknown63(%arg0: memref<4x512xf16>) -> memref<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.040100e-02 : f16
    %c2048 = arith.constant 2048 : index
    %alloc = memref.alloc() : memref<4x512xf16>
    scf.for %arg1 = %c0 to %c2048 step %c1 {
      %0 = arith.remsi %arg1, %c512 : index
      %1 = arith.divsi %arg1, %c512 : index
      %2 = memref.load %arg0[%1, %0] : memref<4x512xf16>
      %3 = arith.mulf %2, %cst : f16
      memref.store %3, %alloc[%1, %0] : memref<4x512xf16>
    }
    return %alloc : memref<4x512xf16>
  }
  func.func private @Unknown64(%arg0: memref<1000xf16>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4000 = arith.constant 4000 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    scf.for %arg2 = %c0 to %c4000 step %c1 {
      %0 = arith.remsi %arg2, %c1000 : index
      %1 = arith.divsi %arg2, %c1000 : index
      %2 = memref.load %arg0[%0] : memref<1000xf16>
      %3 = memref.load %arg1[%1, %0] : memref<4x1000xf16>
      %4 = arith.addf %3, %2 : f16
      memref.store %4, %alloc[%1, %0] : memref<4x1000xf16>
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown65(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c-1 = arith.constant -1 : index
    %c-1024 = arith.constant -1024 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() : memref<4xf16>
    scf.forall (%arg1) in (4) {
      %subview = memref.subview %arg0[%arg1, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (512) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = arith.cmpi slt, %arg2, %c0 : index
        %2 = arith.subi %c-1, %arg2 : index
        %3 = arith.select %1, %2, %arg2 : index
        %4 = arith.divsi %3, %c512 : index
        %5 = arith.subi %c-1, %4 : index
        %6 = arith.select %1, %5, %4 : index
        %7 = arith.muli %6, %c-1024 : index
        %8 = arith.addi %0, %7 : index
        %9 = arith.cmpi slt, %8, %c1000 : index
        %10 = arith.select %9, %8, %c1000 : index
        %11 = arith.addi %8, %c2 : index
        %12 = arith.cmpi slt, %11, %c1000 : index
        %13 = arith.select %12, %11, %c1000 : index
        %14 = arith.subi %13, %10 : index
        %subview_8 = memref.subview %expand_shape[0, %10] [1, %14] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_9 = memref.expand_shape %subview_8 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %15 = arith.cmpi ugt, %14, %c0 : index
        %16 = scf.if %15 -> (f16) {
          %20 = memref.load %expand_shape_9[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %20 : f16
        } else {
          scf.yield %cst : f16
        }
        %17 = arith.cmpi ugt, %14, %c1 : index
        %18 = scf.if %17 -> (f16) {
          %20 = memref.load %expand_shape_9[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %20 : f16
        } else {
          scf.yield %cst : f16
        }
        %19 = arith.maximumf %16, %18 : f16
        memref.store %19, %alloca[%arg2] : memref<512xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_0 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (256) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca[%0] : memref<512xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca[%2] : memref<512xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_0[%arg2] : memref<256xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_1 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (128) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_0[%0] : memref<256xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca_0[%2] : memref<256xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_1[%arg2] : memref<128xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_2 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_1[%0] : memref<128xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca_1[%2] : memref<128xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_2[%arg2] : memref<64xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_3 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (32) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_2[%0] : memref<64xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca_2[%2] : memref<64xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_3[%arg2] : memref<32xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (16) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_3[%0] : memref<32xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca_3[%2] : memref<32xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_4[%arg2] : memref<16xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (8) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_4[%0] : memref<16xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca_4[%2] : memref<16xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_5[%arg2] : memref<8xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_6 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (4) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_5[%0] : memref<8xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca_5[%2] : memref<8xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_6[%arg2] : memref<4xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (2) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_6[%0] : memref<4xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca_6[%2] : memref<4xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloca_7[%arg2] : memref<2xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg2) in (1) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_7[%0] : memref<2xf16, #gpu.address_space<workgroup>>
        %2 = arith.addi %0, %c1 : index
        %3 = memref.load %alloca_7[%2] : memref<2xf16, #gpu.address_space<workgroup>>
        %4 = arith.maximumf %3, %1 : f16
        memref.store %4, %alloc[%arg1] : memref<4xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown66(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4000 = arith.constant 4000 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    scf.for %arg2 = %c0 to %c4000 step %c1 {
      %0 = arith.remsi %arg2, %c1000 : index
      %1 = arith.divsi %arg2, %c1000 : index
      %2 = memref.load %arg0[%1] : memref<4xf16>
      %3 = memref.load %arg1[%1, %0] : memref<4x1000xf16>
      %4 = arith.subf %3, %2 : f16
      memref.store %4, %alloc[%1, %0] : memref<4x1000xf16>
    }
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown67(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c-1 = arith.constant -1 : index
    %c-1024 = arith.constant -1024 : index
    %c1000 = arith.constant 1000 : index
    %alloc = memref.alloc() : memref<4xf16>
    scf.forall (%arg1) in (4) {
      %subview = memref.subview %arg0[%arg1, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (512) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = arith.cmpi slt, %arg2, %c0 : index
        %2 = arith.subi %c-1, %arg2 : index
        %3 = arith.select %1, %2, %arg2 : index
        %4 = arith.divsi %3, %c512 : index
        %5 = arith.subi %c-1, %4 : index
        %6 = arith.select %1, %5, %4 : index
        %7 = arith.muli %6, %c-1024 : index
        %8 = arith.addi %0, %7 : index
        %9 = arith.cmpi slt, %8, %c1000 : index
        %10 = arith.select %9, %8, %c1000 : index
        %11 = arith.addi %8, %c2 : index
        %12 = arith.cmpi slt, %11, %c1000 : index
        %13 = arith.select %12, %11, %c1000 : index
        %14 = arith.subi %13, %10 : index
        %subview_8 = memref.subview %expand_shape[0, %10] [1, %14] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_9 = memref.expand_shape %subview_8 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %15 = arith.cmpi ugt, %14, %c0 : index
        %16 = scf.if %15 -> (f16) {
          %23 = memref.load %expand_shape_9[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %23 : f16
        } else {
          scf.yield %cst : f16
        }
        %17 = math.exp %16 : f16
        %18 = arith.addf %17, %cst : f16
        %19 = arith.cmpi ugt, %14, %c1 : index
        %20 = scf.if %19 -> (f16) {
          %23 = memref.load %expand_shape_9[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %23 : f16
        } else {
          scf.yield %cst : f16
        }
        %21 = math.exp %20 : f16
        %22 = arith.addf %18, %21 : f16
        memref.store %22, %alloca[%arg2] : memref<512xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_0 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (256) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca[%0] : memref<512xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca[%3] : memref<512xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_0[%arg2] : memref<256xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_1 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (128) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_0[%0] : memref<256xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_0[%3] : memref<256xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_1[%arg2] : memref<128xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_2 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (64) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_1[%0] : memref<128xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_1[%3] : memref<128xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_2[%arg2] : memref<64xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_3 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (32) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_2[%0] : memref<64xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_2[%3] : memref<64xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_3[%arg2] : memref<32xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (16) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_3[%0] : memref<32xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_3[%3] : memref<32xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_4[%arg2] : memref<16xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (8) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_4[%0] : memref<16xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_4[%3] : memref<16xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_5[%arg2] : memref<8xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_6 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (4) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_5[%0] : memref<8xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_5[%3] : memref<8xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_6[%arg2] : memref<4xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      scf.forall (%arg2) in (2) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_6[%0] : memref<4xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_6[%3] : memref<4xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloca_7[%arg2] : memref<2xf16, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg2) in (1) {
        %0 = arith.muli %arg2, %c2 : index
        %1 = memref.load %alloca_7[%0] : memref<2xf16, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst : f16
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_7[%3] : memref<2xf16, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f16
        memref.store %5, %alloc[%arg1] : memref<4xf16>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown68(%arg0: memref<4xf16>) -> memref<4xf16> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4xf16>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      %0 = memref.load %arg0[%arg1] : memref<4xf16>
      %1 = math.log %0 : f16
      memref.store %1, %alloc[%arg1] : memref<4xf16>
    }
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown69(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4000 = arith.constant 4000 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    %alloc_0 = memref.alloc() : memref<4x1000xf16>
    scf.for %arg4 = %c0 to %c4000 step %c1 {
      %0 = arith.remsi %arg4, %c1000 : index
      %1 = arith.divsi %arg4, %c1000 : index
      %2 = memref.load %arg2[%1] : memref<4xf16>
      %3 = memref.load %arg0[%1] : memref<4xf16>
      %4 = memref.load %arg1[%1, %0] : memref<4x1000xf16>
      %5 = memref.load %arg3[%1, %0] : memref<4x1000xf16>
      %6 = arith.subf %4, %3 : f16
      %7 = math.exp %6 : f16
      %8 = arith.mulf %7, %2 : f16
      %9 = arith.subf %5, %8 : f16
      memref.store %6, %alloc[%1, %0] : memref<4x1000xf16>
      memref.store %9, %alloc_0[%1, %0] : memref<4x1000xf16>
    }
    return %alloc, %alloc_0 : memref<4x1000xf16>, memref<4x1000xf16>
  }
  func.func private @Unknown70(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 4.900000e+01 : f16
    %c100352 = arith.constant 100352 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg2 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg2, %c7 : index
      %1 = arith.divsi %arg2, %c7 : index
      %2 = arith.remsi %1, %c7 : index
      %3 = arith.divsi %1, %c7 : index
      %4 = arith.remsi %3, %c512 : index
      %5 = arith.divsi %3, %c512 : index
      %6 = memref.load %arg0[%5, %4] : memref<4x512xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x512x7x7xi1>
      %8 = arith.divf %6, %cst_0 : f16
      %9 = arith.select %7, %8, %cst : f16
      memref.store %9, %alloc[%5, %4, %2, %0] : memref<4x512x7x7xf16>
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown74(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c100352 = arith.constant 100352 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg2 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg2, %c7 : index
      %1 = arith.divsi %arg2, %c7 : index
      %2 = arith.remsi %1, %c7 : index
      %3 = arith.divsi %1, %c7 : index
      %4 = arith.remsi %3, %c512 : index
      %5 = arith.divsi %3, %c512 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x512x7x7xi1>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x512x7x7xf16>
      %8 = arith.select %6, %7, %cst : f16
      memref.store %8, %alloc[%5, %4, %2, %0] : memref<4x512x7x7xf16>
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown78(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c100352 = arith.constant 100352 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    scf.for %arg3 = %c0 to %c100352 step %c1 {
      %0 = arith.remsi %arg3, %c7 : index
      %1 = arith.divsi %arg3, %c7 : index
      %2 = arith.remsi %1, %c7 : index
      %3 = arith.divsi %1, %c7 : index
      %4 = arith.remsi %3, %c512 : index
      %5 = arith.divsi %3, %c512 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x512x7x7xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x512x7x7xf16>
      %8 = memref.load %arg2[%5, %4, %2, %0] : memref<4x512x7x7xi1>
      %9 = arith.addf %6, %7 : f16
      %10 = arith.select %8, %9, %cst : f16
      memref.store %10, %alloc[%5, %4, %2, %0] : memref<4x512x7x7xf16>
    }
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown89(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c200704 = arith.constant 200704 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    scf.for %arg3 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg3, %c14 : index
      %1 = arith.divsi %arg3, %c14 : index
      %2 = arith.remsi %1, %c14 : index
      %3 = arith.divsi %1, %c14 : index
      %4 = arith.remsi %3, %c256 : index
      %5 = arith.divsi %3, %c256 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x256x14x14xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x256x14x14xf16>
      %8 = memref.load %arg2[%5, %4, %2, %0] : memref<4x256x14x14xi1>
      %9 = arith.addf %6, %7 : f16
      %10 = arith.select %8, %9, %cst : f16
      memref.store %10, %alloc[%5, %4, %2, %0] : memref<4x256x14x14xf16>
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown93(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %c14 = arith.constant 14 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c200704 = arith.constant 200704 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    scf.for %arg2 = %c0 to %c200704 step %c1 {
      %0 = arith.remsi %arg2, %c14 : index
      %1 = arith.divsi %arg2, %c14 : index
      %2 = arith.remsi %1, %c14 : index
      %3 = arith.divsi %1, %c14 : index
      %4 = arith.remsi %3, %c256 : index
      %5 = arith.divsi %3, %c256 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x256x14x14xi1>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x256x14x14xf16>
      %8 = arith.select %6, %7, %cst : f16
      memref.store %8, %alloc[%5, %4, %2, %0] : memref<4x256x14x14xf16>
    }
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown108(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c401408 = arith.constant 401408 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    scf.for %arg3 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg3, %c28 : index
      %1 = arith.divsi %arg3, %c28 : index
      %2 = arith.remsi %1, %c28 : index
      %3 = arith.divsi %1, %c28 : index
      %4 = arith.remsi %3, %c128 : index
      %5 = arith.divsi %3, %c128 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x128x28x28xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x128x28x28xf16>
      %8 = memref.load %arg2[%5, %4, %2, %0] : memref<4x128x28x28xi1>
      %9 = arith.addf %6, %7 : f16
      %10 = arith.select %8, %9, %cst : f16
      memref.store %10, %alloc[%5, %4, %2, %0] : memref<4x128x28x28xf16>
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown112(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c401408 = arith.constant 401408 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    scf.for %arg2 = %c0 to %c401408 step %c1 {
      %0 = arith.remsi %arg2, %c28 : index
      %1 = arith.divsi %arg2, %c28 : index
      %2 = arith.remsi %1, %c28 : index
      %3 = arith.divsi %1, %c28 : index
      %4 = arith.remsi %3, %c128 : index
      %5 = arith.divsi %3, %c128 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x128x28x28xi1>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x128x28x28xf16>
      %8 = arith.select %6, %7, %cst : f16
      memref.store %8, %alloc[%5, %4, %2, %0] : memref<4x128x28x28xf16>
    }
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown127(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c802816 = arith.constant 802816 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg3 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg3, %c56 : index
      %1 = arith.divsi %arg3, %c56 : index
      %2 = arith.remsi %1, %c56 : index
      %3 = arith.divsi %1, %c56 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      %8 = memref.load %arg2[%5, %4, %2, %0] : memref<4x64x56x56xi1>
      %9 = arith.addf %6, %7 : f16
      %10 = arith.select %8, %9, %cst : f16
      memref.store %10, %alloc[%5, %4, %2, %0] : memref<4x64x56x56xf16>
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown131(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c802816 = arith.constant 802816 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg2 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg2, %c56 : index
      %1 = arith.divsi %arg2, %c56 : index
      %2 = arith.remsi %1, %c56 : index
      %3 = arith.divsi %1, %c56 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x64x56x56xi1>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      %8 = arith.select %6, %7, %cst : f16
      memref.store %8, %alloc[%5, %4, %2, %0] : memref<4x64x56x56xf16>
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown143(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c802816 = arith.constant 802816 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    scf.for %arg2 = %c0 to %c802816 step %c1 {
      %0 = arith.remsi %arg2, %c56 : index
      %1 = arith.divsi %arg2, %c56 : index
      %2 = arith.remsi %1, %c56 : index
      %3 = arith.divsi %1, %c56 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      memref.store %8, %alloc[%5, %4, %2, %0] : memref<4x64x56x56xf16>
    }
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown144(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %c112 = arith.constant 112 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c3211264 = arith.constant 3211264 : index
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    scf.for %arg2 = %c0 to %c3211264 step %c1 {
      %0 = arith.remsi %arg2, %c112 : index
      %1 = arith.divsi %arg2, %c112 : index
      %2 = arith.remsi %1, %c112 : index
      %3 = arith.divsi %1, %c112 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<4x64x112x112xi1>
      %7 = memref.load %arg1[%5, %4, %2, %0] : memref<4x64x112x112xf16>
      %8 = arith.select %6, %7, %cst : f16
      memref.store %8, %alloc[%5, %4, %2, %0] : memref<4x64x112x112xf16>
    }
    return %alloc : memref<4x64x112x112xf16>
  }
  func.func private @Unknown147(%arg0: memref<4x1000xf16>, %arg1: memref<4x1000xf32>) -> memref<f32> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c125 = arith.constant 125 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<f32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<4x1000xf16> into memref<4000xf16>
    %collapse_shape_1 = memref.collapse_shape %arg1 [[0, 1]] : memref<4x1000xf32> into memref<4000xf32>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] : memref<4000xf16> into memref<32x125xf16>
    %expand_shape_2 = memref.expand_shape %collapse_shape_1 [[0, 1]] : memref<4000xf32> into memref<32x125xf32>
    %alloc_3 = memref.alloc() : memref<32xf32>
    scf.forall (%arg2) in (32) {
      %subview = memref.subview %expand_shape[%arg2, 0] [1, 125] [1, 1] : memref<32x125xf16> to memref<125xf16, strided<[1], offset: ?>>
      %expand_shape_4 = memref.expand_shape %subview [[0, 1]] : memref<125xf16, strided<[1], offset: ?>> into memref<1x125xf16, strided<[125, 1], offset: ?>>
      %subview_5 = memref.subview %expand_shape_2[%arg2, 0] [1, 125] [1, 1] : memref<32x125xf32> to memref<125xf32, strided<[1], offset: ?>>
      %expand_shape_6 = memref.expand_shape %subview_5 [[0, 1]] : memref<125xf32, strided<[1], offset: ?>> into memref<1x125xf32, strided<[125, 1], offset: ?>>
      %alloca = memref.alloca() : memref<128xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (128) {
        %0 = arith.remsi %arg3, %c128 : index
        %1 = arith.cmpi slt, %0, %c0 : index
        %2 = arith.addi %0, %c128 : index
        %3 = arith.select %1, %2, %0 : index
        %4 = arith.cmpi slt, %3, %c125 : index
        %5 = arith.select %4, %3, %c125 : index
        %6 = arith.addi %3, %c1 : index
        %7 = arith.cmpi slt, %6, %c125 : index
        %8 = arith.select %7, %6, %c125 : index
        %9 = arith.subi %8, %5 : index
        %subview_13 = memref.subview %expand_shape_4[0, %5] [1, %9] [1, 1] : memref<1x125xf16, strided<[125, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %expand_shape_14 = memref.expand_shape %subview_13 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
        %subview_15 = memref.subview %expand_shape_6[0, %5] [1, %9] [1, 1] : memref<1x125xf32, strided<[125, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %expand_shape_16 = memref.expand_shape %subview_15 [[0, 1]] : memref<?xf32, strided<[1], offset: ?>> into memref<1x?xf32, strided<[?, 1], offset: ?>>
        %10 = arith.cmpi ugt, %9, %c0 : index
        %11:2 = scf.if %10 -> (f16, f32) {
          %15 = memref.load %expand_shape_14[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
          %16 = memref.load %expand_shape_16[%c0, %c0] : memref<1x?xf32, strided<[?, 1], offset: ?>>
          scf.yield %15, %16 : f16, f32
        } else {
          scf.yield %cst, %cst_0 : f16, f32
        }
        %12 = arith.extf %11#0 : f16 to f32
        %13 = arith.mulf %12, %11#1 : f32
        %14 = arith.addf %13, %cst_0 : f32
        memref.store %14, %alloca[%arg3] : memref<128xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<64xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (64) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca[%0] : memref<128xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca[%3] : memref<128xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_7[%arg3] : memref<64xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_8 = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (32) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_7[%0] : memref<64xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_7[%3] : memref<64xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_8[%arg3] : memref<32xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_9 = memref.alloca() : memref<16xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (16) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_8[%0] : memref<32xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_8[%3] : memref<32xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_9[%arg3] : memref<16xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_10 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (8) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_9[%0] : memref<16xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_9[%3] : memref<16xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_10[%arg3] : memref<8xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_11 = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (4) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_10[%0] : memref<8xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_10[%3] : memref<8xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_11[%arg3] : memref<4xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_12 = memref.alloca() : memref<2xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (2) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_11[%0] : memref<4xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_11[%3] : memref<4xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_12[%arg3] : memref<2xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg3) in (1) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_12[%0] : memref<2xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_12[%3] : memref<2xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloc_3[%arg2] : memref<32xf32>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    scf.forall (%arg2) in (1) {
      %alloca = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (32) {
        %0 = arith.muli %arg2, %c32 : index
        %1 = arith.addi %0, %arg3 : index
        %2 = memref.load %alloc_3[%1] : memref<32xf32>
        %3 = arith.addf %2, %cst_0 : f32
        memref.store %3, %alloca[%arg3] : memref<32xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_4 = memref.alloca() : memref<16xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (16) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca[%0] : memref<32xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca[%3] : memref<32xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_4[%arg3] : memref<16xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_5 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (8) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_4[%0] : memref<16xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_4[%3] : memref<16xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_5[%arg3] : memref<8xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_6 = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (4) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_5[%0] : memref<8xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_5[%3] : memref<8xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_6[%arg3] : memref<4xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %alloca_7 = memref.alloca() : memref<2xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg3) in (2) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_6[%0] : memref<4xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_6[%3] : memref<4xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloca_7[%arg3] : memref<2xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      scf.forall (%arg3) in (1) {
        %0 = arith.muli %arg3, %c2 : index
        %1 = memref.load %alloca_7[%0] : memref<2xf32, #gpu.address_space<workgroup>>
        %2 = arith.addf %1, %cst_0 : f32
        %3 = arith.addi %0, %c1 : index
        %4 = memref.load %alloca_7[%3] : memref<2xf32, #gpu.address_space<workgroup>>
        %5 = arith.addf %4, %2 : f32
        memref.store %5, %alloc[] : memref<f32>
      } {mapping = [#gpu.thread<x>]}
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<f32>
  }
  func.func private @Unknown148(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.000000e+00 : f32
    %alloc = memref.alloc() : memref<f32>
    %0 = memref.load %arg0[] : memref<f32>
    %1 = arith.negf %0 : f32
    %2 = arith.divf %1, %cst : f32
    memref.store %2, %alloc[] : memref<f32>
    return %alloc : memref<f32>
  }
  func.func private @Unknown149(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c9408 = arith.constant 9408 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf32>
    scf.for %arg1 = %c0 to %c9408 step %c1 {
      %0 = arith.remsi %arg1, %c7 : index
      %1 = arith.divsi %arg1, %c7 : index
      %2 = arith.remsi %1, %c7 : index
      %3 = arith.divsi %1, %c7 : index
      %4 = arith.remsi %3, %c3 : index
      %5 = arith.divsi %3, %c3 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<64x3x7x7xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<64x3x7x7xf32>
    }
    return %alloc : memref<64x3x7x7xf32>
  }
  func.func private @Unknown150(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c36864 = arith.constant 36864 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    scf.for %arg1 = %c0 to %c36864 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<64x64x3x3xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<64x64x3x3xf32>
    }
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown154(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c73728 = arith.constant 73728 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf32>
    scf.for %arg1 = %c0 to %c73728 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c64 : index
      %5 = arith.divsi %3, %c64 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<128x64x3x3xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<128x64x3x3xf32>
    }
    return %alloc : memref<128x64x3x3xf32>
  }
  func.func private @Unknown155(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c147456 = arith.constant 147456 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    scf.for %arg1 = %c0 to %c147456 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c128 : index
      %5 = arith.divsi %3, %c128 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<128x128x3x3xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<128x128x3x3xf32>
    }
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown156(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf32>
    scf.for %arg1 = %c0 to %c8192 step %c1 {
      %0 = arith.remsi %arg1, %c64 : index
      %1 = arith.divsi %arg1, %c64 : index
      %2 = memref.load %arg0[%1, %0, %c0, %c0] : memref<128x64x1x1xf16>
      %3 = arith.extf %2 : f16 to f32
      memref.store %3, %alloc[%1, %0, %c0, %c0] : memref<128x64x1x1xf32>
    }
    return %alloc : memref<128x64x1x1xf32>
  }
  func.func private @Unknown159(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c294912 = arith.constant 294912 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf32>
    scf.for %arg1 = %c0 to %c294912 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c128 : index
      %5 = arith.divsi %3, %c128 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<256x128x3x3xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<256x128x3x3xf32>
    }
    return %alloc : memref<256x128x3x3xf32>
  }
  func.func private @Unknown160(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c589824 = arith.constant 589824 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    scf.for %arg1 = %c0 to %c589824 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c256 : index
      %5 = arith.divsi %3, %c256 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<256x256x3x3xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<256x256x3x3xf32>
    }
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown161(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf32>
    scf.for %arg1 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg1, %c128 : index
      %1 = arith.divsi %arg1, %c128 : index
      %2 = memref.load %arg0[%1, %0, %c0, %c0] : memref<256x128x1x1xf16>
      %3 = arith.extf %2 : f16 to f32
      memref.store %3, %alloc[%1, %0, %c0, %c0] : memref<256x128x1x1xf32>
    }
    return %alloc : memref<256x128x1x1xf32>
  }
  func.func private @Unknown164(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1179648 = arith.constant 1179648 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf32>
    scf.for %arg1 = %c0 to %c1179648 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c256 : index
      %5 = arith.divsi %3, %c256 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<512x256x3x3xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<512x256x3x3xf32>
    }
    return %alloc : memref<512x256x3x3xf32>
  }
  func.func private @Unknown165(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c2359296 = arith.constant 2359296 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    scf.for %arg1 = %c0 to %c2359296 step %c1 {
      %0 = arith.remsi %arg1, %c3 : index
      %1 = arith.divsi %arg1, %c3 : index
      %2 = arith.remsi %1, %c3 : index
      %3 = arith.divsi %1, %c3 : index
      %4 = arith.remsi %3, %c512 : index
      %5 = arith.divsi %3, %c512 : index
      %6 = memref.load %arg0[%5, %4, %2, %0] : memref<512x512x3x3xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %alloc[%5, %4, %2, %0] : memref<512x512x3x3xf32>
    }
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown166(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c131072 = arith.constant 131072 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf32>
    scf.for %arg1 = %c0 to %c131072 step %c1 {
      %0 = arith.remsi %arg1, %c256 : index
      %1 = arith.divsi %arg1, %c256 : index
      %2 = memref.load %arg0[%1, %0, %c0, %c0] : memref<512x256x1x1xf16>
      %3 = arith.extf %2 : f16 to f32
      memref.store %3, %alloc[%1, %0, %c0, %c0] : memref<512x256x1x1xf32>
    }
    return %alloc : memref<512x256x1x1xf32>
  }
  func.func private @Unknown170(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c512000 = arith.constant 512000 : index
    %alloc = memref.alloc() : memref<1000x512xf32>
    scf.for %arg1 = %c0 to %c512000 step %c1 {
      %0 = arith.remsi %arg1, %c512 : index
      %1 = arith.divsi %arg1, %c512 : index
      %2 = memref.load %arg0[%1, %0] : memref<1000x512xf16>
      %3 = arith.extf %2 : f16 to f32
      memref.store %3, %alloc[%1, %0] : memref<1000x512xf32>
    }
    return %alloc : memref<1000x512xf32>
  }
  func.func private @Unknown171(%arg0: memref<4x1000xf16>) -> memref<1000xf32> attributes {__byteir_reduction_fusion__} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-32 = arith.constant -32 : index
    %c1000 = arith.constant 1000 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<1000xf32>
    scf.forall (%arg1) in (32) {
      %0 = arith.muli %arg1, %c-32 : index
      %1 = arith.addi %0, %c1000 : index
      %2 = arith.cmpi slt, %1, %c32 : index
      %3 = arith.select %2, %1, %c32 : index
      %4 = arith.muli %arg1, %c32 : index
      %alloca = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      %alloca_1 = memref.alloca() : memref<2x32xf32, #gpu.address_space<workgroup>>
      scf.forall (%arg2, %arg3) in (2, 32) {
        %5 = arith.cmpi slt, %3, %arg3 : index
        %6 = arith.select %5, %3, %arg3 : index
        %7 = arith.addi %arg3, %c1 : index
        %8 = arith.cmpi slt, %3, %7 : index
        %9 = arith.select %8, %3, %7 : index
        %10 = arith.subi %9, %6 : index
        %11 = arith.cmpi ugt, %10, %c0 : index
        %12 = scf.if %11 -> (f16) {
          %18 = arith.muli %arg2, %c2 : index
          %19 = arith.addi %4, %6 : index
          %20 = memref.load %arg0[%18, %19] : memref<4x1000xf16>
          scf.yield %20 : f16
        } else {
          scf.yield %cst : f16
        }
        %13 = arith.extf %12 : f16 to f32
        %14 = arith.addf %13, %cst_0 : f32
        %15 = scf.if %11 -> (f16) {
          %18 = arith.muli %arg2, %c2 : index
          %19 = arith.addi %18, %c1 : index
          %20 = arith.addi %4, %6 : index
          %21 = memref.load %arg0[%19, %20] : memref<4x1000xf16>
          scf.yield %21 : f16
        } else {
          scf.yield %cst : f16
        }
        %16 = arith.extf %15 : f16 to f32
        %17 = arith.addf %14, %16 : f32
        memref.store %17, %alloca_1[%arg2, %arg3] : memref<2x32xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.forall (%arg2) in (32) {
        %5 = memref.load %alloca_1[%c0, %arg2] : memref<2x32xf32, #gpu.address_space<workgroup>>
        %6 = arith.addf %5, %cst_0 : f32
        %7 = memref.load %alloca_1[%c1, %arg2] : memref<2x32xf32, #gpu.address_space<workgroup>>
        %8 = arith.addf %7, %6 : f32
        memref.store %8, %alloca[%arg2] : memref<32xf32, #gpu.address_space<workgroup>>
      } {mapping = [#gpu.thread<x>]}
      %subview = memref.subview %alloca[0] [%3] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<?xf32, strided<[1]>, #gpu.address_space<workgroup>>
      %subview_2 = memref.subview %alloc[%4] [%3] [1] : memref<1000xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview, %subview_2 : memref<?xf32, strided<[1]>, #gpu.address_space<workgroup>> to memref<?xf32, strided<[1], offset: ?>>
    } {mapping = [#gpu.block<x>]}
    return %alloc : memref<1000xf32>
  }
  func.func private @Unknown172(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1000xf32>
    scf.for %arg1 = %c0 to %c1000 step %c1 {
      %0 = memref.load %arg0[%arg1] : memref<1000xf32>
      %1 = arith.truncf %0 : f32 to f16
      %2 = arith.extf %1 : f16 to f32
      memref.store %2, %alloc[%arg1] : memref<1000xf32>
    }
    return %alloc : memref<1000xf32>
  }
  func.func @main(%arg0: memref<4x3x224x224xf32>, %arg1: memref<4x1000xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64x64x3x3xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<64xf32>, %arg11: memref<64xf32>, %arg12: memref<64x64x3x3xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64xf32>, %arg16: memref<64xf32>, %arg17: memref<64x64x3x3xf32>, %arg18: memref<64xf32>, %arg19: memref<64xf32>, %arg20: memref<64xf32>, %arg21: memref<64xf32>, %arg22: memref<64x64x3x3xf32>, %arg23: memref<64xf32>, %arg24: memref<64xf32>, %arg25: memref<64xf32>, %arg26: memref<64xf32>, %arg27: memref<128x64x3x3xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128xf32>, %arg31: memref<128xf32>, %arg32: memref<128x128x3x3xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<128xf32>, %arg37: memref<128x64x1x1xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128x3x3xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<128xf32>, %arg47: memref<128x128x3x3xf32>, %arg48: memref<128xf32>, %arg49: memref<128xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<256x128x3x3xf32>, %arg53: memref<256xf32>, %arg54: memref<256xf32>, %arg55: memref<256xf32>, %arg56: memref<256xf32>, %arg57: memref<256x256x3x3xf32>, %arg58: memref<256xf32>, %arg59: memref<256xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256x128x1x1xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256x256x3x3xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<256xf32>, %arg71: memref<256xf32>, %arg72: memref<256x256x3x3xf32>, %arg73: memref<256xf32>, %arg74: memref<256xf32>, %arg75: memref<256xf32>, %arg76: memref<256xf32>, %arg77: memref<512x256x3x3xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<512xf32>, %arg81: memref<512xf32>, %arg82: memref<512x512x3x3xf32>, %arg83: memref<512xf32>, %arg84: memref<512xf32>, %arg85: memref<512xf32>, %arg86: memref<512xf32>, %arg87: memref<512x256x1x1xf32>, %arg88: memref<512xf32>, %arg89: memref<512xf32>, %arg90: memref<512xf32>, %arg91: memref<512xf32>, %arg92: memref<512x512x3x3xf32>, %arg93: memref<512xf32>, %arg94: memref<512xf32>, %arg95: memref<512xf32>, %arg96: memref<512xf32>, %arg97: memref<512x512x3x3xf32>, %arg98: memref<512xf32>, %arg99: memref<512xf32>, %arg100: memref<512xf32>, %arg101: memref<512xf32>, %arg102: memref<1000x512xf32>, %arg103: memref<1000xf32>) -> (memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg0) : (memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16>
    %1 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @ConvOp_f16f16_f16(%0, %1, %alloc) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<64x3x7x7xf16>, memref<4x64x112x112xf16>
    %alloc_0 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc, %arg3, %arg4, %alloc_0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf16>
    %2 = call @Unknown3(%arg7) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %3 = call @Unknown3(%arg12) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %4 = call @Unknown3(%arg17) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %5 = call @Unknown3(%arg22) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %6 = call @Unknown7(%arg37) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %7 = call @Unknown8(%arg27) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %8 = call @Unknown9(%arg32) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %9 = call @Unknown9(%arg42) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %10 = call @Unknown9(%arg47) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %11 = call @Unknown12(%arg62) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %12 = call @Unknown13(%arg52) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %13 = call @Unknown14(%arg57) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %14 = call @Unknown14(%arg67) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %15 = call @Unknown14(%arg72) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %16 = call @Unknown17(%arg87) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %17 = call @Unknown18(%arg77) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %18 = call @Unknown19(%arg82) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %19 = call @Unknown19(%arg92) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %20 = call @Unknown19(%arg97) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %21 = call @Unknown22(%arg1) : (memref<4x1000xf32>) -> memref<4x1000xf16>
    %22 = call @Unknown23(%arg102) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    %23 = call @Unknown24(%arg103) : (memref<1000xf32>) -> memref<1000xf16>
    %24 = call @Unknown25(%21) : (memref<4x1000xf16>) -> memref<4xf16>
    %25:2 = call @Unknown26(%alloc_0) : (memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>)
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @PoolMaxOp_f16_f16(%25#0, %alloc_1) {base_dilations = dense<1> : tensor<4xi64>, memory_effects = [1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%alloc_1, %2, %alloc_2) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_2, %arg8, %arg9, %alloc_3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %26:2 = call @Unknown28(%alloc_3) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_4 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%26#0, %3, %alloc_4) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_4, %arg13, %arg14, %alloc_5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %27:2 = call @Unknown30(%alloc_5, %alloc_1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_6 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%27#0, %4, %alloc_6) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_7 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_6, %arg18, %arg19, %alloc_7) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %28:2 = call @Unknown28(%alloc_7) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_8 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvOp_f16f16_f16(%28#0, %5, %alloc_8) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_9 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_8, %arg23, %arg24, %alloc_9) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %29:2 = call @Unknown30(%alloc_9, %27#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %alloc_10 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%29#0, %6, %alloc_10) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x1x1xf16>, memref<4x128x28x28xf16>
    %alloc_11 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_10, %arg38, %arg39, %alloc_11) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %alloc_12 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%29#0, %7, %alloc_12) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_13 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_12, %arg28, %arg29, %alloc_13) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %30:2 = call @Unknown37(%alloc_13) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_14 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%30#0, %8, %alloc_14) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_15 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_14, %arg33, %arg34, %alloc_15) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %31:2 = call @Unknown39(%alloc_15, %alloc_11) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_16 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%31#0, %9, %alloc_16) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_17 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_16, %arg43, %arg44, %alloc_17) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %32:2 = call @Unknown37(%alloc_17) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_18 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvOp_f16f16_f16(%32#0, %10, %alloc_18) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_19 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_18, %arg48, %arg49, %alloc_19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %33:2 = call @Unknown39(%alloc_19, %31#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %alloc_20 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%33#0, %11, %alloc_20) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x1x1xf16>, memref<4x256x14x14xf16>
    %alloc_21 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_20, %arg63, %arg64, %alloc_21) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %alloc_22 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%33#0, %12, %alloc_22) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_23 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_22, %arg53, %arg54, %alloc_23) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %34:2 = call @Unknown46(%alloc_23) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_24 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%34#0, %13, %alloc_24) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_25 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_24, %arg58, %arg59, %alloc_25) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %35:2 = call @Unknown48(%alloc_25, %alloc_21) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_26 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%35#0, %14, %alloc_26) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_27 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_26, %arg68, %arg69, %alloc_27) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %36:2 = call @Unknown46(%alloc_27) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_28 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvOp_f16f16_f16(%36#0, %15, %alloc_28) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_29 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_28, %arg73, %arg74, %alloc_29) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %37:2 = call @Unknown48(%alloc_29, %35#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %alloc_30 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%37#0, %16, %alloc_30) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x1x1xf16>, memref<4x512x7x7xf16>
    %alloc_31 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_30, %arg88, %arg89, %alloc_31) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %alloc_32 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%37#0, %17, %alloc_32) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_33 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_32, %arg78, %arg79, %alloc_33) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %38:2 = call @Unknown55(%alloc_33) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_34 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%38#0, %18, %alloc_34) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_35 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_34, %arg83, %arg84, %alloc_35) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %39:2 = call @Unknown57(%alloc_35, %alloc_31) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_36 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%39#0, %19, %alloc_36) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_37 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_36, %arg93, %arg94, %alloc_37) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %40:2 = call @Unknown55(%alloc_37) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %alloc_38 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvOp_f16f16_f16(%40#0, %20, %alloc_38) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_39 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_38, %arg98, %arg99, %alloc_39) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %41:2 = call @Unknown57(%alloc_39, %39#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %42 = call @Unknown62(%41#0) : (memref<4x512x7x7xf16>) -> memref<4x512xf16>
    %43 = call @Unknown63(%42) : (memref<4x512xf16>) -> memref<4x512xf16>
    %alloc_40 = memref.alloc() : memref<4x1000xf16>
    byre.compute @MatmulOp_f16f16_f16(%43, %22, %alloc_40) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<4x512xf16>, memref<1000x512xf16>, memref<4x1000xf16>
    %44 = call @Unknown64(%23, %alloc_40) : (memref<1000xf16>, memref<4x1000xf16>) -> memref<4x1000xf16>
    %45 = call @Unknown65(%44) : (memref<4x1000xf16>) -> memref<4xf16>
    %46 = call @Unknown66(%45, %44) : (memref<4xf16>, memref<4x1000xf16>) -> memref<4x1000xf16>
    %47 = call @Unknown67(%46) : (memref<4x1000xf16>) -> memref<4xf16>
    %48 = call @Unknown68(%47) : (memref<4xf16>) -> memref<4xf16>
    %49:2 = call @Unknown69(%48, %46, %24, %21) : (memref<4xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>)
    %alloc_41 = memref.alloc() : memref<4x512xf16>
    byre.compute @MatmulOp_f16f16_f16(%49#1, %22, %alloc_41) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<4x1000xf16>, memref<1000x512xf16>, memref<4x512xf16>
    %50 = call @Unknown70(%alloc_41, %41#1) : (memref<4x512xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %alloc_42 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_43 = memref.alloc() : memref<512xf32>
    %alloc_44 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_38, %arg98, %50, %alloc_42, %alloc_43, %alloc_44) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_45 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_42, %20, %alloc_45) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_46 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%40#0, %alloc_42, %alloc_46) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %51 = call @Unknown74(%40#1, %alloc_45) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %alloc_47 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_48 = memref.alloc() : memref<512xf32>
    %alloc_49 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_36, %arg93, %51, %alloc_47, %alloc_48, %alloc_49) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_50 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_47, %19, %alloc_50) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_51 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%39#0, %alloc_47, %alloc_51) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %52 = call @Unknown78(%50, %alloc_50, %39#1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %alloc_52 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_53 = memref.alloc() : memref<512xf32>
    %alloc_54 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_34, %arg83, %52, %alloc_52, %alloc_53, %alloc_54) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_55 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_52, %18, %alloc_55) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %alloc_56 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%38#0, %alloc_52, %alloc_56) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %53 = call @Unknown74(%38#1, %alloc_55) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %alloc_57 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_58 = memref.alloc() : memref<512xf32>
    %alloc_59 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_32, %arg78, %53, %alloc_57, %alloc_58, %alloc_59) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_60 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_57, %17, %alloc_60) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_61 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%37#0, %alloc_57, %alloc_61) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x3x3xf16>
    %alloc_62 = memref.alloc() : memref<4x512x7x7xf16>
    %alloc_63 = memref.alloc() : memref<512xf32>
    %alloc_64 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_30, %arg88, %52, %alloc_62, %alloc_63, %alloc_64) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %alloc_65 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_62, %16, %alloc_65) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x1x1xf16>, memref<4x256x14x14xf16>
    %alloc_66 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%37#0, %alloc_62, %alloc_66) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x1x1xf16>
    %54 = call @Unknown89(%alloc_65, %alloc_60, %37#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %alloc_67 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_68 = memref.alloc() : memref<256xf32>
    %alloc_69 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_28, %arg73, %54, %alloc_67, %alloc_68, %alloc_69) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_70 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_67, %15, %alloc_70) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_71 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%36#0, %alloc_67, %alloc_71) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %55 = call @Unknown93(%36#1, %alloc_70) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %alloc_72 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_73 = memref.alloc() : memref<256xf32>
    %alloc_74 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_26, %arg68, %55, %alloc_72, %alloc_73, %alloc_74) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_75 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_72, %14, %alloc_75) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_76 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%35#0, %alloc_72, %alloc_76) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %56 = call @Unknown89(%54, %alloc_75, %35#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %alloc_77 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_78 = memref.alloc() : memref<256xf32>
    %alloc_79 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_24, %arg58, %56, %alloc_77, %alloc_78, %alloc_79) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_80 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_77, %13, %alloc_80) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %alloc_81 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%34#0, %alloc_77, %alloc_81) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %57 = call @Unknown93(%34#1, %alloc_80) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %alloc_82 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_83 = memref.alloc() : memref<256xf32>
    %alloc_84 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_22, %arg53, %57, %alloc_82, %alloc_83, %alloc_84) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_85 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_82, %12, %alloc_85) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_86 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%33#0, %alloc_82, %alloc_86) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x3x3xf16>
    %alloc_87 = memref.alloc() : memref<4x256x14x14xf16>
    %alloc_88 = memref.alloc() : memref<256xf32>
    %alloc_89 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_20, %arg63, %56, %alloc_87, %alloc_88, %alloc_89) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %alloc_90 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_87, %11, %alloc_90) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x1x1xf16>, memref<4x128x28x28xf16>
    %alloc_91 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%33#0, %alloc_87, %alloc_91) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x1x1xf16>
    %58 = call @Unknown108(%alloc_90, %alloc_85, %33#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %alloc_92 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_93 = memref.alloc() : memref<128xf32>
    %alloc_94 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_18, %arg48, %58, %alloc_92, %alloc_93, %alloc_94) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_95 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_92, %10, %alloc_95) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_96 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%32#0, %alloc_92, %alloc_96) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %59 = call @Unknown112(%32#1, %alloc_95) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %alloc_97 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_98 = memref.alloc() : memref<128xf32>
    %alloc_99 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_16, %arg43, %59, %alloc_97, %alloc_98, %alloc_99) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_100 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_97, %9, %alloc_100) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_101 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%31#0, %alloc_97, %alloc_101) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %60 = call @Unknown108(%58, %alloc_100, %31#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %alloc_102 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_103 = memref.alloc() : memref<128xf32>
    %alloc_104 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_14, %arg33, %60, %alloc_102, %alloc_103, %alloc_104) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_105 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_102, %8, %alloc_105) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %alloc_106 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%30#0, %alloc_102, %alloc_106) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %61 = call @Unknown112(%30#1, %alloc_105) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %alloc_107 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_108 = memref.alloc() : memref<128xf32>
    %alloc_109 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_12, %arg28, %61, %alloc_107, %alloc_108, %alloc_109) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_110 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_107, %7, %alloc_110) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_111 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%29#0, %alloc_107, %alloc_111) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x3x3xf16>
    %alloc_112 = memref.alloc() : memref<4x128x28x28xf16>
    %alloc_113 = memref.alloc() : memref<128xf32>
    %alloc_114 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_10, %arg38, %60, %alloc_112, %alloc_113, %alloc_114) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %alloc_115 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_112, %6, %alloc_115) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x1x1xf16>, memref<4x64x56x56xf16>
    %alloc_116 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%29#0, %alloc_112, %alloc_116) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x1x1xf16>
    %62 = call @Unknown127(%alloc_115, %alloc_110, %29#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %alloc_117 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_118 = memref.alloc() : memref<64xf32>
    %alloc_119 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_8, %arg23, %62, %alloc_117, %alloc_118, %alloc_119) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_120 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_117, %5, %alloc_120) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_121 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%28#0, %alloc_117, %alloc_121) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %63 = call @Unknown131(%28#1, %alloc_120) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_122 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_123 = memref.alloc() : memref<64xf32>
    %alloc_124 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_6, %arg18, %63, %alloc_122, %alloc_123, %alloc_124) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_125 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_122, %4, %alloc_125) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_126 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%27#0, %alloc_122, %alloc_126) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %64 = call @Unknown127(%62, %alloc_125, %27#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %alloc_127 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_128 = memref.alloc() : memref<64xf32>
    %alloc_129 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_4, %arg13, %64, %alloc_127, %alloc_128, %alloc_129) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_130 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_127, %3, %alloc_130) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_131 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%26#0, %alloc_127, %alloc_131) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %65 = call @Unknown131(%26#1, %alloc_130) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_132 = memref.alloc() : memref<4x64x56x56xf16>
    %alloc_133 = memref.alloc() : memref<64xf32>
    %alloc_134 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_2, %arg8, %65, %alloc_132, %alloc_133, %alloc_134) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %alloc_135 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_132, %2, %alloc_135) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %alloc_136 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%alloc_1, %alloc_132, %alloc_136) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %66 = call @Unknown143(%64, %alloc_135) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %alloc_137 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @PoolMaxGradOp_f16f16_f16(%25#0, %66, %alloc_137) {memory_effects = [1 : i32, 1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>, memref<4x64x112x112xf16>
    %67 = call @Unknown144(%25#1, %alloc_137) : (memref<4x64x112x112xi1>, memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16>
    %alloc_138 = memref.alloc() : memref<4x64x112x112xf16>
    %alloc_139 = memref.alloc() : memref<64xf32>
    %alloc_140 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc, %arg3, %67, %alloc_138, %alloc_139, %alloc_140) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x112x112xf16>, memref<64xf32>, memref<4x64x112x112xf16>, memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    %alloc_141 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%0, %alloc_138, %alloc_141) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<4x64x112x112xf16>, memref<64x3x7x7xf16>
    %68 = call @Unknown147(%49#0, %arg1) : (memref<4x1000xf16>, memref<4x1000xf32>) -> memref<f32>
    %69 = call @Unknown148(%68) : (memref<f32>) -> memref<f32>
    %70 = call @Unknown149(%alloc_141) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %71 = call @Unknown150(%alloc_136) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %72 = call @Unknown150(%alloc_131) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %73 = call @Unknown150(%alloc_126) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %74 = call @Unknown150(%alloc_121) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %75 = call @Unknown154(%alloc_111) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %76 = call @Unknown155(%alloc_106) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %77 = call @Unknown156(%alloc_116) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %78 = call @Unknown155(%alloc_101) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %79 = call @Unknown155(%alloc_96) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %80 = call @Unknown159(%alloc_86) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %81 = call @Unknown160(%alloc_81) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %82 = call @Unknown161(%alloc_91) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %83 = call @Unknown160(%alloc_76) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %84 = call @Unknown160(%alloc_71) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %85 = call @Unknown164(%alloc_61) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %86 = call @Unknown165(%alloc_56) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %87 = call @Unknown166(%alloc_66) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %88 = call @Unknown165(%alloc_51) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %89 = call @Unknown165(%alloc_46) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %alloc_142 = memref.alloc() : memref<1000x512xf16>
    byre.compute @MatmulOp_f16f16_f16(%43, %49#1, %alloc_142) {lhs_contracting_dimension = 0 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_transpose, rhs_contracting_dimension = 0 : i64} : memref<4x512xf16>, memref<4x1000xf16>, memref<1000x512xf16>
    %90 = call @Unknown170(%alloc_142) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %91 = call @Unknown171(%49#1) : (memref<4x1000xf16>) -> memref<1000xf32>
    %92 = call @Unknown172(%91) : (memref<1000xf32>) -> memref<1000xf32>
    return %69, %70, %alloc_139, %alloc_140, %71, %alloc_133, %alloc_134, %72, %alloc_128, %alloc_129, %73, %alloc_123, %alloc_124, %74, %alloc_118, %alloc_119, %75, %alloc_108, %alloc_109, %76, %alloc_103, %alloc_104, %77, %alloc_113, %alloc_114, %78, %alloc_98, %alloc_99, %79, %alloc_93, %alloc_94, %80, %alloc_83, %alloc_84, %81, %alloc_78, %alloc_79, %82, %alloc_88, %alloc_89, %83, %alloc_73, %alloc_74, %84, %alloc_68, %alloc_69, %85, %alloc_58, %alloc_59, %86, %alloc_53, %alloc_54, %87, %alloc_63, %alloc_64, %88, %alloc_48, %alloc_49, %89, %alloc_43, %alloc_44, %90, %92 : memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>
  }
}