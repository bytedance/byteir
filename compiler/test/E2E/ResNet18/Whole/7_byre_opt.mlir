// RUN: byteir-opt %s -byre-opt="append-arg-types entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

module @IrToMhlo.2452 attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown164(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) kernel {
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1000 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<1000xf32>
        %7 = arith.truncf %6 : f32 to f16
        %8 = arith.extf %7 : f16 to f32
        memref.store %8, %arg1[%4] : memref<1000xf32>
      }
      gpu.return
    }
    gpu.func @Unknown163(%arg0: memref<1000x512xf16>, %arg1: memref<1000x512xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c512000 = arith.constant 512000 : index
      %c512 = arith.constant 512 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c512000 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c512 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c512 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<1000x512xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9] : memref<1000x512xf32>
      }
      gpu.return
    }
    gpu.func @Unknown161(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c2359296 = arith.constant 2359296 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown160(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c2359296 = arith.constant 2359296 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown159(%arg0: memref<512x256x1x1xf16>, %arg1: memref<512x256x1x1xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c131072 = arith.constant 131072 : index
      %c256 = arith.constant 256 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c131072 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c256 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c256 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c256 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<512x256x1x1xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<512x256x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown158(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c2359296 = arith.constant 2359296 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown157(%arg0: memref<512x256x3x3xf16>, %arg1: memref<512x256x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1179648 = arith.constant 1179648 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1179648 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x256x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x256x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown156(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c589824 = arith.constant 589824 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown155(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c589824 = arith.constant 589824 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown154(%arg0: memref<256x128x1x1xf16>, %arg1: memref<256x128x1x1xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c128 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<256x128x1x1xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<256x128x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown153(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c589824 = arith.constant 589824 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown152(%arg0: memref<256x128x3x3xf16>, %arg1: memref<256x128x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c294912 = arith.constant 294912 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c294912 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x128x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x128x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown151(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c147456 = arith.constant 147456 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown150(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c147456 = arith.constant 147456 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown149(%arg0: memref<128x64x1x1xf16>, %arg1: memref<128x64x1x1xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c8192 = arith.constant 8192 : index
      %c64 = arith.constant 64 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c8192 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c64 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c64 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c64 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<128x64x1x1xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<128x64x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown148(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c147456 = arith.constant 147456 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown147(%arg0: memref<128x64x3x3xf16>, %arg1: memref<128x64x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c73728 = arith.constant 73728 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c73728 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown146(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c36864 = arith.constant 36864 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown145(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c36864 = arith.constant 36864 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown144(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c36864 = arith.constant 36864 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown143(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c36864 = arith.constant 36864 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown142(%arg0: memref<64x3x7x7xf16>, %arg1: memref<64x3x7x7xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c9408 = arith.constant 9408 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c9408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c3 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c3 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c3 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x3x7x7xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x3x7x7xf32>
      }
      gpu.return
    }
    gpu.func @Unknown141(%arg0: memref<f32>, %arg1: memref<f32>) kernel {
      %cst = arith.constant 4.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1 : index
      scf.if %5 {
        %6 = memref.load %arg0[] : memref<f32>
        %7 = arith.negf %6 : f32
        %8 = arith.divf %7, %cst : f32
        memref.store %8, %arg1[] : memref<f32>
      }
      gpu.return
    }
    gpu.func @Unknown138(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>, %arg2: memref<4x64x112x112xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c3211264 = arith.constant 3211264 : index
      %c112 = arith.constant 112 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c112 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c112 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c112 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c112 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c112 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c112 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x112x112xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x64x112x112xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x64x112x112xf16>
      }
      gpu.return
    }
    gpu.func @Unknown137(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown133(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown129(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>, %arg3: memref<4x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<4x64x56x56xi1>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown125(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown121(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>, %arg3: memref<4x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<4x64x56x56xi1>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown114(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c401408 = arith.constant 401408 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c401408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c28 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x128x28x28xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown110(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>, %arg3: memref<4x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c401408 = arith.constant 401408 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c401408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c28 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<4x128x28x28xi1>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown106(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c401408 = arith.constant 401408 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c401408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c28 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x128x28x28xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown102(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>, %arg3: memref<4x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c401408 = arith.constant 401408 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c401408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c28 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<4x128x28x28xi1>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown95(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c14 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x256x14x14xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown91(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>, %arg3: memref<4x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c14 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<4x256x14x14xi1>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown87(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c14 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x256x14x14xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown83(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>, %arg3: memref<4x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c14 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<4x256x14x14xi1>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown76(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x512x7x7xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown72(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>, %arg3: memref<4x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<4x512x7x7xi1>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown68(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x512x7x7xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %38 = arith.select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown64(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>, %arg2: memref<4x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 4.900000e+01 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg1[%35, %29, %19, %9] : memref<4x512x7x7xi1>
        %37 = memref.load %arg0[%35, %29] : memref<4x512xf16>
        %38 = arith.divf %37, %cst_0 : f16
        %39 = arith.select %36, %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown63(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>, %arg4: memref<4x1000xf32>, %arg5: memref<4x1000xf16>, %arg6: memref<4x1000xf32>, %arg7: memref<4x1000xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c4000 = arith.constant 4000 : index
      %c1000 = arith.constant 1000 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c4000 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c1000 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c1000 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c1000 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg3[%15, %9] : memref<4x1000xf16>
        %17 = memref.load %arg1[%15, %9] : memref<4x1000xf16>
        %18 = memref.load %arg0[%15] : memref<4xf16>
        %19 = memref.load %arg2[%15] : memref<4xf16>
        %20 = memref.load %arg4[%15, %9] : memref<4x1000xf32>
        %21 = math.log %18 : f16
        %22 = arith.subf %17, %21 : f16
        %23 = math.exp %22 : f16
        %24 = arith.mulf %23, %19 : f16
        %25 = arith.subf %16, %24 : f16
        %26 = arith.extf %22 : f16 to f32
        %27 = arith.mulf %26, %20 : f32
        %28 = arith.extf %25 : f16 to f32
        memref.store %25, %arg5[%15, %9] : memref<4x1000xf16>
        memref.store %27, %arg6[%15, %9] : memref<4x1000xf32>
        memref.store %28, %arg7[%15, %9] : memref<4x1000xf32>
      }
      gpu.return
    }
    gpu.func @Unknown62(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4x1000xf16>, %arg3: memref<4x1000xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c4000 = arith.constant 4000 : index
      %c1000 = arith.constant 1000 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c4000 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c1000 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c1000 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c1000 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg1[%15, %9] : memref<4x1000xf16>
        %17 = memref.load %arg0[%15] : memref<4xf16>
        %18 = arith.subf %16, %17 : f16
        %19 = math.exp %18 : f16
        memref.store %18, %arg2[%15, %9] : memref<4x1000xf16>
        memref.store %19, %arg3[%15, %9] : memref<4x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown61(%arg0: memref<1000xf32>, %arg1: memref<4x1000xf16>, %arg2: memref<4x1000xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c4000 = arith.constant 4000 : index
      %c1000 = arith.constant 1000 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c4000 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c1000 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c1000 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c1000 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg1[%15, %9] : memref<4x1000xf16>
        %17 = memref.load %arg0[%9] : memref<1000xf32>
        %18 = arith.truncf %17 : f32 to f16
        %19 = arith.addf %16, %18 : f16
        memref.store %19, %arg2[%15, %9] : memref<4x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown60(%arg0: memref<4x512xf16>, %arg1: memref<4x512xf16>) kernel {
      %cst = arith.constant 2.040100e-02 : f16
      %c0 = arith.constant 0 : index
      %c2048 = arith.constant 2048 : index
      %c512 = arith.constant 512 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c2048 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c512 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c512 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<4x512xf16>
        %17 = arith.mulf %16, %cst : f16
        memref.store %17, %arg1[%15, %9] : memref<4x512xf16>
      }
      gpu.return
    }
    gpu.func @Unknown59(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xf16>, %arg3: memref<4x512x7x7xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x512x7x7xi1>
      }
      gpu.return
    }
    gpu.func @Unknown57(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x512x7x7xi1>
      }
      gpu.return
    }
    gpu.func @Unknown55(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xf16>, %arg3: memref<4x512x7x7xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x512x7x7xi1>
      }
      gpu.return
    }
    gpu.func @Unknown53(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x512x7x7xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x512x7x7xi1>
      }
      gpu.return
    }
    gpu.func @Unknown50(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xf16>, %arg3: memref<4x256x14x14xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c14 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x256x14x14xi1>
      }
      gpu.return
    }
    gpu.func @Unknown48(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c14 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x256x14x14xi1>
      }
      gpu.return
    }
    gpu.func @Unknown46(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xf16>, %arg3: memref<4x256x14x14xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c14 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x256x14x14xi1>
      }
      gpu.return
    }
    gpu.func @Unknown44(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c14 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x256x14x14xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x256x14x14xi1>
      }
      gpu.return
    }
    gpu.func @Unknown41(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xf16>, %arg3: memref<4x128x28x28xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c401408 = arith.constant 401408 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c401408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c28 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x128x28x28xi1>
      }
      gpu.return
    }
    gpu.func @Unknown39(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c401408 = arith.constant 401408 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c401408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c28 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x128x28x28xi1>
      }
      gpu.return
    }
    gpu.func @Unknown37(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xf16>, %arg3: memref<4x128x28x28xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c401408 = arith.constant 401408 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c401408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c28 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x128x28x28xi1>
      }
      gpu.return
    }
    gpu.func @Unknown35(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c401408 = arith.constant 401408 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c401408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c28 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x128x28x28xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x128x28x28xi1>
      }
      gpu.return
    }
    gpu.func @Unknown32(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xf16>, %arg3: memref<4x64x56x56xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x64x56x56xi1>
      }
      gpu.return
    }
    gpu.func @Unknown30(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x64x56x56xi1>
      }
      gpu.return
    }
    gpu.func @Unknown28(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xf16>, %arg3: memref<4x64x56x56xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<4x64x56x56xi1>
      }
      gpu.return
    }
    gpu.func @Unknown26(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c56 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x64x56x56xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x64x56x56xi1>
      }
      gpu.return
    }
    gpu.func @Unknown24(%arg0: memref<4x64x112x112xf16>, %arg1: memref<4x64x112x112xf16>, %arg2: memref<4x64x112x112xi1>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c3211264 = arith.constant 3211264 : index
      %c112 = arith.constant 112 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c112 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c112 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c112 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c112 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c112 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c112 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x64x112x112xf16>
        %37 = arith.maxf %36, %cst : f16
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x64x112x112xf16>
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<4x64x112x112xi1>
      }
      gpu.return
    }
    gpu.func @Unknown23(%arg0: memref<1000x512xf32>, %arg1: memref<1000x512xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c512000 = arith.constant 512000 : index
      %c512 = arith.constant 512 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c512000 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c512 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c512 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<1000x512xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9] : memref<1000x512xf16>
      }
      gpu.return
    }
    gpu.func @Unknown22(%arg0: memref<4x1000xf32>, %arg1: memref<4x1000xf16>) kernel {
      %cst = arith.constant -2.500000e-01 : f32
      %c0 = arith.constant 0 : index
      %c4000 = arith.constant 4000 : index
      %c1000 = arith.constant 1000 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c4000 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c1000 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c1000 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c1000 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<4x1000xf32>
        %17 = arith.mulf %16, %cst : f32
        %18 = arith.truncf %17 : f32 to f16
        memref.store %18, %arg1[%15, %9] : memref<4x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown21(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c2359296 = arith.constant 2359296 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown20(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c2359296 = arith.constant 2359296 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown19(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c2359296 = arith.constant 2359296 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown18(%arg0: memref<512x256x3x3xf32>, %arg1: memref<512x256x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c1179648 = arith.constant 1179648 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1179648 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x256x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown17(%arg0: memref<512x256x1x1xf32>, %arg1: memref<512x256x1x1xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c131072 = arith.constant 131072 : index
      %c256 = arith.constant 256 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c131072 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c256 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c256 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c256 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<512x256x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<512x256x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown16(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c589824 = arith.constant 589824 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown15(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c589824 = arith.constant 589824 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown14(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c589824 = arith.constant 589824 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown13(%arg0: memref<256x128x3x3xf32>, %arg1: memref<256x128x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c294912 = arith.constant 294912 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c294912 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x128x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown12(%arg0: memref<256x128x1x1xf32>, %arg1: memref<256x128x1x1xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c128 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<256x128x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<256x128x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown11(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c147456 = arith.constant 147456 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown10(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c147456 = arith.constant 147456 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown9(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c147456 = arith.constant 147456 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown8(%arg0: memref<128x64x3x3xf32>, %arg1: memref<128x64x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c73728 = arith.constant 73728 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c73728 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown7(%arg0: memref<128x64x1x1xf32>, %arg1: memref<128x64x1x1xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c8192 = arith.constant 8192 : index
      %c64 = arith.constant 64 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c8192 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c64 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c64 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c64 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<128x64x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<128x64x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown6(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c36864 = arith.constant 36864 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown5(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c36864 = arith.constant 36864 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c36864 = arith.constant 36864 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown3(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c36864 = arith.constant 36864 : index
      %c3 = arith.constant 3 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c3 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown1(%arg0: memref<64x3x7x7xf32>, %arg1: memref<64x3x7x7xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c9408 = arith.constant 9408 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c9408 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c7 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c3 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c3 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c3 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x3x7x7xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x3x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<4x3x224x224xf32>, %arg1: memref<4x3x224x224xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c602112 = arith.constant 602112 : index
      %c224 = arith.constant 224 : index
      %c-1 = arith.constant -1 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c602112 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c224 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c224 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c224 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c224 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c224 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c224 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.remsi %25, %c3 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c3 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c3 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<4x3x224x224xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<4x3x224x224xf16>
      }
      gpu.return
    }
  }
  func.func private @Unknown0(memref<4x3x224x224xf32, "cuda">) -> memref<4x3x224x224xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown1(memref<64x3x7x7xf32, "cuda">) -> memref<64x3x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 74 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown1", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp2(%arg0: memref<4x64x112x112xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<64xf32, "cuda">) -> memref<4x64x112x112xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x64x112x112xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x112x112xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x112x112xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x112x112xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x112x112xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x64x112x112xf32, "cuda">, memref<4x64x112x112xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x64x112x112xf16, "cuda">
  }
  func.func private @Unknown3(memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown3", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown4(memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown4", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown5(memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown5", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown6(memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown6", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown7(memref<128x64x1x1xf32, "cuda">) -> memref<128x64x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 64 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown7", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown8(memref<128x64x3x3xf32, "cuda">) -> memref<128x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown8", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown9(memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown9", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown10(memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown10", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown11(memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown11", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown12(memref<256x128x1x1xf32, "cuda">) -> memref<256x128x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown12", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown13(memref<256x128x3x3xf32, "cuda">) -> memref<256x128x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown13", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown14(memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown14", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown15(memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown15", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown16(memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown16", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown17(memref<512x256x1x1xf32, "cuda">) -> memref<512x256x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown17", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown18(memref<512x256x3x3xf32, "cuda">) -> memref<512x256x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 9216 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown18", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown19(memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown19", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown20(memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown20", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown21(memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown21", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown22(memref<4x1000xf32, "cuda">) -> memref<4x1000xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown22", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown23(memref<1000x512xf32, "cuda">) -> memref<1000x512xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4000 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown23", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown24(memref<4x64x112x112xf16, "cuda">) -> (memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown24", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp25(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<64xf32, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @Unknown26(memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown26", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp27(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<64xf32, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @Unknown28(memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown28", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp29(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<64xf32, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @Unknown30(memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown30", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp31(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<64xf32, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @Unknown32(memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown32", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp33(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @BatchNormTrainingOp34(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @Unknown35(memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown35", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp36(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @Unknown37(memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown37", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp38(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @Unknown39(memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown39", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp40(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @Unknown41(memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown41", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp42(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @BatchNormTrainingOp43(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @Unknown44(memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown44", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp45(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @Unknown46(memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown46", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp47(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @Unknown48(memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown48", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp49(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @Unknown50(memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown50", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp51(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x512x7x7xf16, "cuda">
  }
  func.func private @BatchNormTrainingOp52(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x512x7x7xf16, "cuda">
  }
  func.func private @Unknown53(memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown53", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp54(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x512x7x7xf16, "cuda">
  }
  func.func private @Unknown55(memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown55", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp56(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x512x7x7xf16, "cuda">
  }
  func.func private @Unknown57(memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown57", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormTrainingOp58(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_1 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_training"(%alloc, %arg1, %arg2, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_3 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_0, %alloc_3) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_3 : memref<4x512x7x7xf16, "cuda">
  }
  func.func private @Unknown59(memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown59", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown60(memref<4x512xf16, "cuda">) -> memref<4x512xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown60", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown61(memref<1000xf32, "cuda">, memref<4x1000xf16, "cuda">) -> memref<4x1000xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown61", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown62(memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">) -> (memref<4x1000xf16, "cuda">, memref<4x1000xf16, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown62", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown63(memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4x1000xf32, "cuda">) -> (memref<4x1000xf16, "cuda">, memref<4x1000xf32, "cuda">, memref<4x1000xf32, "cuda">) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown63", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown64(memref<4x512xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [2 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown64", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp65(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp66(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512x512x3x3xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x512x7x7xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp67(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<4x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x512x3x3xf16, "cuda">
  }
  func.func private @Unknown68(memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown68", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp69(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp70(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512x512x3x3xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x512x7x7xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp71(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<4x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x512x3x3xf16, "cuda">
  }
  func.func private @Unknown72(memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown72", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp73(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp74(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512x512x3x3xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x512x7x7xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp75(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<4x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x512x3x3xf16, "cuda">
  }
  func.func private @Unknown76(memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown76", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp77(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp78(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512x256x3x3xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (memref<512x256x3x3xf16, "cuda">, memref<3x3x256x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x512xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (memref<3x3x256x512xf16, "cuda">, memref<3x3x256x512xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<3x3x256x512xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp79(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<4x512x7x7xf16, "cuda">) -> memref<512x256x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<3x3x256x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x256x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (memref<3x3x256x512xf16, "cuda">, memref<512x256x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x256x3x3xf16, "cuda">
  }
  func.func private @BatchNormGradOp80(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x512x7x7xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp81(%arg0: memref<4x512x7x7xf16, "cuda">, %arg1: memref<512x256x1x1xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x256x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (memref<512x256x1x1xf16, "cuda">, memref<1x1x256x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<1x1x256x512xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_0 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp82(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<4x512x7x7xf16, "cuda">) -> memref<512x256x1x1xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x256x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<1x1x256x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x256x1x1xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (memref<1x1x256x512xf16, "cuda">, memref<512x256x1x1xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x256x1x1xf16, "cuda">
  }
  func.func private @Unknown83(memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown83", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp84(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp85(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256x256x3x3xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp86(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<4x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x256x3x3xf16, "cuda">
  }
  func.func private @Unknown87(memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown87", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp88(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp89(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256x256x3x3xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp90(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<4x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x256x3x3xf16, "cuda">
  }
  func.func private @Unknown91(memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown91", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp92(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp93(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256x256x3x3xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp94(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<4x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x256x3x3xf16, "cuda">
  }
  func.func private @Unknown95(memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown95", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp96(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp97(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256x128x3x3xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (memref<256x128x3x3xf16, "cuda">, memref<3x3x128x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x256xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (memref<3x3x128x256xf16, "cuda">, memref<3x3x128x256xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<3x3x128x256xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp98(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<4x256x14x14xf16, "cuda">) -> memref<256x128x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<3x3x128x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x128x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (memref<3x3x128x256xf16, "cuda">, memref<256x128x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x128x3x3xf16, "cuda">
  }
  func.func private @BatchNormGradOp99(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x256x14x14xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp100(%arg0: memref<4x256x14x14xf16, "cuda">, %arg1: memref<256x128x1x1xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x128x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (memref<256x128x1x1xf16, "cuda">, memref<1x1x128x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<1x1x128x256xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_0 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp101(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<4x256x14x14xf16, "cuda">) -> memref<256x128x1x1xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x128x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<1x1x128x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x128x1x1xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (memref<1x1x128x256xf16, "cuda">, memref<256x128x1x1xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x128x1x1xf16, "cuda">
  }
  func.func private @Unknown102(memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown102", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp103(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp104(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128x128x3x3xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp105(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<4x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x128x3x3xf16, "cuda">
  }
  func.func private @Unknown106(memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown106", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp107(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp108(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128x128x3x3xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp109(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<4x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x128x3x3xf16, "cuda">
  }
  func.func private @Unknown110(memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown110", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp111(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp112(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128x128x3x3xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp113(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<4x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x128x3x3xf16, "cuda">
  }
  func.func private @Unknown114(memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown114", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp115(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp116(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (memref<128x64x3x3xf16, "cuda">, memref<3x3x64x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x128xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (memref<3x3x64x128xf16, "cuda">, memref<3x3x64x128xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<3x3x64x128xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp117(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<4x128x28x28xf16, "cuda">) -> memref<128x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<3x3x64x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (memref<3x3x64x128xf16, "cuda">, memref<128x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x64x3x3xf16, "cuda">
  }
  func.func private @BatchNormGradOp118(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x128x28x28xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp119(%arg0: memref<4x128x28x28xf16, "cuda">, %arg1: memref<128x64x1x1xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x64x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (memref<128x64x1x1xf16, "cuda">, memref<1x1x64x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<1x1x64x128xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_0 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp120(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<4x128x28x28xf16, "cuda">) -> memref<128x64x1x1xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x64x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<1x1x64x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x64x1x1xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (memref<1x1x64x128xf16, "cuda">, memref<128x64x1x1xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x64x1x1xf16, "cuda">
  }
  func.func private @Unknown121(memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown121", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp122(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp123(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp124(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<4x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x64x3x3xf16, "cuda">
  }
  func.func private @Unknown125(memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown125", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp126(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp127(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp128(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<4x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x64x3x3xf16, "cuda">
  }
  func.func private @Unknown129(memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown129", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp130(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp131(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp132(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<4x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x64x3x3xf16, "cuda">
  }
  func.func private @Unknown133(memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown133", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp134(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x64x56x56xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp135(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<64x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<4x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp136(%arg0: memref<4x64x56x56xf16, "cuda">, %arg1: memref<4x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x64x3x3xf16, "cuda">
  }
  func.func private @Unknown137(memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown137", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown138(memref<4x64x112x112xi1, "cuda">, memref<4x64x112x112xf16, "cuda">) -> memref<4x64x112x112xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown138", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp139(%arg0: memref<4x64x112x112xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<4x64x112x112xf16, "cuda">) -> (memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<4x64x112x112xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<4x64x112x112xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<4x64x112x112xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x112x112xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x112x112xf32, "cuda">, memref<4x64x112x112xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x112x112xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<4x64x112x112xf32, "cuda">, memref<4x64x112x112xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardFilterOp140(%arg0: memref<4x3x224x224xf16, "cuda">, %arg1: memref<4x64x112x112xf16, "cuda">) -> memref<64x3x7x7xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<7x7x3x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x3x224x224xf16, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<7x7x3x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x3x7x7xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (memref<7x7x3x64xf16, "cuda">, memref<64x3x7x7xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x3x7x7xf16, "cuda">
  }
  func.func private @Unknown141(memref<f32, "cuda">) -> memref<f32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [0 : i32, 0 : i32], __byre__kernel_name = "Unknown141", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown142(memref<64x3x7x7xf16, "cuda">) -> memref<64x3x7x7xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 74 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown142", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown143(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown143", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown144(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown144", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown145(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown145", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown146(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown146", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown147(memref<128x64x3x3xf16, "cuda">) -> memref<128x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown147", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown148(memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown148", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown149(memref<128x64x1x1xf16, "cuda">) -> memref<128x64x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 64 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown149", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown150(memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown150", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown151(memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown151", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown152(memref<256x128x3x3xf16, "cuda">) -> memref<256x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown152", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown153(memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown153", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown154(memref<256x128x1x1xf16, "cuda">) -> memref<256x128x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown154", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown155(memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown155", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown156(memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown156", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown157(memref<512x256x3x3xf16, "cuda">) -> memref<512x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 9216 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown157", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown158(memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown158", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown159(memref<512x256x1x1xf16, "cuda">) -> memref<512x256x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown159", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown160(memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown160", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown161(memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown161", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @MatmulOp162(%arg0: memref<4x512xf16, "cuda">, %arg1: memref<4x1000xf16, "cuda">) -> memref<1000x512xf16, "cuda"> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512x1000xf16, "cuda">
    "lmhlo.dot"(%arg0, %arg1, %alloc) {device = "cuda", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<512x1000xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1000x512xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (memref<512x1000xf16, "cuda">, memref<1000x512xf16, "cuda">) -> ()
    return %alloc_0 : memref<1000x512xf16, "cuda">
  }
  func.func private @Unknown163(memref<1000x512xf16, "cuda">) -> memref<1000x512xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4000 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown163", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown164(memref<1000xf32, "cuda">) -> memref<1000xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown164", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func @main(%arg0: memref<4x3x224x224xf32, "cuda">, %arg1: memref<4x1000xf32, "cuda">, %arg2: memref<64x3x7x7xf32, "cuda">, %arg3: memref<64xf32, "cuda">, %arg4: memref<64xf32, "cuda">, %arg5: memref<64xf32, "cuda">, %arg6: memref<64xf32, "cuda">, %arg7: memref<64x64x3x3xf32, "cuda">, %arg8: memref<64xf32, "cuda">, %arg9: memref<64xf32, "cuda">, %arg10: memref<64xf32, "cuda">, %arg11: memref<64xf32, "cuda">, %arg12: memref<64x64x3x3xf32, "cuda">, %arg13: memref<64xf32, "cuda">, %arg14: memref<64xf32, "cuda">, %arg15: memref<64xf32, "cuda">, %arg16: memref<64xf32, "cuda">, %arg17: memref<64x64x3x3xf32, "cuda">, %arg18: memref<64xf32, "cuda">, %arg19: memref<64xf32, "cuda">, %arg20: memref<64xf32, "cuda">, %arg21: memref<64xf32, "cuda">, %arg22: memref<64x64x3x3xf32, "cuda">, %arg23: memref<64xf32, "cuda">, %arg24: memref<64xf32, "cuda">, %arg25: memref<64xf32, "cuda">, %arg26: memref<64xf32, "cuda">, %arg27: memref<128x64x3x3xf32, "cuda">, %arg28: memref<128xf32, "cuda">, %arg29: memref<128xf32, "cuda">, %arg30: memref<128xf32, "cuda">, %arg31: memref<128xf32, "cuda">, %arg32: memref<128x128x3x3xf32, "cuda">, %arg33: memref<128xf32, "cuda">, %arg34: memref<128xf32, "cuda">, %arg35: memref<128xf32, "cuda">, %arg36: memref<128xf32, "cuda">, %arg37: memref<128x64x1x1xf32, "cuda">, %arg38: memref<128xf32, "cuda">, %arg39: memref<128xf32, "cuda">, %arg40: memref<128xf32, "cuda">, %arg41: memref<128xf32, "cuda">, %arg42: memref<128x128x3x3xf32, "cuda">, %arg43: memref<128xf32, "cuda">, %arg44: memref<128xf32, "cuda">, %arg45: memref<128xf32, "cuda">, %arg46: memref<128xf32, "cuda">, %arg47: memref<128x128x3x3xf32, "cuda">, %arg48: memref<128xf32, "cuda">, %arg49: memref<128xf32, "cuda">, %arg50: memref<128xf32, "cuda">, %arg51: memref<128xf32, "cuda">, %arg52: memref<256x128x3x3xf32, "cuda">, %arg53: memref<256xf32, "cuda">, %arg54: memref<256xf32, "cuda">, %arg55: memref<256xf32, "cuda">, %arg56: memref<256xf32, "cuda">, %arg57: memref<256x256x3x3xf32, "cuda">, %arg58: memref<256xf32, "cuda">, %arg59: memref<256xf32, "cuda">, %arg60: memref<256xf32, "cuda">, %arg61: memref<256xf32, "cuda">, %arg62: memref<256x128x1x1xf32, "cuda">, %arg63: memref<256xf32, "cuda">, %arg64: memref<256xf32, "cuda">, %arg65: memref<256xf32, "cuda">, %arg66: memref<256xf32, "cuda">, %arg67: memref<256x256x3x3xf32, "cuda">, %arg68: memref<256xf32, "cuda">, %arg69: memref<256xf32, "cuda">, %arg70: memref<256xf32, "cuda">, %arg71: memref<256xf32, "cuda">, %arg72: memref<256x256x3x3xf32, "cuda">, %arg73: memref<256xf32, "cuda">, %arg74: memref<256xf32, "cuda">, %arg75: memref<256xf32, "cuda">, %arg76: memref<256xf32, "cuda">, %arg77: memref<512x256x3x3xf32, "cuda">, %arg78: memref<512xf32, "cuda">, %arg79: memref<512xf32, "cuda">, %arg80: memref<512xf32, "cuda">, %arg81: memref<512xf32, "cuda">, %arg82: memref<512x512x3x3xf32, "cuda">, %arg83: memref<512xf32, "cuda">, %arg84: memref<512xf32, "cuda">, %arg85: memref<512xf32, "cuda">, %arg86: memref<512xf32, "cuda">, %arg87: memref<512x256x1x1xf32, "cuda">, %arg88: memref<512xf32, "cuda">, %arg89: memref<512xf32, "cuda">, %arg90: memref<512xf32, "cuda">, %arg91: memref<512xf32, "cuda">, %arg92: memref<512x512x3x3xf32, "cuda">, %arg93: memref<512xf32, "cuda">, %arg94: memref<512xf32, "cuda">, %arg95: memref<512xf32, "cuda">, %arg96: memref<512xf32, "cuda">, %arg97: memref<512x512x3x3xf32, "cuda">, %arg98: memref<512xf32, "cuda">, %arg99: memref<512xf32, "cuda">, %arg100: memref<512xf32, "cuda">, %arg101: memref<512xf32, "cuda">, %arg102: memref<1000x512xf32, "cuda">, %arg103: memref<1000xf32, "cuda">) -> (memref<f32, "cuda">, memref<64x3x7x7xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<128x64x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x64x1x1xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<256x128x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x128x1x1xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<512x256x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x256x1x1xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1000x512xf32, "cuda">, memref<1000xf32, "cuda">) {
    %alloc = memref.alloc() : memref<f32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<f16, "cuda">
    "lmhlo.constant"(%alloc_0) {device = "cuda", value = dense<0.000000e+00> : tensor<f16>} : (memref<f16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<f16, "cuda">
    "lmhlo.constant"(%alloc_1) {device = "cuda", value = dense<0xFC00> : tensor<f16>} : (memref<f16, "cuda">) -> ()
    %0 = call @Unknown0(%arg0) : (memref<4x3x224x224xf32, "cuda">) -> memref<4x3x224x224xf16, "cuda">
    %1 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32, "cuda">) -> memref<64x3x7x7xf16, "cuda">
    %alloc_2 = memref.alloc() : memref<4x64x112x112xf16, "cuda">
    lmhlo.convolution(%0, %1, %alloc_2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x3x224x224xf16, "cuda">, memref<64x3x7x7xf16, "cuda">, memref<4x64x112x112xf16, "cuda">) -> ()
    %2 = call @BatchNormTrainingOp2(%alloc_2, %arg3, %arg4) : (memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<4x64x112x112xf16, "cuda">
    %3 = call @Unknown3(%arg7) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %4 = call @Unknown4(%arg12) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %5 = call @Unknown5(%arg17) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %6 = call @Unknown6(%arg22) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %7 = call @Unknown7(%arg37) : (memref<128x64x1x1xf32, "cuda">) -> memref<128x64x1x1xf16, "cuda">
    %8 = call @Unknown8(%arg27) : (memref<128x64x3x3xf32, "cuda">) -> memref<128x64x3x3xf16, "cuda">
    %9 = call @Unknown9(%arg32) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %10 = call @Unknown10(%arg42) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %11 = call @Unknown11(%arg47) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %12 = call @Unknown12(%arg62) : (memref<256x128x1x1xf32, "cuda">) -> memref<256x128x1x1xf16, "cuda">
    %13 = call @Unknown13(%arg52) : (memref<256x128x3x3xf32, "cuda">) -> memref<256x128x3x3xf16, "cuda">
    %14 = call @Unknown14(%arg57) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %15 = call @Unknown15(%arg67) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %16 = call @Unknown16(%arg72) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %17 = call @Unknown17(%arg87) : (memref<512x256x1x1xf32, "cuda">) -> memref<512x256x1x1xf16, "cuda">
    %18 = call @Unknown18(%arg77) : (memref<512x256x3x3xf32, "cuda">) -> memref<512x256x3x3xf16, "cuda">
    %19 = call @Unknown19(%arg82) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %20 = call @Unknown20(%arg92) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %21 = call @Unknown21(%arg97) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %22 = call @Unknown22(%arg1) : (memref<4x1000xf32, "cuda">) -> memref<4x1000xf16, "cuda">
    %23 = call @Unknown23(%arg102) : (memref<1000x512xf32, "cuda">) -> memref<1000x512xf16, "cuda">
    %alloc_3 = memref.alloc() : memref<4xf16, "cuda">
    "lmhlo.reduce"(%22, %alloc_0, %alloc_3) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {device = "cuda", dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16, "cuda">, memref<f16, "cuda">, memref<4xf16, "cuda">) -> ()
    %24:2 = call @Unknown24(%2) : (memref<4x64x112x112xf16, "cuda">) -> (memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xi1, "cuda">)
    %alloc_4 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    "lmhlo.reduce_window"(%24#0, %alloc_1, %alloc_4) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      %alloc_32 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg104, %arg105, %alloc_32) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%alloc_32, %arg106) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, device = "cuda", padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<4x64x112x112xf16, "cuda">, memref<f16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%alloc_4, %3, %alloc_5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    %25 = call @BatchNormTrainingOp25(%alloc_5, %arg8, %arg9) : (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %26:2 = call @Unknown26(%25) : (memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">)
    %alloc_6 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%26#0, %4, %alloc_6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    %27 = call @BatchNormTrainingOp27(%alloc_6, %arg13, %arg14) : (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %28:2 = call @Unknown28(%27, %alloc_4) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">)
    %alloc_7 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%28#0, %5, %alloc_7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    %29 = call @BatchNormTrainingOp29(%alloc_7, %arg18, %arg19) : (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %30:2 = call @Unknown30(%29) : (memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">)
    %alloc_8 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    lmhlo.convolution(%30#0, %6, %alloc_8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> ()
    %31 = call @BatchNormTrainingOp31(%alloc_8, %arg23, %arg24) : (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %32:2 = call @Unknown32(%31, %28#0) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">)
    %alloc_9 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%32#0, %7, %alloc_9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    %33 = call @BatchNormTrainingOp33(%alloc_9, %arg38, %arg39) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %alloc_10 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%32#0, %8, %alloc_10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x64x56x56xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    %34 = call @BatchNormTrainingOp34(%alloc_10, %arg28, %arg29) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %35:2 = call @Unknown35(%34) : (memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">)
    %alloc_11 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%35#0, %9, %alloc_11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    %36 = call @BatchNormTrainingOp36(%alloc_11, %arg33, %arg34) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %37:2 = call @Unknown37(%36, %33) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">)
    %alloc_12 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%37#0, %10, %alloc_12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    %38 = call @BatchNormTrainingOp38(%alloc_12, %arg43, %arg44) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %39:2 = call @Unknown39(%38) : (memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">)
    %alloc_13 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    lmhlo.convolution(%39#0, %11, %alloc_13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> ()
    %40 = call @BatchNormTrainingOp40(%alloc_13, %arg48, %arg49) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %41:2 = call @Unknown41(%40, %37#0) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">)
    %alloc_14 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%41#0, %12, %alloc_14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    %42 = call @BatchNormTrainingOp42(%alloc_14, %arg63, %arg64) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %alloc_15 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%41#0, %13, %alloc_15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x128x28x28xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    %43 = call @BatchNormTrainingOp43(%alloc_15, %arg53, %arg54) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %44:2 = call @Unknown44(%43) : (memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">)
    %alloc_16 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%44#0, %14, %alloc_16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    %45 = call @BatchNormTrainingOp45(%alloc_16, %arg58, %arg59) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %46:2 = call @Unknown46(%45, %42) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">)
    %alloc_17 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%46#0, %15, %alloc_17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    %47 = call @BatchNormTrainingOp47(%alloc_17, %arg68, %arg69) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %48:2 = call @Unknown48(%47) : (memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">)
    %alloc_18 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    lmhlo.convolution(%48#0, %16, %alloc_18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> ()
    %49 = call @BatchNormTrainingOp49(%alloc_18, %arg73, %arg74) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %50:2 = call @Unknown50(%49, %46#0) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">)
    %alloc_19 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    lmhlo.convolution(%50#0, %17, %alloc_19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    %51 = call @BatchNormTrainingOp51(%alloc_19, %arg88, %arg89) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %alloc_20 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    lmhlo.convolution(%50#0, %18, %alloc_20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x256x14x14xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    %52 = call @BatchNormTrainingOp52(%alloc_20, %arg78, %arg79) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %53:2 = call @Unknown53(%52) : (memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">)
    %alloc_21 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    lmhlo.convolution(%53#0, %19, %alloc_21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    %54 = call @BatchNormTrainingOp54(%alloc_21, %arg83, %arg84) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %55:2 = call @Unknown55(%54, %51) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">)
    %alloc_22 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    lmhlo.convolution(%55#0, %20, %alloc_22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    %56 = call @BatchNormTrainingOp56(%alloc_22, %arg93, %arg94) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %57:2 = call @Unknown57(%56) : (memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">)
    %alloc_23 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    lmhlo.convolution(%57#0, %21, %alloc_23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> ()
    %58 = call @BatchNormTrainingOp58(%alloc_23, %arg98, %arg99) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %59:2 = call @Unknown59(%58, %55#0) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">)
    %alloc_24 = memref.alloc() : memref<4x512xf16, "cuda">
    "lmhlo.reduce"(%59#0, %alloc_0, %alloc_24) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {device = "cuda", dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<4x512x7x7xf16, "cuda">, memref<f16, "cuda">, memref<4x512xf16, "cuda">) -> ()
    %60 = call @Unknown60(%alloc_24) : (memref<4x512xf16, "cuda">) -> memref<4x512xf16, "cuda">
    %alloc_25 = memref.alloc() : memref<4x1000xf16, "cuda">
    "lmhlo.dot"(%60, %23, %alloc_25) {device = "cuda", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x512xf16, "cuda">, memref<1000x512xf16, "cuda">, memref<4x1000xf16, "cuda">) -> ()
    %61 = call @Unknown61(%arg103, %alloc_25) : (memref<1000xf32, "cuda">, memref<4x1000xf16, "cuda">) -> memref<4x1000xf16, "cuda">
    %alloc_26 = memref.alloc() : memref<4xf16, "cuda">
    "lmhlo.reduce"(%61, %alloc_1, %alloc_26) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.maximum"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {device = "cuda", dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16, "cuda">, memref<f16, "cuda">, memref<4xf16, "cuda">) -> ()
    %62:2 = call @Unknown62(%alloc_26, %61) : (memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">) -> (memref<4x1000xf16, "cuda">, memref<4x1000xf16, "cuda">)
    %alloc_27 = memref.alloc() : memref<4xf16, "cuda">
    "lmhlo.reduce"(%62#1, %alloc_0, %alloc_27) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {device = "cuda", dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16, "cuda">, memref<f16, "cuda">, memref<4xf16, "cuda">) -> ()
    %63:3 = call @Unknown63(%alloc_27, %62#0, %alloc_3, %22, %arg1) : (memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4x1000xf32, "cuda">) -> (memref<4x1000xf16, "cuda">, memref<4x1000xf32, "cuda">, memref<4x1000xf32, "cuda">)
    %alloc_28 = memref.alloc() : memref<4x512xf16, "cuda">
    "lmhlo.dot"(%63#0, %23, %alloc_28) {device = "cuda", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<4x1000xf16, "cuda">, memref<1000x512xf16, "cuda">, memref<4x512xf16, "cuda">) -> ()
    %64 = call @Unknown64(%alloc_28, %59#1) : (memref<4x512xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %65:3 = call @BatchNormGradOp65(%alloc_23, %arg98, %64) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %66 = call @ConvBackwardDataOp66(%65#0, %21) : (memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %67 = call @ConvBackwardFilterOp67(%57#0, %65#0) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %68 = call @Unknown68(%57#1, %66) : (memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %69:3 = call @BatchNormGradOp69(%alloc_22, %arg93, %68) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %70 = call @ConvBackwardDataOp70(%69#0, %20) : (memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %71 = call @ConvBackwardFilterOp71(%55#0, %69#0) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %72 = call @Unknown72(%64, %70, %55#1) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %73:3 = call @BatchNormGradOp73(%alloc_21, %arg83, %72) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %74 = call @ConvBackwardDataOp74(%73#0, %19) : (memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %75 = call @ConvBackwardFilterOp75(%53#0, %73#0) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %76 = call @Unknown76(%53#1, %74) : (memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %77:3 = call @BatchNormGradOp77(%alloc_20, %arg78, %76) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %78 = call @ConvBackwardDataOp78(%77#0, %18) : (memref<4x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %79 = call @ConvBackwardFilterOp79(%50#0, %77#0) : (memref<4x256x14x14xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<512x256x3x3xf16, "cuda">
    %80:3 = call @BatchNormGradOp80(%alloc_19, %arg88, %72) : (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %81 = call @ConvBackwardDataOp81(%80#0, %17) : (memref<4x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %82 = call @ConvBackwardFilterOp82(%50#0, %80#0) : (memref<4x256x14x14xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<512x256x1x1xf16, "cuda">
    %83 = call @Unknown83(%81, %78, %50#1) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %84:3 = call @BatchNormGradOp84(%alloc_18, %arg73, %83) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %85 = call @ConvBackwardDataOp85(%84#0, %16) : (memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %86 = call @ConvBackwardFilterOp86(%48#0, %84#0) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %87 = call @Unknown87(%48#1, %85) : (memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %88:3 = call @BatchNormGradOp88(%alloc_17, %arg68, %87) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %89 = call @ConvBackwardDataOp89(%88#0, %15) : (memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %90 = call @ConvBackwardFilterOp90(%46#0, %88#0) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %91 = call @Unknown91(%83, %89, %46#1) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %92:3 = call @BatchNormGradOp92(%alloc_16, %arg58, %91) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %93 = call @ConvBackwardDataOp93(%92#0, %14) : (memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %94 = call @ConvBackwardFilterOp94(%44#0, %92#0) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %95 = call @Unknown95(%44#1, %93) : (memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %96:3 = call @BatchNormGradOp96(%alloc_15, %arg53, %95) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %97 = call @ConvBackwardDataOp97(%96#0, %13) : (memref<4x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %98 = call @ConvBackwardFilterOp98(%41#0, %96#0) : (memref<4x128x28x28xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<256x128x3x3xf16, "cuda">
    %99:3 = call @BatchNormGradOp99(%alloc_14, %arg63, %91) : (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %100 = call @ConvBackwardDataOp100(%99#0, %12) : (memref<4x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %101 = call @ConvBackwardFilterOp101(%41#0, %99#0) : (memref<4x128x28x28xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<256x128x1x1xf16, "cuda">
    %102 = call @Unknown102(%100, %97, %41#1) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %103:3 = call @BatchNormGradOp103(%alloc_13, %arg48, %102) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %104 = call @ConvBackwardDataOp104(%103#0, %11) : (memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %105 = call @ConvBackwardFilterOp105(%39#0, %103#0) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %106 = call @Unknown106(%39#1, %104) : (memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %107:3 = call @BatchNormGradOp107(%alloc_12, %arg43, %106) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %108 = call @ConvBackwardDataOp108(%107#0, %10) : (memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %109 = call @ConvBackwardFilterOp109(%37#0, %107#0) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %110 = call @Unknown110(%102, %108, %37#1) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %111:3 = call @BatchNormGradOp111(%alloc_11, %arg33, %110) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %112 = call @ConvBackwardDataOp112(%111#0, %9) : (memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %113 = call @ConvBackwardFilterOp113(%35#0, %111#0) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %114 = call @Unknown114(%35#1, %112) : (memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %115:3 = call @BatchNormGradOp115(%alloc_10, %arg28, %114) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %116 = call @ConvBackwardDataOp116(%115#0, %8) : (memref<4x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %117 = call @ConvBackwardFilterOp117(%32#0, %115#0) : (memref<4x64x56x56xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<128x64x3x3xf16, "cuda">
    %118:3 = call @BatchNormGradOp118(%alloc_9, %arg38, %110) : (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %119 = call @ConvBackwardDataOp119(%118#0, %7) : (memref<4x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %120 = call @ConvBackwardFilterOp120(%32#0, %118#0) : (memref<4x64x56x56xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<128x64x1x1xf16, "cuda">
    %121 = call @Unknown121(%119, %116, %32#1) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %122:3 = call @BatchNormGradOp122(%alloc_8, %arg23, %121) : (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %123 = call @ConvBackwardDataOp123(%122#0, %6) : (memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %124 = call @ConvBackwardFilterOp124(%30#0, %122#0) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %125 = call @Unknown125(%30#1, %123) : (memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %126:3 = call @BatchNormGradOp126(%alloc_7, %arg18, %125) : (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %127 = call @ConvBackwardDataOp127(%126#0, %5) : (memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %128 = call @ConvBackwardFilterOp128(%28#0, %126#0) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %129 = call @Unknown129(%121, %127, %28#1) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %130:3 = call @BatchNormGradOp130(%alloc_6, %arg13, %129) : (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %131 = call @ConvBackwardDataOp131(%130#0, %4) : (memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %132 = call @ConvBackwardFilterOp132(%26#0, %130#0) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %133 = call @Unknown133(%26#1, %131) : (memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %134:3 = call @BatchNormGradOp134(%alloc_5, %arg8, %133) : (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %135 = call @ConvBackwardDataOp135(%134#0, %3) : (memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %136 = call @ConvBackwardFilterOp136(%alloc_4, %134#0) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %137 = call @Unknown137(%129, %135) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %alloc_29 = memref.alloc() : memref<4x64x112x112xf16, "cuda">
    "lmhlo.select_and_scatter"(%24#0, %137, %alloc_0, %alloc_29) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %165 = mhlo.compare  GE, %arg104, %arg105 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %165 : tensor<i1>
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %165 = mhlo.add %arg104, %arg105 : tensor<f16>
      mhlo.return %165 : tensor<f16>
    }) {device = "cuda", padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<4x64x112x112xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<f16, "cuda">, memref<4x64x112x112xf16, "cuda">) -> ()
    %138 = call @Unknown138(%24#1, %alloc_29) : (memref<4x64x112x112xi1, "cuda">, memref<4x64x112x112xf16, "cuda">) -> memref<4x64x112x112xf16, "cuda">
    %139:3 = call @BatchNormGradOp139(%alloc_2, %arg3, %138) : (memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x112x112xf16, "cuda">) -> (memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %140 = call @ConvBackwardFilterOp140(%0, %139#0) : (memref<4x3x224x224xf16, "cuda">, memref<4x64x112x112xf16, "cuda">) -> memref<64x3x7x7xf16, "cuda">
    %alloc_30 = memref.alloc() : memref<f32, "cuda">
    "lmhlo.reduce"(%63#1, %alloc, %alloc_30) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<4x1000xf32, "cuda">, memref<f32, "cuda">, memref<f32, "cuda">) -> ()
    %141 = call @Unknown141(%alloc_30) : (memref<f32, "cuda">) -> memref<f32, "cuda">
    %142 = call @Unknown142(%140) : (memref<64x3x7x7xf16, "cuda">) -> memref<64x3x7x7xf32, "cuda">
    %143 = call @Unknown143(%136) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %144 = call @Unknown144(%132) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %145 = call @Unknown145(%128) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %146 = call @Unknown146(%124) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %147 = call @Unknown147(%117) : (memref<128x64x3x3xf16, "cuda">) -> memref<128x64x3x3xf32, "cuda">
    %148 = call @Unknown148(%113) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %149 = call @Unknown149(%120) : (memref<128x64x1x1xf16, "cuda">) -> memref<128x64x1x1xf32, "cuda">
    %150 = call @Unknown150(%109) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %151 = call @Unknown151(%105) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %152 = call @Unknown152(%98) : (memref<256x128x3x3xf16, "cuda">) -> memref<256x128x3x3xf32, "cuda">
    %153 = call @Unknown153(%94) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %154 = call @Unknown154(%101) : (memref<256x128x1x1xf16, "cuda">) -> memref<256x128x1x1xf32, "cuda">
    %155 = call @Unknown155(%90) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %156 = call @Unknown156(%86) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %157 = call @Unknown157(%79) : (memref<512x256x3x3xf16, "cuda">) -> memref<512x256x3x3xf32, "cuda">
    %158 = call @Unknown158(%75) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %159 = call @Unknown159(%82) : (memref<512x256x1x1xf16, "cuda">) -> memref<512x256x1x1xf32, "cuda">
    %160 = call @Unknown160(%71) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %161 = call @Unknown161(%67) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %162 = call @MatmulOp162(%60, %63#0) : (memref<4x512xf16, "cuda">, memref<4x1000xf16, "cuda">) -> memref<1000x512xf16, "cuda">
    %163 = call @Unknown163(%162) : (memref<1000x512xf16, "cuda">) -> memref<1000x512xf32, "cuda">
    %alloc_31 = memref.alloc() : memref<1000xf32, "cuda">
    "lmhlo.reduce"(%63#2, %alloc, %alloc_31) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {device = "cuda", dimensions = dense<0> : tensor<1xi64>} : (memref<4x1000xf32, "cuda">, memref<f32, "cuda">, memref<1000xf32, "cuda">) -> ()
    %164 = call @Unknown164(%alloc_31) : (memref<1000xf32, "cuda">) -> memref<1000xf32, "cuda">
    return %141, %142, %139#1, %139#2, %143, %134#1, %134#2, %144, %130#1, %130#2, %145, %126#1, %126#2, %146, %122#1, %122#2, %147, %115#1, %115#2, %148, %111#1, %111#2, %149, %118#1, %118#2, %150, %107#1, %107#2, %151, %103#1, %103#2, %152, %96#1, %96#2, %153, %92#1, %92#2, %154, %99#1, %99#2, %155, %88#1, %88#2, %156, %84#1, %84#2, %157, %77#1, %77#2, %158, %73#1, %73#2, %159, %80#1, %80#2, %160, %69#1, %69#2, %161, %65#1, %65#2, %163, %164 : memref<f32, "cuda">, memref<64x3x7x7xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<128x64x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x64x1x1xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<256x128x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x128x1x1xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<512x256x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x256x1x1xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1000x512xf32, "cuda">, memref<1000xf32, "cuda">
  }
}