// RUN: byteir-opt %s -byre-opt="append-arg-types" | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown98(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
    gpu.func @Unknown97(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
    gpu.func @Unknown96(%arg0: memref<512x256x1x1xf16>, %arg1: memref<512x256x1x1xf32>) kernel {
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
    gpu.func @Unknown95(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
    gpu.func @Unknown94(%arg0: memref<512x256x3x3xf16>, %arg1: memref<512x256x3x3xf32>) kernel {
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
    gpu.func @Unknown93(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
    gpu.func @Unknown92(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
    gpu.func @Unknown91(%arg0: memref<256x128x1x1xf16>, %arg1: memref<256x128x1x1xf32>) kernel {
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
    gpu.func @Unknown90(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
    gpu.func @Unknown89(%arg0: memref<256x128x3x3xf16>, %arg1: memref<256x128x3x3xf32>) kernel {
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
    gpu.func @Unknown88(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
    gpu.func @Unknown87(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
    gpu.func @Unknown86(%arg0: memref<128x64x1x1xf16>, %arg1: memref<128x64x1x1xf32>) kernel {
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
    gpu.func @Unknown85(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
    gpu.func @Unknown84(%arg0: memref<128x64x3x3xf16>, %arg1: memref<128x64x3x3xf32>) kernel {
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
    gpu.func @Unknown83(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
    gpu.func @Unknown82(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
    gpu.func @Unknown81(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
    gpu.func @Unknown80(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
    gpu.func @Unknown79(%arg0: memref<1000x512xf16>, %arg1: memref<1000x512xf32>) kernel {
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
    gpu.func @Unknown78(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1000 : index
      scf.if %5 {
        %6 = arith.cmpi slt, %4, %c0 : index
        %7 = arith.addi %4, %c1000 : index
        %8 = arith.select %6, %7, %4 : index
        %9 = memref.load %arg0[%c0, %8] : memref<1x1000xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%c0, %8] : memref<1x1000xf32>
      }
      gpu.return
    }
    gpu.func @Unknown77(%arg0: memref<64x3x7x7xf16>, %arg1: memref<64x3x7x7xf32>) kernel {
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
    gpu.func @Unknown74(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>, %arg2: memref<1x64x112x112xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c802816 = arith.constant 802816 : index
      %c112 = arith.constant 112 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x112x112xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x112x112xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x112x112xf16>
      }
      gpu.return
    }
    gpu.func @Unknown73(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown69(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown65(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>, %arg3: memref<1x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
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
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = arith.select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown61(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown57(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>, %arg3: memref<1x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
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
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = arith.select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown50(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown46(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>, %arg3: memref<1x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
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
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = arith.select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown42(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown38(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>, %arg3: memref<1x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
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
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = arith.select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c50176 = arith.constant 50176 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c50176 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown27(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>, %arg3: memref<1x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c50176 = arith.constant 50176 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c50176 : index
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
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = arith.select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown23(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c50176 = arith.constant 50176 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c50176 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown19(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>, %arg3: memref<1x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c50176 = arith.constant 50176 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c50176 : index
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
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = arith.select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown12(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c25088 = arith.constant 25088 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c25088 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown8(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>, %arg3: memref<1x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c25088 = arith.constant 25088 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c25088 : index
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
        %36 = memref.load %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %38 = memref.load %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = arith.select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c25088 = arith.constant 25088 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c25088 : index
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = arith.select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %cst = arith.constant 4.900000e+01 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c25088 = arith.constant 25088 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c25088 : index
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
        %36 = memref.load %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = memref.load %arg0[%35, %29] : memref<1x512xf16>
        %38 = arith.divf %37, %cst : f16
        %39 = arith.cmpf ogt, %36, %cst_0 : f16
        %40 = arith.select %39, %38, %cst_0 : f16
        memref.store %40, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  func.func private @Unknown0(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [2 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c196 = arith.constant 196 : index
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown0 blocks in (%c196, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512xf16>, %arg1 : memref<1x512x7x7xf16>, %alloc : memref<1x512x7x7xf16>)
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @BatchNormGradOp1(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp2(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %alloc_1 : memref<1x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp3(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %alloc_0 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown4(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown4", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c196 = arith.constant 196 : index
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown4 blocks in (%c196, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512x7x7xf16>, %arg1 : memref<1x512x7x7xf16>, %alloc : memref<1x512x7x7xf16>)
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @BatchNormGradOp5(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp6(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %alloc_1 : memref<1x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp7(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %alloc_0 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown8(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown8", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c196 = arith.constant 196 : index
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown8 blocks in (%c196, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512x7x7xf16>, %arg1 : memref<1x512x7x7xf16>, %arg2 : memref<1x512x7x7xf16>, %alloc : memref<1x512x7x7xf16>)
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @BatchNormGradOp9(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp10(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %alloc_1 : memref<1x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp11(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %alloc_0 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown12(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown12", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c196 = arith.constant 196 : index
    %alloc = memref.alloc() : memref<1x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown12 blocks in (%c196, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512x7x7xf16>, %arg1 : memref<1x512x7x7xf16>, %alloc : memref<1x512x7x7xf16>)
    return %alloc : memref<1x512x7x7xf16>
  }
  func.func private @BatchNormGradOp13(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp14(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<3x3x256x512xf16>, memref<1x256x14x14xf16>) -> ()
    return %alloc_1 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp15(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x256x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x256x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %alloc_0 : memref<512x256x3x3xf16>
  }
  func.func private @BatchNormGradOp16(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32>
    %alloc_3 = memref.alloc() : memref<512xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp17(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<1x1x256x512xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<1x1x256x512xf16>, memref<1x256x14x14xf16>) -> ()
    return %alloc_0 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp18(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<1x1x256x512xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<512x256x1x1xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %alloc_0 : memref<512x256x1x1xf16>
  }
  func.func private @Unknown19(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown19", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c392 = arith.constant 392 : index
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    gpu.launch_func  @unified::@Unknown19 blocks in (%c392, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x256x14x14xf16>, %arg1 : memref<1x256x14x14xf16>, %arg2 : memref<1x256x14x14xf16>, %alloc : memref<1x256x14x14xf16>)
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp20(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp21(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %alloc_1 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp22(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %alloc_0 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown23(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown23", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c392 = arith.constant 392 : index
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    gpu.launch_func  @unified::@Unknown23 blocks in (%c392, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x256x14x14xf16>, %arg1 : memref<1x256x14x14xf16>, %alloc : memref<1x256x14x14xf16>)
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp24(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp25(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %alloc_1 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp26(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %alloc_0 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown27(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown27", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c392 = arith.constant 392 : index
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    gpu.launch_func  @unified::@Unknown27 blocks in (%c392, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x256x14x14xf16>, %arg1 : memref<1x256x14x14xf16>, %arg2 : memref<1x256x14x14xf16>, %alloc : memref<1x256x14x14xf16>)
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp28(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp29(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %alloc_1 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp30(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %alloc_0 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown31", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c392 = arith.constant 392 : index
    %alloc = memref.alloc() : memref<1x256x14x14xf16>
    gpu.launch_func  @unified::@Unknown31 blocks in (%c392, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x256x14x14xf16>, %arg1 : memref<1x256x14x14xf16>, %alloc : memref<1x256x14x14xf16>)
    return %alloc : memref<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp32(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp33(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<3x3x128x256xf16>, memref<1x128x28x28xf16>) -> ()
    return %alloc_1 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp34(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x128x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x128x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %alloc_0 : memref<256x128x3x3xf16>
  }
  func.func private @BatchNormGradOp35(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32>
    %alloc_3 = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp36(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<1x1x128x256xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x1x128x256xf16>, memref<1x128x28x28xf16>) -> ()
    return %alloc_0 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp37(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<1x1x128x256xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<256x128x1x1xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %alloc_0 : memref<256x128x1x1xf16>
  }
  func.func private @Unknown38(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown38", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c784 = arith.constant 784 : index
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    gpu.launch_func  @unified::@Unknown38 blocks in (%c784, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128x28x28xf16>, %arg1 : memref<1x128x28x28xf16>, %arg2 : memref<1x128x28x28xf16>, %alloc : memref<1x128x28x28xf16>)
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp39(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp40(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %alloc_1 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp41(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %alloc_0 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown42(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown42", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c784 = arith.constant 784 : index
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    gpu.launch_func  @unified::@Unknown42 blocks in (%c784, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128x28x28xf16>, %arg1 : memref<1x128x28x28xf16>, %alloc : memref<1x128x28x28xf16>)
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp43(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp44(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %alloc_1 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp45(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %alloc_0 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown46(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown46", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c784 = arith.constant 784 : index
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    gpu.launch_func  @unified::@Unknown46 blocks in (%c784, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128x28x28xf16>, %arg1 : memref<1x128x28x28xf16>, %arg2 : memref<1x128x28x28xf16>, %alloc : memref<1x128x28x28xf16>)
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp47(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp48(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %alloc_1 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp49(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %alloc_0 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown50(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown50", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c784 = arith.constant 784 : index
    %alloc = memref.alloc() : memref<1x128x28x28xf16>
    gpu.launch_func  @unified::@Unknown50 blocks in (%c784, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128x28x28xf16>, %arg1 : memref<1x128x28x28xf16>, %alloc : memref<1x128x28x28xf16>)
    return %alloc : memref<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp51(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp52(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<3x3x64x128xf16>, memref<1x64x56x56xf16>) -> ()
    return %alloc_1 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp53(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %alloc_0 : memref<128x64x3x3xf16>
  }
  func.func private @BatchNormGradOp54(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32>
    %alloc_3 = memref.alloc() : memref<128xf32>
    %alloc_4 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp55(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<1x1x64x128xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x1x64x128xf16>, memref<1x64x56x56xf16>) -> ()
    return %alloc_0 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp56(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<1x1x64x128xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<128x64x1x1xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %alloc_0 : memref<128x64x1x1xf16>
  }
  func.func private @Unknown57(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown57", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1568 = arith.constant 1568 : index
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown57 blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %arg1 : memref<1x64x56x56xf16>, %arg2 : memref<1x64x56x56xf16>, %alloc : memref<1x64x56x56xf16>)
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp58(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x56x56xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp59(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %alloc_1 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp60(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %alloc_0 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown61(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown61", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1568 = arith.constant 1568 : index
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown61 blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %arg1 : memref<1x64x56x56xf16>, %alloc : memref<1x64x56x56xf16>)
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp62(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x56x56xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp63(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %alloc_1 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp64(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %alloc_0 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown65(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown65", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1568 = arith.constant 1568 : index
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown65 blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %arg1 : memref<1x64x56x56xf16>, %arg2 : memref<1x64x56x56xf16>, %alloc : memref<1x64x56x56xf16>)
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp66(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x56x56xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp67(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %alloc_1 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp68(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %alloc_0 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown69(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown69", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1568 = arith.constant 1568 : index
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown69 blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %arg1 : memref<1x64x56x56xf16>, %alloc : memref<1x64x56x56xf16>)
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp70(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x56x56xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp71(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %alloc) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%alloc, %alloc_0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %alloc_1 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp72(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %alloc_0 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown73(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown73", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1568 = arith.constant 1568 : index
    %alloc = memref.alloc() : memref<1x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown73 blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %arg1 : memref<1x64x56x56xf16>, %alloc : memref<1x64x56x56xf16>)
    return %alloc : memref<1x64x56x56xf16>
  }
  func.func private @Unknown74(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown74", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c6272 = arith.constant 6272 : index
    %alloc = memref.alloc() : memref<1x64x112x112xf16>
    gpu.launch_func  @unified::@Unknown74 blocks in (%c6272, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x112x112xf16>, %arg1 : memref<1x64x112x112xf16>, %alloc : memref<1x64x112x112xf16>)
    return %alloc : memref<1x64x112x112xf16>
  }
  func.func private @BatchNormGradOp75(%arg0: memref<1x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x112x112xf16>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %alloc = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg0, %alloc_0) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg2, %alloc_1) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x112x112xf32>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %alloc_4 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf32>, memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x112x112xf16>
    "lmhlo.convert"(%alloc_2, %alloc_5) : (memref<1x64x112x112xf32>, memref<1x64x112x112xf16>) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardFilterOp76(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %alloc = memref.alloc() : memref<7x7x3x64xf16>
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    %alloc_0 = memref.alloc() : memref<64x3x7x7xf16>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %alloc_0 : memref<64x3x7x7xf16>
  }
  func.func private @Unknown77(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 74 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown77", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c74 = arith.constant 74 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf32>
    gpu.launch_func  @unified::@Unknown77 blocks in (%c74, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x3x7x7xf16>, %alloc : memref<64x3x7x7xf32>)
    return %alloc : memref<64x3x7x7xf32>
  }
  func.func private @Unknown78(%arg0: memref<1x1000xf16>) -> memref<1x1000xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown78", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %alloc = memref.alloc() : memref<1x1000xf32>
    gpu.launch_func  @unified::@Unknown78 blocks in (%c8, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x1000xf16>, %alloc : memref<1x1000xf32>)
    return %alloc : memref<1x1000xf32>
  }
  func.func private @Unknown79(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4000 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown79", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c4000 = arith.constant 4000 : index
    %alloc = memref.alloc() : memref<1000x512xf32>
    gpu.launch_func  @unified::@Unknown79 blocks in (%c4000, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1000x512xf16>, %alloc : memref<1000x512xf32>)
    return %alloc : memref<1000x512xf32>
  }
  func.func private @Unknown80(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown80", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c288 = arith.constant 288 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @unified::@Unknown80 blocks in (%c288, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x64x3x3xf16>, %alloc : memref<64x64x3x3xf32>)
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown81(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown81", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c288 = arith.constant 288 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @unified::@Unknown81 blocks in (%c288, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x64x3x3xf16>, %alloc : memref<64x64x3x3xf32>)
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown82(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown82", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c288 = arith.constant 288 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @unified::@Unknown82 blocks in (%c288, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x64x3x3xf16>, %alloc : memref<64x64x3x3xf32>)
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown83(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown83", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c288 = arith.constant 288 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @unified::@Unknown83 blocks in (%c288, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x64x3x3xf16>, %alloc : memref<64x64x3x3xf32>)
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown84(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown84", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c576 = arith.constant 576 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf32>
    gpu.launch_func  @unified::@Unknown84 blocks in (%c576, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x64x3x3xf16>, %alloc : memref<128x64x3x3xf32>)
    return %alloc : memref<128x64x3x3xf32>
  }
  func.func private @Unknown85(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown85", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1152 = arith.constant 1152 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    gpu.launch_func  @unified::@Unknown85 blocks in (%c1152, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x128x3x3xf16>, %alloc : memref<128x128x3x3xf32>)
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown86(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 64 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown86", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf32>
    gpu.launch_func  @unified::@Unknown86 blocks in (%c64, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x64x1x1xf16>, %alloc : memref<128x64x1x1xf32>)
    return %alloc : memref<128x64x1x1xf32>
  }
  func.func private @Unknown87(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown87", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1152 = arith.constant 1152 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    gpu.launch_func  @unified::@Unknown87 blocks in (%c1152, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x128x3x3xf16>, %alloc : memref<128x128x3x3xf32>)
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown88(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown88", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1152 = arith.constant 1152 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    gpu.launch_func  @unified::@Unknown88 blocks in (%c1152, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x128x3x3xf16>, %alloc : memref<128x128x3x3xf32>)
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown89(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown89", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2304 = arith.constant 2304 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf32>
    gpu.launch_func  @unified::@Unknown89 blocks in (%c2304, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x128x3x3xf16>, %alloc : memref<256x128x3x3xf32>)
    return %alloc : memref<256x128x3x3xf32>
  }
  func.func private @Unknown90(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown90", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c4608 = arith.constant 4608 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    gpu.launch_func  @unified::@Unknown90 blocks in (%c4608, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x256x3x3xf16>, %alloc : memref<256x256x3x3xf32>)
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown91(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown91", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf32>
    gpu.launch_func  @unified::@Unknown91 blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x128x1x1xf16>, %alloc : memref<256x128x1x1xf32>)
    return %alloc : memref<256x128x1x1xf32>
  }
  func.func private @Unknown92(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown92", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c4608 = arith.constant 4608 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    gpu.launch_func  @unified::@Unknown92 blocks in (%c4608, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x256x3x3xf16>, %alloc : memref<256x256x3x3xf32>)
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown93(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown93", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c4608 = arith.constant 4608 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    gpu.launch_func  @unified::@Unknown93 blocks in (%c4608, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x256x3x3xf16>, %alloc : memref<256x256x3x3xf32>)
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown94(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 9216 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown94", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c9216 = arith.constant 9216 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf32>
    gpu.launch_func  @unified::@Unknown94 blocks in (%c9216, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x256x3x3xf16>, %alloc : memref<512x256x3x3xf32>)
    return %alloc : memref<512x256x3x3xf32>
  }
  func.func private @Unknown95(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown95", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c18432 = arith.constant 18432 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    gpu.launch_func  @unified::@Unknown95 blocks in (%c18432, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x512x3x3xf16>, %alloc : memref<512x512x3x3xf32>)
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown96(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown96", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf32>
    gpu.launch_func  @unified::@Unknown96 blocks in (%c1024, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x256x1x1xf16>, %alloc : memref<512x256x1x1xf32>)
    return %alloc : memref<512x256x1x1xf32>
  }
  func.func private @Unknown97(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown97", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c18432 = arith.constant 18432 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    gpu.launch_func  @unified::@Unknown97 blocks in (%c18432, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x512x3x3xf16>, %alloc : memref<512x512x3x3xf32>)
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown98(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown98", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c18432 = arith.constant 18432 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    gpu.launch_func  @unified::@Unknown98 blocks in (%c18432, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x512x3x3xf16>, %alloc : memref<512x512x3x3xf32>)
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func @main(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<128xf32>, %arg11: memref<128xf32>, %arg12: memref<128xf32>, %arg13: memref<128xf32>, %arg14: memref<128xf32>, %arg15: memref<128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<128xf32>, %arg20: memref<256xf32>, %arg21: memref<256xf32>, %arg22: memref<256xf32>, %arg23: memref<256xf32>, %arg24: memref<256xf32>, %arg25: memref<256xf32>, %arg26: memref<256xf32>, %arg27: memref<256xf32>, %arg28: memref<256xf32>, %arg29: memref<256xf32>, %arg30: memref<512xf32>, %arg31: memref<512xf32>, %arg32: memref<512xf32>, %arg33: memref<512xf32>, %arg34: memref<512xf32>, %arg35: memref<512xf32>, %arg36: memref<512xf32>, %arg37: memref<512xf32>, %arg38: memref<512xf32>, %arg39: memref<512xf32>, %arg40: memref<64xf32>, %arg41: memref<64xf32>, %arg42: memref<64xf32>, %arg43: memref<64xf32>, %arg44: memref<64xf32>, %arg45: memref<64xf32>, %arg46: memref<64xf32>, %arg47: memref<64xf32>, %arg48: memref<64xf32>, %arg49: memref<64xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<128xf32>, %arg53: memref<128xf32>, %arg54: memref<128xf32>, %arg55: memref<128xf32>, %arg56: memref<128xf32>, %arg57: memref<128xf32>, %arg58: memref<128xf32>, %arg59: memref<128xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<512xf32>, %arg71: memref<512xf32>, %arg72: memref<512xf32>, %arg73: memref<512xf32>, %arg74: memref<512xf32>, %arg75: memref<512xf32>, %arg76: memref<512xf32>, %arg77: memref<512xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<64x3x7x7xf16>, %arg81: memref<1x3x224x224xf16>, %arg82: memref<1x64x112x112xf16>, %arg83: memref<1x64x112x112xf16>, %arg84: memref<1x64x56x56xf16>, %arg85: memref<64x64x3x3xf16>, %arg86: memref<1x64x56x56xf16>, %arg87: memref<1x64x56x56xf16>, %arg88: memref<64x64x3x3xf16>, %arg89: memref<1x64x56x56xf16>, %arg90: memref<1x64x56x56xf16>, %arg91: memref<64x64x3x3xf16>, %arg92: memref<1x64x56x56xf16>, %arg93: memref<1x64x56x56xf16>, %arg94: memref<64x64x3x3xf16>, %arg95: memref<1x64x56x56xf16>, %arg96: memref<1x64x56x56xf16>, %arg97: memref<128x64x3x3xf16>, %arg98: memref<1x128x28x28xf16>, %arg99: memref<1x128x28x28xf16>, %arg100: memref<128x128x3x3xf16>, %arg101: memref<1x128x28x28xf16>, %arg102: memref<128x64x1x1xf16>, %arg103: memref<1x128x28x28xf16>, %arg104: memref<1x128x28x28xf16>, %arg105: memref<128x128x3x3xf16>, %arg106: memref<1x128x28x28xf16>, %arg107: memref<1x128x28x28xf16>, %arg108: memref<128x128x3x3xf16>, %arg109: memref<1x128x28x28xf16>, %arg110: memref<1x128x28x28xf16>, %arg111: memref<256x128x3x3xf16>, %arg112: memref<1x256x14x14xf16>, %arg113: memref<1x256x14x14xf16>, %arg114: memref<256x256x3x3xf16>, %arg115: memref<1x256x14x14xf16>, %arg116: memref<256x128x1x1xf16>, %arg117: memref<1x256x14x14xf16>, %arg118: memref<1x256x14x14xf16>, %arg119: memref<256x256x3x3xf16>, %arg120: memref<1x256x14x14xf16>, %arg121: memref<1x256x14x14xf16>, %arg122: memref<256x256x3x3xf16>, %arg123: memref<1x256x14x14xf16>, %arg124: memref<1x256x14x14xf16>, %arg125: memref<512x256x3x3xf16>, %arg126: memref<1x512x7x7xf16>, %arg127: memref<1x512x7x7xf16>, %arg128: memref<512x512x3x3xf16>, %arg129: memref<1x512x7x7xf16>, %arg130: memref<512x256x1x1xf16>, %arg131: memref<1x512x7x7xf16>, %arg132: memref<1x512x7x7xf16>, %arg133: memref<512x512x3x3xf16>, %arg134: memref<1x512x7x7xf16>, %arg135: memref<1x512x7x7xf16>, %arg136: memref<512x512x3x3xf16>, %arg137: memref<1x512x7x7xf16>, %arg138: memref<1x512x7x7xf16>, %arg139: memref<1x512xf16>, %arg140: memref<512x1000xf16>, %arg141: memref<1x1000xf16>) -> (memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>) {
    %alloc = memref.alloc() : memref<f32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %alloc_0 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%alloc_0) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    %alloc_1 = memref.alloc() : memref<1x512xf16>
    "lmhlo.dot"(%arg141, %arg140, %alloc_1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x1000xf16>, memref<512x1000xf16>, memref<1x512xf16>) -> ()
    %0 = call @Unknown0(%alloc_1, %arg138) : (memref<1x512xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %1:3 = call @BatchNormGradOp1(%arg137, %arg39, %0) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %2 = call @ConvBackwardDataOp2(%1#0, %arg136) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %3 = call @ConvBackwardFilterOp3(%arg135, %1#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %4 = call @Unknown4(%arg135, %2) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %5:3 = call @BatchNormGradOp5(%arg134, %arg37, %4) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %6 = call @ConvBackwardDataOp6(%5#0, %arg133) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %7 = call @ConvBackwardFilterOp7(%arg132, %5#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %8 = call @Unknown8(%0, %6, %arg132) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %9:3 = call @BatchNormGradOp9(%arg129, %arg33, %8) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %10 = call @ConvBackwardDataOp10(%9#0, %arg128) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %11 = call @ConvBackwardFilterOp11(%arg127, %9#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %12 = call @Unknown12(%arg127, %10) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %13:3 = call @BatchNormGradOp13(%arg126, %arg31, %12) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %14 = call @ConvBackwardDataOp14(%13#0, %arg125) : (memref<1x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %15 = call @ConvBackwardFilterOp15(%arg124, %13#0) : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %16:3 = call @BatchNormGradOp16(%arg131, %arg35, %8) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %17 = call @ConvBackwardDataOp17(%16#0, %arg130) : (memref<1x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<1x256x14x14xf16>
    %18 = call @ConvBackwardFilterOp18(%arg124, %16#0) : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %19 = call @Unknown19(%17, %14, %arg124) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %20:3 = call @BatchNormGradOp20(%arg123, %arg29, %19) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %21 = call @ConvBackwardDataOp21(%20#0, %arg122) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %22 = call @ConvBackwardFilterOp22(%arg121, %20#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %23 = call @Unknown23(%arg121, %21) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %24:3 = call @BatchNormGradOp24(%arg120, %arg27, %23) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %25 = call @ConvBackwardDataOp25(%24#0, %arg119) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %26 = call @ConvBackwardFilterOp26(%arg118, %24#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %27 = call @Unknown27(%19, %25, %arg118) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %28:3 = call @BatchNormGradOp28(%arg115, %arg23, %27) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %29 = call @ConvBackwardDataOp29(%28#0, %arg114) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %30 = call @ConvBackwardFilterOp30(%arg113, %28#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %31 = call @Unknown31(%arg113, %29) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %32:3 = call @BatchNormGradOp32(%arg112, %arg21, %31) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %33 = call @ConvBackwardDataOp33(%32#0, %arg111) : (memref<1x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %34 = call @ConvBackwardFilterOp34(%arg110, %32#0) : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %35:3 = call @BatchNormGradOp35(%arg117, %arg25, %27) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %36 = call @ConvBackwardDataOp36(%35#0, %arg116) : (memref<1x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<1x128x28x28xf16>
    %37 = call @ConvBackwardFilterOp37(%arg110, %35#0) : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %38 = call @Unknown38(%36, %33, %arg110) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %39:3 = call @BatchNormGradOp39(%arg109, %arg19, %38) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %40 = call @ConvBackwardDataOp40(%39#0, %arg108) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %41 = call @ConvBackwardFilterOp41(%arg107, %39#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %42 = call @Unknown42(%arg107, %40) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %43:3 = call @BatchNormGradOp43(%arg106, %arg17, %42) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %44 = call @ConvBackwardDataOp44(%43#0, %arg105) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %45 = call @ConvBackwardFilterOp45(%arg104, %43#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %46 = call @Unknown46(%38, %44, %arg104) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %47:3 = call @BatchNormGradOp47(%arg101, %arg13, %46) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %48 = call @ConvBackwardDataOp48(%47#0, %arg100) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %49 = call @ConvBackwardFilterOp49(%arg99, %47#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %50 = call @Unknown50(%arg99, %48) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %51:3 = call @BatchNormGradOp51(%arg98, %arg11, %50) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %52 = call @ConvBackwardDataOp52(%51#0, %arg97) : (memref<1x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %53 = call @ConvBackwardFilterOp53(%arg96, %51#0) : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %54:3 = call @BatchNormGradOp54(%arg103, %arg15, %46) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %55 = call @ConvBackwardDataOp55(%54#0, %arg102) : (memref<1x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<1x64x56x56xf16>
    %56 = call @ConvBackwardFilterOp56(%arg96, %54#0) : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %57 = call @Unknown57(%55, %52, %arg96) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %58:3 = call @BatchNormGradOp58(%arg95, %arg9, %57) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %59 = call @ConvBackwardDataOp59(%58#0, %arg94) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %60 = call @ConvBackwardFilterOp60(%arg93, %58#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %61 = call @Unknown61(%arg93, %59) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %62:3 = call @BatchNormGradOp62(%arg92, %arg7, %61) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %63 = call @ConvBackwardDataOp63(%62#0, %arg91) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %64 = call @ConvBackwardFilterOp64(%arg90, %62#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %65 = call @Unknown65(%57, %63, %arg90) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %66:3 = call @BatchNormGradOp66(%arg89, %arg5, %65) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %67 = call @ConvBackwardDataOp67(%66#0, %arg88) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %68 = call @ConvBackwardFilterOp68(%arg87, %66#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %69 = call @Unknown69(%arg87, %67) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %70:3 = call @BatchNormGradOp70(%arg86, %arg3, %69) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %71 = call @ConvBackwardDataOp71(%70#0, %arg85) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %72 = call @ConvBackwardFilterOp72(%arg84, %70#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %73 = call @Unknown73(%65, %71) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %alloc_2 = memref.alloc() : memref<1x64x112x112xf16>
    "lmhlo.select_and_scatter"(%arg83, %73, %alloc_0, %alloc_2) ({
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %99 = mhlo.compare  GE, %arg142, %arg143 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %99 : tensor<i1>
    }, {
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %99 = mhlo.add %arg142, %arg143 : tensor<f16>
      mhlo.return %99 : tensor<f16>
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<f16>, memref<1x64x112x112xf16>) -> ()
    %74 = call @Unknown74(%arg83, %alloc_2) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    %75:3 = call @BatchNormGradOp75(%arg82, %arg1, %74) : (memref<1x64x112x112xf16>, memref<64xf32>, memref<1x64x112x112xf16>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %76 = call @ConvBackwardFilterOp76(%arg81, %75#0) : (memref<1x3x224x224xf16>, memref<1x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %77 = call @Unknown77(%76) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %78 = call @Unknown78(%arg141) : (memref<1x1000xf16>) -> memref<1x1000xf32>
    %alloc_3 = memref.alloc() : memref<1000xf32>
    "lmhlo.reduce"(%78, %alloc, %alloc_3) ({
    ^bb0(%arg142: memref<f32>, %arg143: memref<f32>, %arg144: memref<f32>):
      "lmhlo.add"(%arg142, %arg143, %arg144) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<1x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %alloc_4 = memref.alloc() : memref<1000x1xf16>
    "lmhlo.reshape"(%arg141, %alloc_4) : (memref<1x1000xf16>, memref<1000x1xf16>) -> ()
    %alloc_5 = memref.alloc() : memref<1000x512xf16>
    "lmhlo.dot"(%alloc_4, %arg139, %alloc_5) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1000x1xf16>, memref<1x512xf16>, memref<1000x512xf16>) -> ()
    %79 = call @Unknown79(%alloc_5) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %80 = call @Unknown80(%72) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %81 = call @Unknown81(%68) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %82 = call @Unknown82(%64) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %83 = call @Unknown83(%60) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %84 = call @Unknown84(%53) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %85 = call @Unknown85(%49) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %86 = call @Unknown86(%56) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %87 = call @Unknown87(%45) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %88 = call @Unknown88(%41) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %89 = call @Unknown89(%34) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %90 = call @Unknown90(%30) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %91 = call @Unknown91(%37) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %92 = call @Unknown92(%26) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %93 = call @Unknown93(%22) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %94 = call @Unknown94(%15) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %95 = call @Unknown95(%11) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %96 = call @Unknown96(%18) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %97 = call @Unknown97(%7) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %98 = call @Unknown98(%3) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    return %75#2, %75#1, %77, %alloc_3, %79, %70#2, %70#1, %66#2, %66#1, %80, %81, %62#2, %62#1, %58#2, %58#1, %82, %83, %51#2, %51#1, %47#2, %47#1, %84, %85, %86, %54#2, %54#1, %43#2, %43#1, %39#2, %39#1, %87, %88, %32#2, %32#1, %28#2, %28#1, %89, %90, %91, %35#2, %35#1, %24#2, %24#1, %20#2, %20#1, %92, %93, %13#2, %13#1, %9#2, %9#1, %94, %95, %96, %16#2, %16#1, %5#2, %5#1, %1#2, %1#1, %97, %98 : memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>
  }
}