// RUN: byteir-opt %s -byre-opt="append-arg-types entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown99(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
    gpu.func @Unknown97(%arg0: memref<512x256x1x1xf16>, %arg1: memref<512x256x1x1xf32>) kernel {
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
    gpu.func @Unknown96(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
    gpu.func @Unknown95(%arg0: memref<512x256x3x3xf16>, %arg1: memref<512x256x3x3xf32>) kernel {
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
    gpu.func @Unknown94(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
    gpu.func @Unknown92(%arg0: memref<256x128x1x1xf16>, %arg1: memref<256x128x1x1xf32>) kernel {
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
    gpu.func @Unknown91(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
    gpu.func @Unknown90(%arg0: memref<256x128x3x3xf16>, %arg1: memref<256x128x3x3xf32>) kernel {
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
    gpu.func @Unknown89(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
    gpu.func @Unknown87(%arg0: memref<128x64x1x1xf16>, %arg1: memref<128x64x1x1xf32>) kernel {
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
    gpu.func @Unknown86(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
    gpu.func @Unknown85(%arg0: memref<128x64x3x3xf16>, %arg1: memref<128x64x3x3xf32>) kernel {
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
    gpu.func @Unknown84(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
    gpu.func @Unknown80(%arg0: memref<1000x512xf16>, %arg1: memref<1000x512xf32>) kernel {
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
    gpu.func @Unknown79(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) kernel {
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
        %6 = memref.load %arg0[%c0, %4] : memref<1x1000xf16>
        %7 = arith.extf %6 : f16 to f32
        memref.store %7, %arg1[%c0, %4] : memref<1x1000xf32>
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x64x112x112xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x64x112x112xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x64x112x112xf16>
      }
      gpu.return
    }
    gpu.func @Unknown73(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %28 = arith.addf %26, %27 : f16
        memref.store %28, %arg2[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown69(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown65(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>, %arg3: memref<1x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg2[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %27 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %28 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %29 = arith.addf %27, %28 : f16
        %30 = arith.cmpf ogt, %26, %cst : f16
        %31 = arith.select %30, %29, %cst : f16
        memref.store %31, %arg3[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown61(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown57(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>, %arg3: memref<1x64x56x56xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c200704 = arith.constant 200704 : index
      %c56 = arith.constant 56 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg2[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %27 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %28 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
        %29 = arith.addf %27, %28 : f16
        %30 = arith.cmpf ogt, %26, %cst : f16
        %31 = arith.select %30, %29, %cst : f16
        memref.store %31, %arg3[%c0, %25, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown50(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown46(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>, %arg3: memref<1x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg2[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %27 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %28 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %29 = arith.addf %27, %28 : f16
        %30 = arith.cmpf ogt, %26, %cst : f16
        %31 = arith.select %30, %29, %cst : f16
        memref.store %31, %arg3[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown42(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown38(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>, %arg3: memref<1x128x28x28xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c100352 = arith.constant 100352 : index
      %c28 = arith.constant 28 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg2[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %27 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %28 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
        %29 = arith.addf %27, %28 : f16
        %30 = arith.cmpf ogt, %26, %cst : f16
        %31 = arith.select %30, %29, %cst : f16
        memref.store %31, %arg3[%c0, %25, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c50176 = arith.constant 50176 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown27(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>, %arg3: memref<1x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c50176 = arith.constant 50176 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg2[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %27 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %28 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %29 = arith.addf %27, %28 : f16
        %30 = arith.cmpf ogt, %26, %cst : f16
        %31 = arith.select %30, %29, %cst : f16
        memref.store %31, %arg3[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown23(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c50176 = arith.constant 50176 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown19(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>, %arg3: memref<1x256x14x14xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c50176 = arith.constant 50176 : index
      %c14 = arith.constant 14 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg2[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %27 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %28 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
        %29 = arith.addf %27, %28 : f16
        %30 = arith.cmpf ogt, %26, %cst : f16
        %31 = arith.select %30, %29, %cst : f16
        memref.store %31, %arg3[%c0, %25, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown12(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c25088 = arith.constant 25088 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown8(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>, %arg3: memref<1x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c25088 = arith.constant 25088 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg2[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
        %27 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
        %28 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
        %29 = arith.addf %27, %28 : f16
        %30 = arith.cmpf ogt, %26, %cst : f16
        %31 = arith.select %30, %29, %cst : f16
        memref.store %31, %arg3[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c25088 = arith.constant 25088 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg0[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
        %27 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
        %28 = arith.cmpf ogt, %26, %cst : f16
        %29 = arith.select %28, %27, %cst : f16
        memref.store %29, %arg2[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 4.900000e+01 : f16
      %c0 = arith.constant 0 : index
      %c25088 = arith.constant 25088 : index
      %c7 = arith.constant 7 : index
      %c-1 = arith.constant -1 : index
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
        %26 = memref.load %arg1[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
        %27 = memref.load %arg0[%c0, %25] : memref<1x512xf16>
        %28 = arith.divf %27, %cst_0 : f16
        %29 = arith.cmpf ogt, %26, %cst : f16
        %30 = arith.select %29, %28, %cst : f16
        memref.store %30, %arg2[%c0, %25, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  func.func private @Unknown0(memref<1x512xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [2 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp1(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp2(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512x512x3x3xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x512x7x7xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp3(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<1x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x512x3x3xf16, "cuda">
  }
  func.func private @Unknown4(memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown4", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp5(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp6(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512x512x3x3xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x512x7x7xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp7(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<1x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x512x3x3xf16, "cuda">
  }
  func.func private @Unknown8(memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown8", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp9(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp10(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512x512x3x3xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x512x7x7xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp11(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<1x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x512x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<3x3x512x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x512x3x3xf16, "cuda">
  }
  func.func private @Unknown12(memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown12", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp13(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp14(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512x256x3x3xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x3x3xf16, "cuda">, memref<3x3x256x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x512xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x512xf16, "cuda">, memref<3x3x256x512xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16, "cuda">, memref<3x3x256x512xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp15(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<1x512x7x7xf16, "cuda">) -> memref<512x256x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<3x3x256x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x256x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x512xf16, "cuda">, memref<512x256x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x256x3x3xf16, "cuda">
  }
  func.func private @BatchNormGradOp16(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512xf32, "cuda">, %arg2: memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x512x7x7xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<512xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x512x7x7xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp17(%arg0: memref<1x512x7x7xf16, "cuda">, %arg1: memref<512x256x1x1xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x256x512xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x1x1xf16, "cuda">, memref<1x1x256x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16, "cuda">, memref<1x1x256x512xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_0 : memref<1x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp18(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<1x512x7x7xf16, "cuda">) -> memref<512x256x1x1xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x256x512xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x1x256x512xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<512x256x1x1xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x256x512xf16, "cuda">, memref<512x256x1x1xf16, "cuda">) -> ()
    return %alloc_0 : memref<512x256x1x1xf16, "cuda">
  }
  func.func private @Unknown19(memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown19", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp20(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp21(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256x256x3x3xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp22(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<1x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x256x3x3xf16, "cuda">
  }
  func.func private @Unknown23(memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown23", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp24(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp25(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256x256x3x3xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp26(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<1x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x256x3x3xf16, "cuda">
  }
  func.func private @Unknown27(memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown27", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp28(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp29(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256x256x3x3xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x256x14x14xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp30(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<1x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x256x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<3x3x256x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x256x3x3xf16, "cuda">
  }
  func.func private @Unknown31(memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown31", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp32(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp33(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256x128x3x3xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x3x3xf16, "cuda">, memref<3x3x128x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x256xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x256xf16, "cuda">, memref<3x3x128x256xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<3x3x128x256xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp34(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<1x256x14x14xf16, "cuda">) -> memref<256x128x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<3x3x128x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x128x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x256xf16, "cuda">, memref<256x128x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x128x3x3xf16, "cuda">
  }
  func.func private @BatchNormGradOp35(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256xf32, "cuda">, %arg2: memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x256x14x14xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<256xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x256x14x14xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp36(%arg0: memref<1x256x14x14xf16, "cuda">, %arg1: memref<256x128x1x1xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x128x256xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x1x1xf16, "cuda">, memref<1x1x128x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16, "cuda">, memref<1x1x128x256xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_0 : memref<1x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp37(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<1x256x14x14xf16, "cuda">) -> memref<256x128x1x1xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x128x256xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x1x128x256xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<256x128x1x1xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x128x256xf16, "cuda">, memref<256x128x1x1xf16, "cuda">) -> ()
    return %alloc_0 : memref<256x128x1x1xf16, "cuda">
  }
  func.func private @Unknown38(memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown38", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp39(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp40(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128x128x3x3xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp41(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<1x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x128x3x3xf16, "cuda">
  }
  func.func private @Unknown42(memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown42", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp43(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp44(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128x128x3x3xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp45(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<1x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x128x3x3xf16, "cuda">
  }
  func.func private @Unknown46(memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown46", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp47(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp48(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128x128x3x3xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x128x28x28xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp49(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<1x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x128x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<3x3x128x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x128x3x3xf16, "cuda">
  }
  func.func private @Unknown50(memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown50", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp51(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp52(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x3x3xf16, "cuda">, memref<3x3x64x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x128xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x128xf16, "cuda">, memref<3x3x64x128xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<3x3x64x128xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp53(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<1x128x28x28xf16, "cuda">) -> memref<128x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<3x3x64x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x128xf16, "cuda">, memref<128x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x64x3x3xf16, "cuda">
  }
  func.func private @BatchNormGradOp54(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128xf32, "cuda">, %arg2: memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x128x28x28xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<128xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x128x28x28xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp55(%arg0: memref<1x128x28x28xf16, "cuda">, %arg1: memref<128x64x1x1xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x64x128xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x1x1xf16, "cuda">, memref<1x1x64x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc, %alloc_0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16, "cuda">, memref<1x1x64x128xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_0 : memref<1x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp56(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<1x128x28x28xf16, "cuda">) -> memref<128x64x1x1xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<1x1x64x128xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x1x64x128xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<128x64x1x1xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x64x128xf16, "cuda">, memref<128x64x1x1xf16, "cuda">) -> ()
    return %alloc_0 : memref<128x64x1x1xf16, "cuda">
  }
  func.func private @Unknown57(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown57", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp58(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<1x64x56x56xf16, "cuda">) -> (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf32, "cuda">, memref<1x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x64x56x56xf32, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp59(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<64x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp60(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<1x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x64x3x3xf16, "cuda">
  }
  func.func private @Unknown61(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown61", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp62(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<1x64x56x56xf16, "cuda">) -> (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf32, "cuda">, memref<1x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x64x56x56xf32, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp63(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<64x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp64(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<1x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x64x3x3xf16, "cuda">
  }
  func.func private @Unknown65(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown65", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp66(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<1x64x56x56xf16, "cuda">) -> (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf32, "cuda">, memref<1x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x64x56x56xf32, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp67(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<64x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp68(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<1x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x64x3x3xf16, "cuda">
  }
  func.func private @Unknown69(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown69", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp70(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<1x64x56x56xf16, "cuda">) -> (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x56x56xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf32, "cuda">, memref<1x64x56x56xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x64x56x56xf32, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardDataOp71(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<64x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.transpose"(%arg1, %alloc) {device = "cuda", minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    "lmhlo.reverse"(%alloc, %alloc_0) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    lmhlo.convolution(%arg0, %alloc_0, %alloc_1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> ()
    return %alloc_1 : memref<1x64x56x56xf16, "cuda">
  }
  func.func private @ConvBackwardFilterOp72(%arg0: memref<1x64x56x56xf16, "cuda">, %arg1: memref<1x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<3x3x64x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<3x3x64x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x64x3x3xf16, "cuda">
  }
  func.func private @Unknown73(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown73", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown74(memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">) -> memref<1x64x112x112xf16, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown74", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @BatchNormGradOp75(%arg0: memref<1x64x112x112xf16, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<1x64x112x112xf16, "cuda">) -> (memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<1x64x112x112xf32, "cuda">
    "lmhlo.convert"(%arg0, %alloc_0) {device = "cuda"} : (memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf32, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x64x112x112xf32, "cuda">
    "lmhlo.convert"(%arg2, %alloc_1) {device = "cuda"} : (memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf32, "cuda">) -> ()
    %alloc_2 = memref.alloc() : memref<1x64x112x112xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<64xf32, "cuda">
    "lmhlo.batch_norm_grad"(%alloc_0, %arg1, %alloc, %alloc, %alloc_1, %alloc_2, %alloc_3, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x112x112xf32, "cuda">, memref<1x64x112x112xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1x64x112x112xf16, "cuda">
    "lmhlo.convert"(%alloc_2, %alloc_5) {device = "cuda"} : (memref<1x64x112x112xf32, "cuda">, memref<1x64x112x112xf16, "cuda">) -> ()
    return %alloc_5, %alloc_3, %alloc_4 : memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
  }
  func.func private @ConvBackwardFilterOp76(%arg0: memref<1x3x224x224xf16, "cuda">, %arg1: memref<1x64x112x112xf16, "cuda">) -> memref<64x3x7x7xf16, "cuda"> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp", device = "cuda"} {
    %alloc = memref.alloc() : memref<7x7x3x64xf16, "cuda">
    lmhlo.convolution(%arg0, %arg1, %alloc) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x3x224x224xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<7x7x3x64xf16, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<64x3x7x7xf16, "cuda">
    "lmhlo.transpose"(%alloc, %alloc_0) {device = "cuda", minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<7x7x3x64xf16, "cuda">, memref<64x3x7x7xf16, "cuda">) -> ()
    return %alloc_0 : memref<64x3x7x7xf16, "cuda">
  }
  func.func private @Unknown77(memref<64x3x7x7xf16, "cuda">) -> memref<64x3x7x7xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 74 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown77", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown78(memref<1x1000xf16, "cuda">) -> memref<1x1000xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown78", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown79(memref<1000xf32, "cuda">) -> memref<1000xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown79", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown80(memref<1000x512xf16, "cuda">) -> memref<1000x512xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4000 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown80", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown81(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown81", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown82(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown82", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown83(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown83", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown84(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown84", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown85(memref<128x64x3x3xf16, "cuda">) -> memref<128x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown85", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown86(memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown86", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown87(memref<128x64x1x1xf16, "cuda">) -> memref<128x64x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 64 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown87", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown88(memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown88", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown89(memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown89", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown90(memref<256x128x3x3xf16, "cuda">) -> memref<256x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown90", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown91(memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown91", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown92(memref<256x128x1x1xf16, "cuda">) -> memref<256x128x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown92", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown93(memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown93", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown94(memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown94", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown95(memref<512x256x3x3xf16, "cuda">) -> memref<512x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 9216 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown95", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown96(memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown96", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown97(memref<512x256x1x1xf16, "cuda">) -> memref<512x256x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown97", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown98(memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown98", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown99(memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown99", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func @main(%arg0: memref<64xf32, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<64xf32, "cuda">, %arg3: memref<64xf32, "cuda">, %arg4: memref<64xf32, "cuda">, %arg5: memref<64xf32, "cuda">, %arg6: memref<64xf32, "cuda">, %arg7: memref<64xf32, "cuda">, %arg8: memref<64xf32, "cuda">, %arg9: memref<64xf32, "cuda">, %arg10: memref<128xf32, "cuda">, %arg11: memref<128xf32, "cuda">, %arg12: memref<128xf32, "cuda">, %arg13: memref<128xf32, "cuda">, %arg14: memref<128xf32, "cuda">, %arg15: memref<128xf32, "cuda">, %arg16: memref<128xf32, "cuda">, %arg17: memref<128xf32, "cuda">, %arg18: memref<128xf32, "cuda">, %arg19: memref<128xf32, "cuda">, %arg20: memref<256xf32, "cuda">, %arg21: memref<256xf32, "cuda">, %arg22: memref<256xf32, "cuda">, %arg23: memref<256xf32, "cuda">, %arg24: memref<256xf32, "cuda">, %arg25: memref<256xf32, "cuda">, %arg26: memref<256xf32, "cuda">, %arg27: memref<256xf32, "cuda">, %arg28: memref<256xf32, "cuda">, %arg29: memref<256xf32, "cuda">, %arg30: memref<512xf32, "cuda">, %arg31: memref<512xf32, "cuda">, %arg32: memref<512xf32, "cuda">, %arg33: memref<512xf32, "cuda">, %arg34: memref<512xf32, "cuda">, %arg35: memref<512xf32, "cuda">, %arg36: memref<512xf32, "cuda">, %arg37: memref<512xf32, "cuda">, %arg38: memref<512xf32, "cuda">, %arg39: memref<512xf32, "cuda">, %arg40: memref<64xf32, "cuda">, %arg41: memref<64xf32, "cuda">, %arg42: memref<64xf32, "cuda">, %arg43: memref<64xf32, "cuda">, %arg44: memref<64xf32, "cuda">, %arg45: memref<64xf32, "cuda">, %arg46: memref<64xf32, "cuda">, %arg47: memref<64xf32, "cuda">, %arg48: memref<64xf32, "cuda">, %arg49: memref<64xf32, "cuda">, %arg50: memref<128xf32, "cuda">, %arg51: memref<128xf32, "cuda">, %arg52: memref<128xf32, "cuda">, %arg53: memref<128xf32, "cuda">, %arg54: memref<128xf32, "cuda">, %arg55: memref<128xf32, "cuda">, %arg56: memref<128xf32, "cuda">, %arg57: memref<128xf32, "cuda">, %arg58: memref<128xf32, "cuda">, %arg59: memref<128xf32, "cuda">, %arg60: memref<256xf32, "cuda">, %arg61: memref<256xf32, "cuda">, %arg62: memref<256xf32, "cuda">, %arg63: memref<256xf32, "cuda">, %arg64: memref<256xf32, "cuda">, %arg65: memref<256xf32, "cuda">, %arg66: memref<256xf32, "cuda">, %arg67: memref<256xf32, "cuda">, %arg68: memref<256xf32, "cuda">, %arg69: memref<256xf32, "cuda">, %arg70: memref<512xf32, "cuda">, %arg71: memref<512xf32, "cuda">, %arg72: memref<512xf32, "cuda">, %arg73: memref<512xf32, "cuda">, %arg74: memref<512xf32, "cuda">, %arg75: memref<512xf32, "cuda">, %arg76: memref<512xf32, "cuda">, %arg77: memref<512xf32, "cuda">, %arg78: memref<512xf32, "cuda">, %arg79: memref<512xf32, "cuda">, %arg80: memref<64x3x7x7xf16, "cuda">, %arg81: memref<1x3x224x224xf16, "cuda">, %arg82: memref<1x64x112x112xf16, "cuda">, %arg83: memref<1x64x112x112xf16, "cuda">, %arg84: memref<1x64x56x56xf16, "cuda">, %arg85: memref<64x64x3x3xf16, "cuda">, %arg86: memref<1x64x56x56xf16, "cuda">, %arg87: memref<1x64x56x56xf16, "cuda">, %arg88: memref<64x64x3x3xf16, "cuda">, %arg89: memref<1x64x56x56xf16, "cuda">, %arg90: memref<1x64x56x56xf16, "cuda">, %arg91: memref<64x64x3x3xf16, "cuda">, %arg92: memref<1x64x56x56xf16, "cuda">, %arg93: memref<1x64x56x56xf16, "cuda">, %arg94: memref<64x64x3x3xf16, "cuda">, %arg95: memref<1x64x56x56xf16, "cuda">, %arg96: memref<1x64x56x56xf16, "cuda">, %arg97: memref<128x64x3x3xf16, "cuda">, %arg98: memref<1x128x28x28xf16, "cuda">, %arg99: memref<1x128x28x28xf16, "cuda">, %arg100: memref<128x128x3x3xf16, "cuda">, %arg101: memref<1x128x28x28xf16, "cuda">, %arg102: memref<128x64x1x1xf16, "cuda">, %arg103: memref<1x128x28x28xf16, "cuda">, %arg104: memref<1x128x28x28xf16, "cuda">, %arg105: memref<128x128x3x3xf16, "cuda">, %arg106: memref<1x128x28x28xf16, "cuda">, %arg107: memref<1x128x28x28xf16, "cuda">, %arg108: memref<128x128x3x3xf16, "cuda">, %arg109: memref<1x128x28x28xf16, "cuda">, %arg110: memref<1x128x28x28xf16, "cuda">, %arg111: memref<256x128x3x3xf16, "cuda">, %arg112: memref<1x256x14x14xf16, "cuda">, %arg113: memref<1x256x14x14xf16, "cuda">, %arg114: memref<256x256x3x3xf16, "cuda">, %arg115: memref<1x256x14x14xf16, "cuda">, %arg116: memref<256x128x1x1xf16, "cuda">, %arg117: memref<1x256x14x14xf16, "cuda">, %arg118: memref<1x256x14x14xf16, "cuda">, %arg119: memref<256x256x3x3xf16, "cuda">, %arg120: memref<1x256x14x14xf16, "cuda">, %arg121: memref<1x256x14x14xf16, "cuda">, %arg122: memref<256x256x3x3xf16, "cuda">, %arg123: memref<1x256x14x14xf16, "cuda">, %arg124: memref<1x256x14x14xf16, "cuda">, %arg125: memref<512x256x3x3xf16, "cuda">, %arg126: memref<1x512x7x7xf16, "cuda">, %arg127: memref<1x512x7x7xf16, "cuda">, %arg128: memref<512x512x3x3xf16, "cuda">, %arg129: memref<1x512x7x7xf16, "cuda">, %arg130: memref<512x256x1x1xf16, "cuda">, %arg131: memref<1x512x7x7xf16, "cuda">, %arg132: memref<1x512x7x7xf16, "cuda">, %arg133: memref<512x512x3x3xf16, "cuda">, %arg134: memref<1x512x7x7xf16, "cuda">, %arg135: memref<1x512x7x7xf16, "cuda">, %arg136: memref<512x512x3x3xf16, "cuda">, %arg137: memref<1x512x7x7xf16, "cuda">, %arg138: memref<1x512x7x7xf16, "cuda">, %arg139: memref<1x512xf16, "cuda">, %arg140: memref<512x1000xf16, "cuda">, %arg141: memref<1x1000xf16, "cuda">) -> (memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x3x7x7xf32, "cuda">, memref<1000xf32, "cuda">, memref<1000x512xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x64x3x3xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128x64x1x1xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x128x3x3xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256x128x1x1xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x256x3x3xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512x256x1x1xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf32, "cuda">) {
    %alloc = memref.alloc() : memref<f32, "cuda">
    "lmhlo.constant"(%alloc) {device = "cuda", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, "cuda">) -> ()
    %alloc_0 = memref.alloc() : memref<f16, "cuda">
    "lmhlo.constant"(%alloc_0) {device = "cuda", value = dense<0.000000e+00> : tensor<f16>} : (memref<f16, "cuda">) -> ()
    %alloc_1 = memref.alloc() : memref<1x512xf16, "cuda">
    "lmhlo.dot"(%arg141, %arg140, %alloc_1) {device = "cuda", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x1000xf16, "cuda">, memref<512x1000xf16, "cuda">, memref<1x512xf16, "cuda">) -> ()
    %0 = call @Unknown0(%alloc_1, %arg138) : (memref<1x512xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %1:3 = call @BatchNormGradOp1(%arg137, %arg39, %0) : (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %2 = call @ConvBackwardDataOp2(%1#0, %arg136) : (memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %3 = call @ConvBackwardFilterOp3(%arg135, %1#0) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %4 = call @Unknown4(%arg135, %2) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %5:3 = call @BatchNormGradOp5(%arg134, %arg37, %4) : (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %6 = call @ConvBackwardDataOp6(%5#0, %arg133) : (memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %7 = call @ConvBackwardFilterOp7(%arg132, %5#0) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %8 = call @Unknown8(%0, %6, %arg132) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %9:3 = call @BatchNormGradOp9(%arg129, %arg33, %8) : (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %10 = call @ConvBackwardDataOp10(%9#0, %arg128) : (memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %11 = call @ConvBackwardFilterOp11(%arg127, %9#0) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %12 = call @Unknown12(%arg127, %10) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %13:3 = call @BatchNormGradOp13(%arg126, %arg31, %12) : (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %14 = call @ConvBackwardDataOp14(%13#0, %arg125) : (memref<1x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %15 = call @ConvBackwardFilterOp15(%arg124, %13#0) : (memref<1x256x14x14xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<512x256x3x3xf16, "cuda">
    %16:3 = call @BatchNormGradOp16(%arg131, %arg35, %8) : (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">) -> (memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">)
    %17 = call @ConvBackwardDataOp17(%16#0, %arg130) : (memref<1x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %18 = call @ConvBackwardFilterOp18(%arg124, %16#0) : (memref<1x256x14x14xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<512x256x1x1xf16, "cuda">
    %19 = call @Unknown19(%17, %14, %arg124) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %20:3 = call @BatchNormGradOp20(%arg123, %arg29, %19) : (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %21 = call @ConvBackwardDataOp21(%20#0, %arg122) : (memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %22 = call @ConvBackwardFilterOp22(%arg121, %20#0) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %23 = call @Unknown23(%arg121, %21) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %24:3 = call @BatchNormGradOp24(%arg120, %arg27, %23) : (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %25 = call @ConvBackwardDataOp25(%24#0, %arg119) : (memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %26 = call @ConvBackwardFilterOp26(%arg118, %24#0) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %27 = call @Unknown27(%19, %25, %arg118) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %28:3 = call @BatchNormGradOp28(%arg115, %arg23, %27) : (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %29 = call @ConvBackwardDataOp29(%28#0, %arg114) : (memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %30 = call @ConvBackwardFilterOp30(%arg113, %28#0) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %31 = call @Unknown31(%arg113, %29) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %32:3 = call @BatchNormGradOp32(%arg112, %arg21, %31) : (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %33 = call @ConvBackwardDataOp33(%32#0, %arg111) : (memref<1x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %34 = call @ConvBackwardFilterOp34(%arg110, %32#0) : (memref<1x128x28x28xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<256x128x3x3xf16, "cuda">
    %35:3 = call @BatchNormGradOp35(%arg117, %arg25, %27) : (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">) -> (memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">)
    %36 = call @ConvBackwardDataOp36(%35#0, %arg116) : (memref<1x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %37 = call @ConvBackwardFilterOp37(%arg110, %35#0) : (memref<1x128x28x28xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<256x128x1x1xf16, "cuda">
    %38 = call @Unknown38(%36, %33, %arg110) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %39:3 = call @BatchNormGradOp39(%arg109, %arg19, %38) : (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %40 = call @ConvBackwardDataOp40(%39#0, %arg108) : (memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %41 = call @ConvBackwardFilterOp41(%arg107, %39#0) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %42 = call @Unknown42(%arg107, %40) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %43:3 = call @BatchNormGradOp43(%arg106, %arg17, %42) : (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %44 = call @ConvBackwardDataOp44(%43#0, %arg105) : (memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %45 = call @ConvBackwardFilterOp45(%arg104, %43#0) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %46 = call @Unknown46(%38, %44, %arg104) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %47:3 = call @BatchNormGradOp47(%arg101, %arg13, %46) : (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %48 = call @ConvBackwardDataOp48(%47#0, %arg100) : (memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %49 = call @ConvBackwardFilterOp49(%arg99, %47#0) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %50 = call @Unknown50(%arg99, %48) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %51:3 = call @BatchNormGradOp51(%arg98, %arg11, %50) : (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %52 = call @ConvBackwardDataOp52(%51#0, %arg97) : (memref<1x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %53 = call @ConvBackwardFilterOp53(%arg96, %51#0) : (memref<1x64x56x56xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<128x64x3x3xf16, "cuda">
    %54:3 = call @BatchNormGradOp54(%arg103, %arg15, %46) : (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">) -> (memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">)
    %55 = call @ConvBackwardDataOp55(%54#0, %arg102) : (memref<1x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %56 = call @ConvBackwardFilterOp56(%arg96, %54#0) : (memref<1x64x56x56xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<128x64x1x1xf16, "cuda">
    %57 = call @Unknown57(%55, %52, %arg96) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %58:3 = call @BatchNormGradOp58(%arg95, %arg9, %57) : (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">) -> (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %59 = call @ConvBackwardDataOp59(%58#0, %arg94) : (memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %60 = call @ConvBackwardFilterOp60(%arg93, %58#0) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %61 = call @Unknown61(%arg93, %59) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %62:3 = call @BatchNormGradOp62(%arg92, %arg7, %61) : (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">) -> (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %63 = call @ConvBackwardDataOp63(%62#0, %arg91) : (memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %64 = call @ConvBackwardFilterOp64(%arg90, %62#0) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %65 = call @Unknown65(%57, %63, %arg90) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %66:3 = call @BatchNormGradOp66(%arg89, %arg5, %65) : (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">) -> (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %67 = call @ConvBackwardDataOp67(%66#0, %arg88) : (memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %68 = call @ConvBackwardFilterOp68(%arg87, %66#0) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %69 = call @Unknown69(%arg87, %67) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %70:3 = call @BatchNormGradOp70(%arg86, %arg3, %69) : (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">) -> (memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %71 = call @ConvBackwardDataOp71(%70#0, %arg85) : (memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %72 = call @ConvBackwardFilterOp72(%arg84, %70#0) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %73 = call @Unknown73(%65, %71) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %alloc_2 = memref.alloc() : memref<1x64x112x112xf16, "cuda">
    "lmhlo.select_and_scatter"(%arg83, %73, %alloc_0, %alloc_2) ({
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %100 = mhlo.compare  GE, %arg142, %arg143 : (tensor<f16>, tensor<f16>) -> tensor<i1>
      mhlo.return %100 : tensor<i1>
    }, {
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %100 = mhlo.add %arg142, %arg143 : tensor<f16>
      mhlo.return %100 : tensor<f16>
    }) {device = "cuda", padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<1x64x112x112xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<f16, "cuda">, memref<1x64x112x112xf16, "cuda">) -> ()
    %74 = call @Unknown74(%arg83, %alloc_2) : (memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">) -> memref<1x64x112x112xf16, "cuda">
    %75:3 = call @BatchNormGradOp75(%arg82, %arg1, %74) : (memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x112x112xf16, "cuda">) -> (memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">)
    %76 = call @ConvBackwardFilterOp76(%arg81, %75#0) : (memref<1x3x224x224xf16, "cuda">, memref<1x64x112x112xf16, "cuda">) -> memref<64x3x7x7xf16, "cuda">
    %77 = call @Unknown77(%76) : (memref<64x3x7x7xf16, "cuda">) -> memref<64x3x7x7xf32, "cuda">
    %78 = call @Unknown78(%arg141) : (memref<1x1000xf16, "cuda">) -> memref<1x1000xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<1000xf32, "cuda">
    "lmhlo.reduce"(%78, %alloc, %alloc_3) ({
    ^bb0(%arg142: memref<f32>, %arg143: memref<f32>, %arg144: memref<f32>):
      "lmhlo.add"(%arg142, %arg143, %arg144) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {device = "cuda", dimensions = dense<0> : tensor<1xi64>} : (memref<1x1000xf32, "cuda">, memref<f32, "cuda">, memref<1000xf32, "cuda">) -> ()
    %79 = call @Unknown79(%alloc_3) : (memref<1000xf32, "cuda">) -> memref<1000xf32, "cuda">
    %alloc_4 = memref.alloc() : memref<1000x1xf16, "cuda">
    "lmhlo.reshape"(%arg141, %alloc_4) {device = "cuda"} : (memref<1x1000xf16, "cuda">, memref<1000x1xf16, "cuda">) -> ()
    %alloc_5 = memref.alloc() : memref<1000x512xf16, "cuda">
    "lmhlo.dot"(%alloc_4, %arg139, %alloc_5) {device = "cuda", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1000x1xf16, "cuda">, memref<1x512xf16, "cuda">, memref<1000x512xf16, "cuda">) -> ()
    %80 = call @Unknown80(%alloc_5) : (memref<1000x512xf16, "cuda">) -> memref<1000x512xf32, "cuda">
    %81 = call @Unknown81(%72) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %82 = call @Unknown82(%68) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %83 = call @Unknown83(%64) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %84 = call @Unknown84(%60) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %85 = call @Unknown85(%53) : (memref<128x64x3x3xf16, "cuda">) -> memref<128x64x3x3xf32, "cuda">
    %86 = call @Unknown86(%49) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %87 = call @Unknown87(%56) : (memref<128x64x1x1xf16, "cuda">) -> memref<128x64x1x1xf32, "cuda">
    %88 = call @Unknown88(%45) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %89 = call @Unknown89(%41) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %90 = call @Unknown90(%34) : (memref<256x128x3x3xf16, "cuda">) -> memref<256x128x3x3xf32, "cuda">
    %91 = call @Unknown91(%30) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %92 = call @Unknown92(%37) : (memref<256x128x1x1xf16, "cuda">) -> memref<256x128x1x1xf32, "cuda">
    %93 = call @Unknown93(%26) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %94 = call @Unknown94(%22) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %95 = call @Unknown95(%15) : (memref<512x256x3x3xf16, "cuda">) -> memref<512x256x3x3xf32, "cuda">
    %96 = call @Unknown96(%11) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %97 = call @Unknown97(%18) : (memref<512x256x1x1xf16, "cuda">) -> memref<512x256x1x1xf32, "cuda">
    %98 = call @Unknown98(%7) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %99 = call @Unknown99(%3) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    return %75#2, %75#1, %77, %79, %80, %70#2, %70#1, %66#2, %66#1, %81, %82, %62#2, %62#1, %58#2, %58#1, %83, %84, %51#2, %51#1, %47#2, %47#1, %85, %86, %87, %54#2, %54#1, %43#2, %43#1, %39#2, %39#1, %88, %89, %32#2, %32#1, %28#2, %28#1, %90, %91, %92, %35#2, %35#1, %24#2, %24#1, %20#2, %20#1, %93, %94, %13#2, %13#1, %9#2, %9#1, %95, %96, %97, %16#2, %16#1, %5#2, %5#1, %1#2, %1#1, %98, %99 : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x3x7x7xf32, "cuda">, memref<1000xf32, "cuda">, memref<1000x512xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x64x3x3xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128x64x1x1xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x128x3x3xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256x128x1x1xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x256x3x3xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512x256x1x1xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf32, "cuda">
  }
}