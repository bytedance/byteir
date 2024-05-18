// RUN: byteir-opt %s -byre-opt="append-arg-types entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown96(%arg0: memref<512x256x1x1xf16>, %arg1: memref<512x256x1x1xf32>) kernel {
      %c131072 = arith.constant 131072 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c131072 step %6 {
        %7 = arith.remsi %arg2, %c256 : index
        %8 = arith.divsi %arg2, %c256 : index
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<512x256x1x1xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<512x256x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown95(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
      %c2359296 = arith.constant 2359296 : index
      %c512 = arith.constant 512 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c2359296 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c512 : index
        %12 = arith.divsi %10, %c512 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<512x512x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<512x512x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown94(%arg0: memref<512x256x3x3xf16>, %arg1: memref<512x256x3x3xf32>) kernel {
      %c1179648 = arith.constant 1179648 : index
      %c256 = arith.constant 256 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c1179648 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c256 : index
        %12 = arith.divsi %10, %c256 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<512x256x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<512x256x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown91(%arg0: memref<256x128x1x1xf16>, %arg1: memref<256x128x1x1xf32>) kernel {
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c32768 step %6 {
        %7 = arith.remsi %arg2, %c128 : index
        %8 = arith.divsi %arg2, %c128 : index
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<256x128x1x1xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<256x128x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown90(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
      %c589824 = arith.constant 589824 : index
      %c256 = arith.constant 256 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c589824 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c256 : index
        %12 = arith.divsi %10, %c256 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<256x256x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<256x256x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown89(%arg0: memref<256x128x3x3xf16>, %arg1: memref<256x128x3x3xf32>) kernel {
      %c294912 = arith.constant 294912 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c294912 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c128 : index
        %12 = arith.divsi %10, %c128 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<256x128x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<256x128x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown86(%arg0: memref<128x64x1x1xf16>, %arg1: memref<128x64x1x1xf32>) kernel {
      %c8192 = arith.constant 8192 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c8192 step %6 {
        %7 = arith.remsi %arg2, %c64 : index
        %8 = arith.divsi %arg2, %c64 : index
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<128x64x1x1xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<128x64x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown85(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
      %c147456 = arith.constant 147456 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c147456 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c128 : index
        %12 = arith.divsi %10, %c128 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<128x128x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<128x128x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown84(%arg0: memref<128x64x3x3xf16>, %arg1: memref<128x64x3x3xf32>) kernel {
      %c73728 = arith.constant 73728 : index
      %c64 = arith.constant 64 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c73728 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<128x64x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<128x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown80(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %c36864 = arith.constant 36864 : index
      %c64 = arith.constant 64 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c36864 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<64x64x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown79(%arg0: memref<1000x512xf16>, %arg1: memref<1000x512xf32>) kernel {
      %c512000 = arith.constant 512000 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c512000 step %6 {
        %7 = arith.remsi %arg2, %c512 : index
        %8 = arith.divsi %arg2, %c512 : index
        %9 = memref.load %arg0[%8, %7] : memref<1000x512xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%8, %7] : memref<1000x512xf32>
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
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c1000 step %6 {
        %7 = memref.load %arg0[%c0, %arg2] : memref<1x1000xf16>
        %8 = arith.extf %7 : f16 to f32
        %9 = arith.truncf %8 : f32 to f16
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%c0, %arg2] : memref<1x1000xf32>
      }
      gpu.return
    }
    gpu.func @Unknown77(%arg0: memref<64x3x7x7xf16>, %arg1: memref<64x3x7x7xf32>) kernel {
      %c9408 = arith.constant 9408 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c9408 step %6 {
        %7 = arith.remsi %arg2, %c7 : index
        %8 = arith.divsi %arg2, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = arith.remsi %10, %c3 : index
        %12 = arith.divsi %10, %c3 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<64x3x7x7xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<64x3x7x7xf32>
      }
      gpu.return
    }
    gpu.func @Unknown74(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>, %arg2: memref<1x64x112x112xf16>) kernel {
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c112 = arith.constant 112 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg3, %c112 : index
        %8 = arith.divsi %arg3, %c112 : index
        %9 = arith.remsi %8, %c112 : index
        %10 = arith.divsi %8, %c112 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
      }
      gpu.return
    }
    gpu.func @Unknown73(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %c200704 = arith.constant 200704 : index
      %c0 = arith.constant 0 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg3, %c56 : index
        %8 = arith.divsi %arg3, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %13 = arith.addf %11, %12 : f16
        memref.store %13, %arg2[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown61(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg3, %c56 : index
        %8 = arith.divsi %arg3, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown57(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>, %arg3: memref<1x64x56x56xf16>) kernel {
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg4, %c56 : index
        %8 = arith.divsi %arg4, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %13 = memref.load %arg2[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %14 = arith.addf %11, %12 : f16
        %15 = arith.cmpf ogt, %13, %cst : f16
        %16 = arith.select %15, %14, %cst : f16
        memref.store %16, %arg3[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown42(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c28 = arith.constant 28 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg3, %c28 : index
        %8 = arith.divsi %arg3, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown38(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>, %arg3: memref<1x128x28x28xf16>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c28 = arith.constant 28 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg4, %c28 : index
        %8 = arith.divsi %arg4, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %13 = memref.load %arg2[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %14 = arith.addf %11, %12 : f16
        %15 = arith.cmpf ogt, %13, %cst : f16
        %16 = arith.select %15, %14, %cst : f16
        memref.store %16, %arg3[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown23(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %c50176 = arith.constant 50176 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c50176 step %6 {
        %7 = arith.remsi %arg3, %c14 : index
        %8 = arith.divsi %arg3, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown19(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>, %arg3: memref<1x256x14x14xf16>) kernel {
      %c50176 = arith.constant 50176 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c50176 step %6 {
        %7 = arith.remsi %arg4, %c14 : index
        %8 = arith.divsi %arg4, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %13 = memref.load %arg2[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %14 = arith.addf %11, %12 : f16
        %15 = arith.cmpf ogt, %13, %cst : f16
        %16 = arith.select %15, %14, %cst : f16
        memref.store %16, %arg3[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown8(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>, %arg3: memref<1x512x7x7xf16>) kernel {
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c25088 step %6 {
        %7 = arith.remsi %arg4, %c7 : index
        %8 = arith.divsi %arg4, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %13 = memref.load %arg2[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %14 = arith.addf %11, %12 : f16
        %15 = arith.cmpf ogt, %13, %cst : f16
        %16 = arith.select %15, %14, %cst : f16
        memref.store %16, %arg3[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c25088 step %6 {
        %7 = arith.remsi %arg3, %c7 : index
        %8 = arith.divsi %arg3, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 4.900000e+01 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c25088 step %6 {
        %7 = arith.remsi %arg3, %c7 : index
        %8 = arith.divsi %arg3, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = memref.load %arg0[%c0, %10] : memref<1x512xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %13 = arith.divf %11, %cst : f16
        %14 = arith.cmpf ogt, %12, %cst_0 : f16
        %15 = arith.select %14, %13, %cst_0 : f16
        memref.store %15, %arg2[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  func.func private @Unknown0(memref<1x512xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 25 : i32, __byre__arg_ranks = [2 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown4(memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 25 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown4", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown8(memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 25 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown8", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown19(memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 49 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown19", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown23(memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 49 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown23", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown38(memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown38", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown42(memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown42", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown57(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown57", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown61(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown61", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown73(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown73", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown74(memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">) -> memref<1x64x112x112xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown74", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown77(memref<64x3x7x7xf16, "cuda">) -> memref<64x3x7x7xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 10 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown77", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown78(memref<1x1000xf16, "cuda">) -> memref<1000xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown78", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown79(memref<1000x512xf16, "cuda">) -> memref<1000x512xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 500 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown79", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown80(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 36 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown80", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown84(memref<128x64x3x3xf16, "cuda">) -> memref<128x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 72 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown84", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown85(memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 144 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown85", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown86(memref<128x64x1x1xf16, "cuda">) -> memref<128x64x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown86", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown89(memref<256x128x3x3xf16, "cuda">) -> memref<256x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown89", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown90(memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown90", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown91(memref<256x128x1x1xf16, "cuda">) -> memref<256x128x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown91", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown94(memref<512x256x3x3xf16, "cuda">) -> memref<512x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown94", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown95(memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown95", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown96(memref<512x256x1x1xf16, "cuda">) -> memref<512x256x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 128 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown96", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func @main(%arg0: memref<64xf32, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<64xf32, "cuda">, %arg3: memref<64xf32, "cuda">, %arg4: memref<64xf32, "cuda">, %arg5: memref<64xf32, "cuda">, %arg6: memref<64xf32, "cuda">, %arg7: memref<64xf32, "cuda">, %arg8: memref<64xf32, "cuda">, %arg9: memref<64xf32, "cuda">, %arg10: memref<128xf32, "cuda">, %arg11: memref<128xf32, "cuda">, %arg12: memref<128xf32, "cuda">, %arg13: memref<128xf32, "cuda">, %arg14: memref<128xf32, "cuda">, %arg15: memref<128xf32, "cuda">, %arg16: memref<128xf32, "cuda">, %arg17: memref<128xf32, "cuda">, %arg18: memref<128xf32, "cuda">, %arg19: memref<128xf32, "cuda">, %arg20: memref<256xf32, "cuda">, %arg21: memref<256xf32, "cuda">, %arg22: memref<256xf32, "cuda">, %arg23: memref<256xf32, "cuda">, %arg24: memref<256xf32, "cuda">, %arg25: memref<256xf32, "cuda">, %arg26: memref<256xf32, "cuda">, %arg27: memref<256xf32, "cuda">, %arg28: memref<256xf32, "cuda">, %arg29: memref<256xf32, "cuda">, %arg30: memref<512xf32, "cuda">, %arg31: memref<512xf32, "cuda">, %arg32: memref<512xf32, "cuda">, %arg33: memref<512xf32, "cuda">, %arg34: memref<512xf32, "cuda">, %arg35: memref<512xf32, "cuda">, %arg36: memref<512xf32, "cuda">, %arg37: memref<512xf32, "cuda">, %arg38: memref<512xf32, "cuda">, %arg39: memref<512xf32, "cuda">, %arg40: memref<64xf32, "cuda">, %arg41: memref<64xf32, "cuda">, %arg42: memref<64xf32, "cuda">, %arg43: memref<64xf32, "cuda">, %arg44: memref<64xf32, "cuda">, %arg45: memref<64xf32, "cuda">, %arg46: memref<64xf32, "cuda">, %arg47: memref<64xf32, "cuda">, %arg48: memref<64xf32, "cuda">, %arg49: memref<64xf32, "cuda">, %arg50: memref<128xf32, "cuda">, %arg51: memref<128xf32, "cuda">, %arg52: memref<128xf32, "cuda">, %arg53: memref<128xf32, "cuda">, %arg54: memref<128xf32, "cuda">, %arg55: memref<128xf32, "cuda">, %arg56: memref<128xf32, "cuda">, %arg57: memref<128xf32, "cuda">, %arg58: memref<128xf32, "cuda">, %arg59: memref<128xf32, "cuda">, %arg60: memref<256xf32, "cuda">, %arg61: memref<256xf32, "cuda">, %arg62: memref<256xf32, "cuda">, %arg63: memref<256xf32, "cuda">, %arg64: memref<256xf32, "cuda">, %arg65: memref<256xf32, "cuda">, %arg66: memref<256xf32, "cuda">, %arg67: memref<256xf32, "cuda">, %arg68: memref<256xf32, "cuda">, %arg69: memref<256xf32, "cuda">, %arg70: memref<512xf32, "cuda">, %arg71: memref<512xf32, "cuda">, %arg72: memref<512xf32, "cuda">, %arg73: memref<512xf32, "cuda">, %arg74: memref<512xf32, "cuda">, %arg75: memref<512xf32, "cuda">, %arg76: memref<512xf32, "cuda">, %arg77: memref<512xf32, "cuda">, %arg78: memref<512xf32, "cuda">, %arg79: memref<512xf32, "cuda">, %arg80: memref<64x3x7x7xf16, "cuda">, %arg81: memref<1x3x224x224xf16, "cuda">, %arg82: memref<1x64x112x112xf16, "cuda">, %arg83: memref<1x64x112x112xf16, "cuda">, %arg84: memref<1x64x56x56xf16, "cuda">, %arg85: memref<64x64x3x3xf16, "cuda">, %arg86: memref<1x64x56x56xf16, "cuda">, %arg87: memref<1x64x56x56xf16, "cuda">, %arg88: memref<64x64x3x3xf16, "cuda">, %arg89: memref<1x64x56x56xf16, "cuda">, %arg90: memref<1x64x56x56xf16, "cuda">, %arg91: memref<64x64x3x3xf16, "cuda">, %arg92: memref<1x64x56x56xf16, "cuda">, %arg93: memref<1x64x56x56xf16, "cuda">, %arg94: memref<64x64x3x3xf16, "cuda">, %arg95: memref<1x64x56x56xf16, "cuda">, %arg96: memref<1x64x56x56xf16, "cuda">, %arg97: memref<128x64x3x3xf16, "cuda">, %arg98: memref<1x128x28x28xf16, "cuda">, %arg99: memref<1x128x28x28xf16, "cuda">, %arg100: memref<128x128x3x3xf16, "cuda">, %arg101: memref<1x128x28x28xf16, "cuda">, %arg102: memref<128x64x1x1xf16, "cuda">, %arg103: memref<1x128x28x28xf16, "cuda">, %arg104: memref<1x128x28x28xf16, "cuda">, %arg105: memref<128x128x3x3xf16, "cuda">, %arg106: memref<1x128x28x28xf16, "cuda">, %arg107: memref<1x128x28x28xf16, "cuda">, %arg108: memref<128x128x3x3xf16, "cuda">, %arg109: memref<1x128x28x28xf16, "cuda">, %arg110: memref<1x128x28x28xf16, "cuda">, %arg111: memref<256x128x3x3xf16, "cuda">, %arg112: memref<1x256x14x14xf16, "cuda">, %arg113: memref<1x256x14x14xf16, "cuda">, %arg114: memref<256x256x3x3xf16, "cuda">, %arg115: memref<1x256x14x14xf16, "cuda">, %arg116: memref<256x128x1x1xf16, "cuda">, %arg117: memref<1x256x14x14xf16, "cuda">, %arg118: memref<1x256x14x14xf16, "cuda">, %arg119: memref<256x256x3x3xf16, "cuda">, %arg120: memref<1x256x14x14xf16, "cuda">, %arg121: memref<1x256x14x14xf16, "cuda">, %arg122: memref<256x256x3x3xf16, "cuda">, %arg123: memref<1x256x14x14xf16, "cuda">, %arg124: memref<1x256x14x14xf16, "cuda">, %arg125: memref<512x256x3x3xf16, "cuda">, %arg126: memref<1x512x7x7xf16, "cuda">, %arg127: memref<1x512x7x7xf16, "cuda">, %arg128: memref<512x512x3x3xf16, "cuda">, %arg129: memref<1x512x7x7xf16, "cuda">, %arg130: memref<512x256x1x1xf16, "cuda">, %arg131: memref<1x512x7x7xf16, "cuda">, %arg132: memref<1x512x7x7xf16, "cuda">, %arg133: memref<512x512x3x3xf16, "cuda">, %arg134: memref<1x512x7x7xf16, "cuda">, %arg135: memref<1x512x7x7xf16, "cuda">, %arg136: memref<512x512x3x3xf16, "cuda">, %arg137: memref<1x512x7x7xf16, "cuda">, %arg138: memref<1x512x7x7xf16, "cuda">, %arg139: memref<1x512xf16, "cuda">, %arg140: memref<512x1000xf16, "cuda">, %arg141: memref<1x1000xf16, "cuda">) -> (memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x3x7x7xf32, "cuda">, memref<1000xf32, "cuda">, memref<1000x512xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x64x3x3xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128x64x1x1xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x128x3x3xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256x128x1x1xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x256x3x3xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512x256x1x1xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf32, "cuda">) attributes {__placeholder__byre.entry_point} {
    %alloc = memref.alloc() : memref<1x512xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%arg141, %arg140, %alloc) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<1x1000xf16, "cuda">, memref<512x1000xf16, "cuda">, memref<1x512xf16, "cuda">
    %0 = call @Unknown0(%alloc, %arg138) : (memref<1x512xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %alloc_0 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_1 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg137, %arg39, %0, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_0, %arg136, %alloc_3) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %alloc_4 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg135, %alloc_0, %alloc_4) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    %1 = call @Unknown4(%arg135, %alloc_3) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %alloc_5 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_6 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_7 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg134, %arg37, %1, %alloc_5, %alloc_6, %alloc_7) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_8 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_5, %arg133, %alloc_8) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %alloc_9 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg132, %alloc_5, %alloc_9) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    %2 = call @Unknown8(%0, %alloc_8, %arg132) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %alloc_10 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_11 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_12 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg129, %arg33, %2, %alloc_10, %alloc_11, %alloc_12) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_13 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_10, %arg128, %alloc_13) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %alloc_14 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg127, %alloc_10, %alloc_14) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    %3 = call @Unknown4(%arg127, %alloc_13) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %alloc_15 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_16 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_17 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg126, %arg31, %3, %alloc_15, %alloc_16, %alloc_17) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_18 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_15, %arg125, %alloc_18) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_19 = memref.alloc() : memref<512x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg124, %alloc_15, %alloc_19) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">
    %alloc_20 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_21 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_22 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg131, %arg35, %2, %alloc_20, %alloc_21, %alloc_22) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_23 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_20, %arg130, %alloc_23) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_24 = memref.alloc() : memref<512x256x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg124, %alloc_20, %alloc_24) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">
    %4 = call @Unknown19(%alloc_23, %alloc_18, %arg124) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %alloc_25 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_26 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_27 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg123, %arg29, %4, %alloc_25, %alloc_26, %alloc_27) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_28 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_25, %arg122, %alloc_28) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_29 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg121, %alloc_25, %alloc_29) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    %5 = call @Unknown23(%arg121, %alloc_28) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %alloc_30 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_31 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_32 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg120, %arg27, %5, %alloc_30, %alloc_31, %alloc_32) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_33 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_30, %arg119, %alloc_33) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_34 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg118, %alloc_30, %alloc_34) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    %6 = call @Unknown19(%4, %alloc_33, %arg118) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %alloc_35 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_36 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_37 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg115, %arg23, %6, %alloc_35, %alloc_36, %alloc_37) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_38 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_35, %arg114, %alloc_38) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_39 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg113, %alloc_35, %alloc_39) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    %7 = call @Unknown23(%arg113, %alloc_38) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %alloc_40 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_41 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_42 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg112, %arg21, %7, %alloc_40, %alloc_41, %alloc_42) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_43 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_40, %arg111, %alloc_43) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_44 = memref.alloc() : memref<256x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg110, %alloc_40, %alloc_44) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">
    %alloc_45 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_46 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_47 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg117, %arg25, %6, %alloc_45, %alloc_46, %alloc_47) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_48 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_45, %arg116, %alloc_48) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_49 = memref.alloc() : memref<256x128x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg110, %alloc_45, %alloc_49) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">
    %8 = call @Unknown38(%alloc_48, %alloc_43, %arg110) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %alloc_50 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_51 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_52 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg109, %arg19, %8, %alloc_50, %alloc_51, %alloc_52) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_53 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_50, %arg108, %alloc_53) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_54 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg107, %alloc_50, %alloc_54) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    %9 = call @Unknown42(%arg107, %alloc_53) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %alloc_55 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_56 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_57 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg106, %arg17, %9, %alloc_55, %alloc_56, %alloc_57) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_58 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_55, %arg105, %alloc_58) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_59 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg104, %alloc_55, %alloc_59) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    %10 = call @Unknown38(%8, %alloc_58, %arg104) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %alloc_60 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_61 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_62 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg101, %arg13, %10, %alloc_60, %alloc_61, %alloc_62) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_63 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_60, %arg100, %alloc_63) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_64 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg99, %alloc_60, %alloc_64) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    %11 = call @Unknown42(%arg99, %alloc_63) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %alloc_65 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_66 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_67 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg98, %arg11, %11, %alloc_65, %alloc_66, %alloc_67) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_68 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_65, %arg97, %alloc_68) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_69 = memref.alloc() : memref<128x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg96, %alloc_65, %alloc_69) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">
    %alloc_70 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_71 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_72 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg103, %arg15, %10, %alloc_70, %alloc_71, %alloc_72) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_73 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_70, %arg102, %alloc_73) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_74 = memref.alloc() : memref<128x64x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg96, %alloc_70, %alloc_74) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">
    %12 = call @Unknown57(%alloc_73, %alloc_68, %arg96) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %alloc_75 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    %alloc_76 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_77 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg95, %arg9, %12, %alloc_75, %alloc_76, %alloc_77) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_78 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_75, %arg94, %alloc_78) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_79 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg93, %alloc_75, %alloc_79) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    %13 = call @Unknown61(%arg93, %alloc_78) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %alloc_80 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    %alloc_81 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_82 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg92, %arg7, %13, %alloc_80, %alloc_81, %alloc_82) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_83 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_80, %arg91, %alloc_83) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_84 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg90, %alloc_80, %alloc_84) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    %14 = call @Unknown57(%12, %alloc_83, %arg90) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %alloc_85 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    %alloc_86 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_87 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg89, %arg5, %14, %alloc_85, %alloc_86, %alloc_87) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_88 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_85, %arg88, %alloc_88) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_89 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg87, %alloc_85, %alloc_89) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    %15 = call @Unknown61(%arg87, %alloc_88) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %alloc_90 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    %alloc_91 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_92 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg86, %arg3, %15, %alloc_90, %alloc_91, %alloc_92) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_93 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_90, %arg85, %alloc_93) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_94 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg84, %alloc_90, %alloc_94) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    %16 = call @Unknown73(%14, %alloc_93) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %alloc_95 = memref.alloc() : memref<1x64x112x112xf16, "cuda">
    byre.compute @PoolMaxGradOp_f16f16_f16(%arg83, %16, %alloc_95) {device = "cuda", memory_effects = [1 : i32, 1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<1x64x112x112xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x112x112xf16, "cuda">
    %17 = call @Unknown74(%arg83, %alloc_95) : (memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">) -> memref<1x64x112x112xf16, "cuda">
    %alloc_96 = memref.alloc() : memref<1x64x112x112xf16, "cuda">
    %alloc_97 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_98 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg82, %arg1, %17, %alloc_96, %alloc_97, %alloc_98) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_99 = memref.alloc() : memref<64x3x7x7xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg81, %alloc_96, %alloc_99) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x3x224x224xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<64x3x7x7xf16, "cuda">
    %18 = call @Unknown77(%alloc_99) : (memref<64x3x7x7xf16, "cuda">) -> memref<64x3x7x7xf32, "cuda">
    %19 = call @Unknown78(%arg141) : (memref<1x1000xf16, "cuda">) -> memref<1000xf32, "cuda">
    %collapse_shape = memref.collapse_shape %arg141 [[0, 1]] : memref<1x1000xf16, "cuda"> into memref<1000xf16, "cuda">
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] : memref<1000xf16, "cuda"> into memref<1000x1xf16, "cuda">
    %alloc_100 = memref.alloc() : memref<1000x512xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%expand_shape, %arg139, %alloc_100) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<1000x1xf16, "cuda">, memref<1x512xf16, "cuda">, memref<1000x512xf16, "cuda">
    %20 = call @Unknown79(%alloc_100) : (memref<1000x512xf16, "cuda">) -> memref<1000x512xf32, "cuda">
    %21 = call @Unknown80(%alloc_94) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %22 = call @Unknown80(%alloc_89) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %23 = call @Unknown80(%alloc_84) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %24 = call @Unknown80(%alloc_79) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %25 = call @Unknown84(%alloc_69) : (memref<128x64x3x3xf16, "cuda">) -> memref<128x64x3x3xf32, "cuda">
    %26 = call @Unknown85(%alloc_64) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %27 = call @Unknown86(%alloc_74) : (memref<128x64x1x1xf16, "cuda">) -> memref<128x64x1x1xf32, "cuda">
    %28 = call @Unknown85(%alloc_59) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %29 = call @Unknown85(%alloc_54) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %30 = call @Unknown89(%alloc_44) : (memref<256x128x3x3xf16, "cuda">) -> memref<256x128x3x3xf32, "cuda">
    %31 = call @Unknown90(%alloc_39) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %32 = call @Unknown91(%alloc_49) : (memref<256x128x1x1xf16, "cuda">) -> memref<256x128x1x1xf32, "cuda">
    %33 = call @Unknown90(%alloc_34) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %34 = call @Unknown90(%alloc_29) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %35 = call @Unknown94(%alloc_19) : (memref<512x256x3x3xf16, "cuda">) -> memref<512x256x3x3xf32, "cuda">
    %36 = call @Unknown95(%alloc_14) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %37 = call @Unknown96(%alloc_24) : (memref<512x256x1x1xf16, "cuda">) -> memref<512x256x1x1xf32, "cuda">
    %38 = call @Unknown95(%alloc_9) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %39 = call @Unknown95(%alloc_4) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    return %alloc_98, %alloc_97, %18, %19, %20, %alloc_92, %alloc_91, %alloc_87, %alloc_86, %21, %22, %alloc_82, %alloc_81, %alloc_77, %alloc_76, %23, %24, %alloc_67, %alloc_66, %alloc_62, %alloc_61, %25, %26, %27, %alloc_72, %alloc_71, %alloc_57, %alloc_56, %alloc_52, %alloc_51, %28, %29, %alloc_42, %alloc_41, %alloc_37, %alloc_36, %30, %31, %32, %alloc_47, %alloc_46, %alloc_32, %alloc_31, %alloc_27, %alloc_26, %33, %34, %alloc_17, %alloc_16, %alloc_12, %alloc_11, %35, %36, %37, %alloc_22, %alloc_21, %alloc_7, %alloc_6, %alloc_2, %alloc_1, %38, %39 : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x3x7x7xf32, "cuda">, memref<1000xf32, "cuda">, memref<1000x512xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x64x3x3xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128x64x1x1xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x128x3x3xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256x128x1x1xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x256x3x3xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512x256x1x1xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf32, "cuda">
  }
}