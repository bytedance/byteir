// RUN: byteir-opt %s -byre-opt="append-arg-types entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

module @IrToMhlo.2452 attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown172(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) kernel {
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c1000 step %6 {
        %7 = memref.load %arg0[%arg2] : memref<1000xf32>
        %8 = arith.truncf %7 : f32 to f16
        %9 = arith.extf %8 : f16 to f32
        memref.store %9, %arg1[%arg2] : memref<1000xf32>
      }
      gpu.return
    }
    gpu.func @Unknown170(%arg0: memref<1000x512xf16>, %arg1: memref<1000x512xf32>) kernel {
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
    gpu.func @Unknown166(%arg0: memref<512x256x1x1xf16>, %arg1: memref<512x256x1x1xf32>) kernel {
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
    gpu.func @Unknown165(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
    gpu.func @Unknown164(%arg0: memref<512x256x3x3xf16>, %arg1: memref<512x256x3x3xf32>) kernel {
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
    gpu.func @Unknown161(%arg0: memref<256x128x1x1xf16>, %arg1: memref<256x128x1x1xf32>) kernel {
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
    gpu.func @Unknown160(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
    gpu.func @Unknown159(%arg0: memref<256x128x3x3xf16>, %arg1: memref<256x128x3x3xf32>) kernel {
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
    gpu.func @Unknown156(%arg0: memref<128x64x1x1xf16>, %arg1: memref<128x64x1x1xf32>) kernel {
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
    gpu.func @Unknown155(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
    gpu.func @Unknown154(%arg0: memref<128x64x3x3xf16>, %arg1: memref<128x64x3x3xf32>) kernel {
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
    gpu.func @Unknown150(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
    gpu.func @Unknown149(%arg0: memref<64x3x7x7xf16>, %arg1: memref<64x3x7x7xf32>) kernel {
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
    gpu.func @Unknown148(%arg0: memref<f32>, %arg1: memref<f32>) kernel {
      %c1 = arith.constant 1 : index
      %cst = arith.constant 4.000000e+00 : f32
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c1 step %6 {
        %7 = memref.load %arg0[] : memref<f32>
        %8 = arith.negf %7 : f32
        %9 = arith.divf %8, %cst : f32
        memref.store %9, %arg1[] : memref<f32>
      }
      gpu.return
    }
    gpu.func @Unknown144(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>, %arg2: memref<4x64x112x112xf16>) kernel {
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c112 = arith.constant 112 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c3211264 step %6 {
        %7 = arith.remsi %arg3, %c112 : index
        %8 = arith.divsi %arg3, %c112 : index
        %9 = arith.remsi %8, %c112 : index
        %10 = arith.divsi %8, %c112 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x64x112x112xi1>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x64x112x112xf16>
        %15 = arith.select %13, %14, %cst : f16
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x64x112x112xf16>
      }
      gpu.return
    }
    gpu.func @Unknown143(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xf16>) kernel {
      %c802816 = arith.constant 802816 : index
      %c64 = arith.constant 64 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg3, %c56 : index
        %8 = arith.divsi %arg3, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        %15 = arith.addf %13, %14 : f16
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown131(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xf16>) kernel {
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg3, %c56 : index
        %8 = arith.divsi %arg3, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x64x56x56xi1>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        %15 = arith.select %13, %14, %cst : f16
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown127(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>, %arg3: memref<4x64x56x56xf16>) kernel {
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg4, %c56 : index
        %8 = arith.divsi %arg4, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        %15 = memref.load %arg2[%12, %11, %9, %7] : memref<4x64x56x56xi1>
        %16 = arith.addf %13, %14 : f16
        %17 = arith.select %15, %16, %cst : f16
        memref.store %17, %arg3[%12, %11, %9, %7] : memref<4x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown112(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xf16>) kernel {
      %c401408 = arith.constant 401408 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c28 = arith.constant 28 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c401408 step %6 {
        %7 = arith.remsi %arg3, %c28 : index
        %8 = arith.divsi %arg3, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = arith.remsi %10, %c128 : index
        %12 = arith.divsi %10, %c128 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x128x28x28xi1>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x128x28x28xf16>
        %15 = arith.select %13, %14, %cst : f16
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown108(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>, %arg3: memref<4x128x28x28xf16>) kernel {
      %c401408 = arith.constant 401408 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c28 = arith.constant 28 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c401408 step %6 {
        %7 = arith.remsi %arg4, %c28 : index
        %8 = arith.divsi %arg4, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = arith.remsi %10, %c128 : index
        %12 = arith.divsi %10, %c128 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x128x28x28xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x128x28x28xf16>
        %15 = memref.load %arg2[%12, %11, %9, %7] : memref<4x128x28x28xi1>
        %16 = arith.addf %13, %14 : f16
        %17 = arith.select %15, %16, %cst : f16
        memref.store %17, %arg3[%12, %11, %9, %7] : memref<4x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown93(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xf16>) kernel {
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c14 = arith.constant 14 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg3, %c14 : index
        %8 = arith.divsi %arg3, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = arith.remsi %10, %c256 : index
        %12 = arith.divsi %10, %c256 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x256x14x14xi1>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x256x14x14xf16>
        %15 = arith.select %13, %14, %cst : f16
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown89(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>, %arg3: memref<4x256x14x14xf16>) kernel {
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c14 = arith.constant 14 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg4, %c14 : index
        %8 = arith.divsi %arg4, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = arith.remsi %10, %c256 : index
        %12 = arith.divsi %10, %c256 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x256x14x14xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x256x14x14xf16>
        %15 = memref.load %arg2[%12, %11, %9, %7] : memref<4x256x14x14xi1>
        %16 = arith.addf %13, %14 : f16
        %17 = arith.select %15, %16, %cst : f16
        memref.store %17, %arg3[%12, %11, %9, %7] : memref<4x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown78(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>, %arg3: memref<4x512x7x7xf16>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg4, %c7 : index
        %8 = arith.divsi %arg4, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = arith.remsi %10, %c512 : index
        %12 = arith.divsi %10, %c512 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x512x7x7xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x512x7x7xf16>
        %15 = memref.load %arg2[%12, %11, %9, %7] : memref<4x512x7x7xi1>
        %16 = arith.addf %13, %14 : f16
        %17 = arith.select %15, %16, %cst : f16
        memref.store %17, %arg3[%12, %11, %9, %7] : memref<4x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown74(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xf16>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg3, %c7 : index
        %8 = arith.divsi %arg3, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = arith.remsi %10, %c512 : index
        %12 = arith.divsi %10, %c512 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x512x7x7xi1>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x512x7x7xf16>
        %15 = arith.select %13, %14, %cst : f16
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown70(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>, %arg2: memref<4x512x7x7xf16>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 4.900000e+01 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg3, %c7 : index
        %8 = arith.divsi %arg3, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = arith.remsi %10, %c512 : index
        %12 = arith.divsi %10, %c512 : index
        %13 = memref.load %arg0[%12, %11] : memref<4x512xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x512x7x7xi1>
        %15 = arith.divf %13, %cst : f16
        %16 = arith.select %14, %15, %cst_0 : f16
        memref.store %16, %arg2[%12, %11, %9, %7] : memref<4x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown69(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>, %arg4: memref<4x1000xf16>, %arg5: memref<4x1000xf16>) kernel {
      %c4000 = arith.constant 4000 : index
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg6 = %4 to %c4000 step %6 {
        %7 = arith.remsi %arg6, %c1000 : index
        %8 = arith.divsi %arg6, %c1000 : index
        %9 = memref.load %arg2[%8] : memref<4xf16>
        %10 = memref.load %arg0[%8] : memref<4xf16>
        %11 = memref.load %arg1[%8, %7] : memref<4x1000xf16>
        %12 = memref.load %arg3[%8, %7] : memref<4x1000xf16>
        %13 = arith.subf %11, %10 : f16
        %14 = math.exp %13 : f16
        %15 = arith.mulf %14, %9 : f16
        %16 = arith.subf %12, %15 : f16
        memref.store %13, %arg4[%8, %7] : memref<4x1000xf16>
        memref.store %16, %arg5[%8, %7] : memref<4x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown68(%arg0: memref<4xf16>, %arg1: memref<4xf16>) kernel {
      %c4 = arith.constant 4 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c4 step %6 {
        %7 = memref.load %arg0[%arg2] : memref<4xf16>
        %8 = math.log %7 : f16
        memref.store %8, %arg1[%arg2] : memref<4xf16>
      }
      gpu.return
    }
    gpu.func @Unknown66(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4x1000xf16>) kernel {
      %c4000 = arith.constant 4000 : index
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c4000 step %6 {
        %7 = arith.remsi %arg3, %c1000 : index
        %8 = arith.divsi %arg3, %c1000 : index
        %9 = memref.load %arg0[%8] : memref<4xf16>
        %10 = memref.load %arg1[%8, %7] : memref<4x1000xf16>
        %11 = arith.subf %10, %9 : f16
        memref.store %11, %arg2[%8, %7] : memref<4x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown64(%arg0: memref<1000xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4x1000xf16>) kernel {
      %c4000 = arith.constant 4000 : index
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c4000 step %6 {
        %7 = arith.remsi %arg3, %c1000 : index
        %8 = arith.divsi %arg3, %c1000 : index
        %9 = memref.load %arg0[%7] : memref<1000xf16>
        %10 = memref.load %arg1[%8, %7] : memref<4x1000xf16>
        %11 = arith.addf %10, %9 : f16
        memref.store %11, %arg2[%8, %7] : memref<4x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown63(%arg0: memref<4x512xf16>, %arg1: memref<4x512xf16>) kernel {
      %c2048 = arith.constant 2048 : index
      %cst = arith.constant 2.040100e-02 : f16
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c2048 step %6 {
        %7 = arith.remsi %arg2, %c512 : index
        %8 = arith.divsi %arg2, %c512 : index
        %9 = memref.load %arg0[%8, %7] : memref<4x512xf16>
        %10 = arith.mulf %9, %cst : f16
        memref.store %10, %arg1[%8, %7] : memref<4x512xf16>
      }
      gpu.return
    }
    gpu.func @Unknown57(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xf16>, %arg3: memref<4x512x7x7xi1>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg4, %c7 : index
        %8 = arith.divsi %arg4, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = arith.remsi %10, %c512 : index
        %12 = arith.divsi %10, %c512 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x512x7x7xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x512x7x7xf16>
        %15 = arith.addf %13, %14 : f16
        %16 = arith.maximumf %15, %cst : f16
        %17 = arith.cmpf ogt, %16, %cst : f16
        memref.store %16, %arg2[%12, %11, %9, %7] : memref<4x512x7x7xf16>
        memref.store %17, %arg3[%12, %11, %9, %7] : memref<4x512x7x7xi1>
      }
      gpu.return
    }
    gpu.func @Unknown55(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg3, %c7 : index
        %8 = arith.divsi %arg3, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = arith.remsi %10, %c512 : index
        %12 = arith.divsi %10, %c512 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x512x7x7xf16>
        %14 = arith.maximumf %13, %cst : f16
        %15 = arith.cmpf ogt, %14, %cst : f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<4x512x7x7xf16>
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x512x7x7xi1>
      }
      gpu.return
    }
    gpu.func @Unknown48(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xf16>, %arg3: memref<4x256x14x14xi1>) kernel {
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c14 = arith.constant 14 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg4, %c14 : index
        %8 = arith.divsi %arg4, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = arith.remsi %10, %c256 : index
        %12 = arith.divsi %10, %c256 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x256x14x14xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x256x14x14xf16>
        %15 = arith.addf %13, %14 : f16
        %16 = arith.maximumf %15, %cst : f16
        %17 = arith.cmpf ogt, %16, %cst : f16
        memref.store %16, %arg2[%12, %11, %9, %7] : memref<4x256x14x14xf16>
        memref.store %17, %arg3[%12, %11, %9, %7] : memref<4x256x14x14xi1>
      }
      gpu.return
    }
    gpu.func @Unknown46(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) kernel {
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c14 = arith.constant 14 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg3, %c14 : index
        %8 = arith.divsi %arg3, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = arith.remsi %10, %c256 : index
        %12 = arith.divsi %10, %c256 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x256x14x14xf16>
        %14 = arith.maximumf %13, %cst : f16
        %15 = arith.cmpf ogt, %14, %cst : f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<4x256x14x14xf16>
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x256x14x14xi1>
      }
      gpu.return
    }
    gpu.func @Unknown39(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xf16>, %arg3: memref<4x128x28x28xi1>) kernel {
      %c401408 = arith.constant 401408 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c28 = arith.constant 28 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c401408 step %6 {
        %7 = arith.remsi %arg4, %c28 : index
        %8 = arith.divsi %arg4, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = arith.remsi %10, %c128 : index
        %12 = arith.divsi %10, %c128 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x128x28x28xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x128x28x28xf16>
        %15 = arith.addf %13, %14 : f16
        %16 = arith.maximumf %15, %cst : f16
        %17 = arith.cmpf ogt, %16, %cst : f16
        memref.store %16, %arg2[%12, %11, %9, %7] : memref<4x128x28x28xf16>
        memref.store %17, %arg3[%12, %11, %9, %7] : memref<4x128x28x28xi1>
      }
      gpu.return
    }
    gpu.func @Unknown37(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) kernel {
      %c401408 = arith.constant 401408 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c28 = arith.constant 28 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c401408 step %6 {
        %7 = arith.remsi %arg3, %c28 : index
        %8 = arith.divsi %arg3, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = arith.remsi %10, %c128 : index
        %12 = arith.divsi %10, %c128 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x128x28x28xf16>
        %14 = arith.maximumf %13, %cst : f16
        %15 = arith.cmpf ogt, %14, %cst : f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<4x128x28x28xf16>
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x128x28x28xi1>
      }
      gpu.return
    }
    gpu.func @Unknown30(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xf16>, %arg3: memref<4x64x56x56xi1>) kernel {
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg4 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg4, %c56 : index
        %8 = arith.divsi %arg4, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        %14 = memref.load %arg1[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        %15 = arith.addf %13, %14 : f16
        %16 = arith.maximumf %15, %cst : f16
        %17 = arith.cmpf ogt, %16, %cst : f16
        memref.store %16, %arg2[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        memref.store %17, %arg3[%12, %11, %9, %7] : memref<4x64x56x56xi1>
      }
      gpu.return
    }
    gpu.func @Unknown28(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) kernel {
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg3, %c56 : index
        %8 = arith.divsi %arg3, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        %14 = arith.maximumf %13, %cst : f16
        %15 = arith.cmpf ogt, %14, %cst : f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<4x64x56x56xf16>
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x64x56x56xi1>
      }
      gpu.return
    }
    gpu.func @Unknown26(%arg0: memref<4x64x112x112xf16>, %arg1: memref<4x64x112x112xf16>, %arg2: memref<4x64x112x112xi1>) kernel {
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c112 = arith.constant 112 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c3211264 step %6 {
        %7 = arith.remsi %arg3, %c112 : index
        %8 = arith.divsi %arg3, %c112 : index
        %9 = arith.remsi %8, %c112 : index
        %10 = arith.divsi %8, %c112 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x64x112x112xf16>
        %14 = arith.maximumf %13, %cst : f16
        %15 = arith.cmpf ogt, %14, %cst : f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<4x64x112x112xf16>
        memref.store %15, %arg2[%12, %11, %9, %7] : memref<4x64x112x112xi1>
      }
      gpu.return
    }
    gpu.func @Unknown24(%arg0: memref<1000xf32>, %arg1: memref<1000xf16>) kernel {
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c1000 step %6 {
        %7 = memref.load %arg0[%arg2] : memref<1000xf32>
        %8 = arith.truncf %7 : f32 to f16
        memref.store %8, %arg1[%arg2] : memref<1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown23(%arg0: memref<1000x512xf32>, %arg1: memref<1000x512xf16>) kernel {
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
        %9 = memref.load %arg0[%8, %7] : memref<1000x512xf32>
        %10 = arith.truncf %9 : f32 to f16
        memref.store %10, %arg1[%8, %7] : memref<1000x512xf16>
      }
      gpu.return
    }
    gpu.func @Unknown22(%arg0: memref<4x1000xf32>, %arg1: memref<4x1000xf16>) kernel {
      %c4000 = arith.constant 4000 : index
      %cst = arith.constant -2.500000e-01 : f32
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c4000 step %6 {
        %7 = arith.remsi %arg2, %c1000 : index
        %8 = arith.divsi %arg2, %c1000 : index
        %9 = memref.load %arg0[%8, %7] : memref<4x1000xf32>
        %10 = arith.mulf %9, %cst : f32
        %11 = arith.truncf %10 : f32 to f16
        memref.store %11, %arg1[%8, %7] : memref<4x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown19(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<512x512x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown18(%arg0: memref<512x256x3x3xf32>, %arg1: memref<512x256x3x3xf16>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<512x256x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<512x256x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown17(%arg0: memref<512x256x1x1xf32>, %arg1: memref<512x256x1x1xf16>) kernel {
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
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<512x256x1x1xf32>
        %10 = arith.truncf %9 : f32 to f16
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<512x256x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown14(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<256x256x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown13(%arg0: memref<256x128x3x3xf32>, %arg1: memref<256x128x3x3xf16>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<256x128x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<256x128x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown12(%arg0: memref<256x128x1x1xf32>, %arg1: memref<256x128x1x1xf16>) kernel {
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
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<256x128x1x1xf32>
        %10 = arith.truncf %9 : f32 to f16
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<256x128x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown9(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<128x128x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown8(%arg0: memref<128x64x3x3xf32>, %arg1: memref<128x64x3x3xf16>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<128x64x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<128x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown7(%arg0: memref<128x64x1x1xf32>, %arg1: memref<128x64x1x1xf16>) kernel {
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
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<128x64x1x1xf32>
        %10 = arith.truncf %9 : f32 to f16
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<128x64x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown3(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<64x64x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown1(%arg0: memref<64x3x7x7xf32>, %arg1: memref<64x3x7x7xf16>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<64x3x7x7xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<64x3x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<4x3x224x224xf32>, %arg1: memref<4x3x224x224xf16>) kernel {
      %c602112 = arith.constant 602112 : index
      %c3 = arith.constant 3 : index
      %c224 = arith.constant 224 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c602112 step %6 {
        %7 = arith.remsi %arg2, %c224 : index
        %8 = arith.divsi %arg2, %c224 : index
        %9 = arith.remsi %8, %c224 : index
        %10 = arith.divsi %8, %c224 : index
        %11 = arith.remsi %10, %c3 : index
        %12 = arith.divsi %10, %c3 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<4x3x224x224xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<4x3x224x224xf16>
      }
      gpu.return
    }
    gpu.func @Unknown25_kernel(%arg0: memref<4x1000xf16>, %arg1: memref<4xf16>) kernel attributes {gpu.known_block_size = array<i32: 512, 1, 1>, gpu.known_grid_size = array<i32: 4, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1000 = arith.constant 1000 : index
      %c-1024 = arith.constant -1024 : index
      %c512 = arith.constant 512 : index
      %c-1 = arith.constant -1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %0 = gpu.block_id  x
      %subview = memref.subview %arg0[%0, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      %1 = gpu.thread_id  x
      %2 = arith.muli %1, %c2 : index
      %3 = arith.cmpi slt, %1, %c0 : index
      %4 = arith.subi %c-1, %1 : index
      %5 = arith.select %3, %4, %1 : index
      %6 = arith.divsi %5, %c512 : index
      %7 = arith.subi %c-1, %6 : index
      %8 = arith.select %3, %7, %6 : index
      %9 = arith.muli %8, %c-1024 : index
      %10 = arith.addi %2, %9 : index
      %11 = arith.cmpi slt, %10, %c1000 : index
      %12 = arith.select %11, %10, %c1000 : index
      %13 = arith.addi %10, %c2 : index
      %14 = arith.cmpi slt, %13, %c1000 : index
      %15 = arith.select %14, %13, %c1000 : index
      %16 = arith.subi %15, %12 : index
      %subview_0 = memref.subview %expand_shape[0, %12] [1, %16] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %17 = arith.cmpi ugt, %16, %c0 : index
      %18 = scf.if %17 -> (f16) {
        %32 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %32 : f16
      } else {
        scf.yield %cst : f16
      }
      %19 = arith.addf %18, %cst : f16
      %20 = arith.cmpi ugt, %16, %c1 : index
      %21 = scf.if %20 -> (f16) {
        %32 = memref.load %expand_shape_1[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %32 : f16
      } else {
        scf.yield %cst : f16
      }
      %22 = arith.addf %19, %21 : f16
      memref.store %22, %alloca[%1] : memref<512xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      %23 = arith.cmpi ult, %1, %c256 : index
      scf.if %23 {
        %32 = memref.load %alloca[%2] : memref<512xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca[%34] : memref<512xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %alloca_2[%1] : memref<256xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      %24 = arith.cmpi ult, %1, %c128 : index
      scf.if %24 {
        %32 = memref.load %alloca_2[%2] : memref<256xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca_2[%34] : memref<256xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %alloca_3[%1] : memref<128xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %25 = arith.cmpi ult, %1, %c64 : index
      scf.if %25 {
        %32 = memref.load %alloca_3[%2] : memref<128xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca_3[%34] : memref<128xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %alloca_4[%1] : memref<64xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %26 = arith.cmpi ult, %1, %c32 : index
      scf.if %26 {
        %32 = memref.load %alloca_4[%2] : memref<64xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca_4[%34] : memref<64xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %alloca_5[%1] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %27 = arith.cmpi ult, %1, %c16 : index
      scf.if %27 {
        %32 = memref.load %alloca_5[%2] : memref<32xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca_5[%34] : memref<32xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %alloca_6[%1] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_7 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %28 = arith.cmpi ult, %1, %c8 : index
      scf.if %28 {
        %32 = memref.load %alloca_6[%2] : memref<16xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca_6[%34] : memref<16xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %alloca_7[%1] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_8 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %29 = arith.cmpi ult, %1, %c4 : index
      scf.if %29 {
        %32 = memref.load %alloca_7[%2] : memref<8xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca_7[%34] : memref<8xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %alloca_8[%1] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_9 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %30 = arith.cmpi ult, %1, %c2 : index
      scf.if %30 {
        %32 = memref.load %alloca_8[%2] : memref<4xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca_8[%34] : memref<4xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %alloca_9[%1] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %31 = arith.cmpi ult, %1, %c1 : index
      scf.if %31 {
        %32 = memref.load %alloca_9[%2] : memref<2xf16, #gpu.address_space<workgroup>>
        %33 = arith.addf %32, %cst : f16
        %34 = arith.addi %2, %c1 : index
        %35 = memref.load %alloca_9[%34] : memref<2xf16, #gpu.address_space<workgroup>>
        %36 = arith.addf %35, %33 : f16
        memref.store %36, %arg1[%0] : memref<4xf16>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown62_kernel(%arg0: memref<2048x49xf16>, %arg1: memref<2048xf16>) kernel attributes {gpu.known_block_size = array<i32: 64, 1, 1>, gpu.known_grid_size = array<i32: 2048, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c49 = arith.constant 49 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %subview = memref.subview %arg0[%0, 0] [1, 49] [1, 1] : memref<2048x49xf16> to memref<49xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<49xf16, strided<[1], offset: ?>> into memref<1x49xf16, strided<[49, 1], offset: ?>>
      %alloca = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %1 = gpu.thread_id  x
      %2 = arith.remsi %1, %c64 : index
      %3 = arith.cmpi slt, %2, %c0 : index
      %4 = arith.addi %2, %c64 : index
      %5 = arith.select %3, %4, %2 : index
      %6 = arith.cmpi slt, %5, %c49 : index
      %7 = arith.select %6, %5, %c49 : index
      %8 = arith.addi %5, %c1 : index
      %9 = arith.cmpi slt, %8, %c49 : index
      %10 = arith.select %9, %8, %c49 : index
      %11 = arith.subi %10, %7 : index
      %subview_0 = memref.subview %expand_shape[0, %7] [1, %11] [1, 1] : memref<1x49xf16, strided<[49, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %12 = arith.cmpi ugt, %11, %c0 : index
      %13 = scf.if %12 -> (f16) {
        %21 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %21 : f16
      } else {
        scf.yield %cst : f16
      }
      %14 = arith.addf %13, %cst : f16
      memref.store %14, %alloca[%1] : memref<64xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %15 = arith.cmpi ult, %1, %c32 : index
      scf.if %15 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca[%21] : memref<64xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca[%24] : memref<64xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_2[%1] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %16 = arith.cmpi ult, %1, %c16 : index
      scf.if %16 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_2[%21] : memref<32xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_2[%24] : memref<32xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_3[%1] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %17 = arith.cmpi ult, %1, %c8 : index
      scf.if %17 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_3[%21] : memref<16xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_3[%24] : memref<16xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_4[%1] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %18 = arith.cmpi ult, %1, %c4 : index
      scf.if %18 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_4[%21] : memref<8xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_4[%24] : memref<8xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_5[%1] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %19 = arith.cmpi ult, %1, %c2 : index
      scf.if %19 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_5[%21] : memref<4xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_5[%24] : memref<4xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_6[%1] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %20 = arith.cmpi ult, %1, %c1 : index
      scf.if %20 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_6[%21] : memref<2xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_6[%24] : memref<2xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %arg1[%0] : memref<2048xf16>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown65_kernel(%arg0: memref<4x1000xf16>, %arg1: memref<4xf16>) kernel attributes {gpu.known_block_size = array<i32: 512, 1, 1>, gpu.known_grid_size = array<i32: 4, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1000 = arith.constant 1000 : index
      %c-1024 = arith.constant -1024 : index
      %c512 = arith.constant 512 : index
      %c-1 = arith.constant -1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %0 = gpu.block_id  x
      %subview = memref.subview %arg0[%0, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      %1 = gpu.thread_id  x
      %2 = arith.muli %1, %c2 : index
      %3 = arith.cmpi slt, %1, %c0 : index
      %4 = arith.subi %c-1, %1 : index
      %5 = arith.select %3, %4, %1 : index
      %6 = arith.divsi %5, %c512 : index
      %7 = arith.subi %c-1, %6 : index
      %8 = arith.select %3, %7, %6 : index
      %9 = arith.muli %8, %c-1024 : index
      %10 = arith.addi %2, %9 : index
      %11 = arith.cmpi slt, %10, %c1000 : index
      %12 = arith.select %11, %10, %c1000 : index
      %13 = arith.addi %10, %c2 : index
      %14 = arith.cmpi slt, %13, %c1000 : index
      %15 = arith.select %14, %13, %c1000 : index
      %16 = arith.subi %15, %12 : index
      %subview_0 = memref.subview %expand_shape[0, %12] [1, %16] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %17 = arith.cmpi ugt, %16, %c0 : index
      %18 = scf.if %17 -> (f16) {
        %31 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %31 : f16
      } else {
        scf.yield %cst : f16
      }
      %19 = arith.cmpi ugt, %16, %c1 : index
      %20 = scf.if %19 -> (f16) {
        %31 = memref.load %expand_shape_1[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %31 : f16
      } else {
        scf.yield %cst : f16
      }
      %21 = arith.maximumf %18, %20 : f16
      memref.store %21, %alloca[%1] : memref<512xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      %22 = arith.cmpi ult, %1, %c256 : index
      scf.if %22 {
        %31 = memref.load %alloca[%2] : memref<512xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca[%32] : memref<512xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %alloca_2[%1] : memref<256xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      %23 = arith.cmpi ult, %1, %c128 : index
      scf.if %23 {
        %31 = memref.load %alloca_2[%2] : memref<256xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca_2[%32] : memref<256xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %alloca_3[%1] : memref<128xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %24 = arith.cmpi ult, %1, %c64 : index
      scf.if %24 {
        %31 = memref.load %alloca_3[%2] : memref<128xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca_3[%32] : memref<128xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %alloca_4[%1] : memref<64xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %25 = arith.cmpi ult, %1, %c32 : index
      scf.if %25 {
        %31 = memref.load %alloca_4[%2] : memref<64xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca_4[%32] : memref<64xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %alloca_5[%1] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %26 = arith.cmpi ult, %1, %c16 : index
      scf.if %26 {
        %31 = memref.load %alloca_5[%2] : memref<32xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca_5[%32] : memref<32xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %alloca_6[%1] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_7 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %27 = arith.cmpi ult, %1, %c8 : index
      scf.if %27 {
        %31 = memref.load %alloca_6[%2] : memref<16xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca_6[%32] : memref<16xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %alloca_7[%1] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_8 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %28 = arith.cmpi ult, %1, %c4 : index
      scf.if %28 {
        %31 = memref.load %alloca_7[%2] : memref<8xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca_7[%32] : memref<8xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %alloca_8[%1] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_9 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %29 = arith.cmpi ult, %1, %c2 : index
      scf.if %29 {
        %31 = memref.load %alloca_8[%2] : memref<4xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca_8[%32] : memref<4xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %alloca_9[%1] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %30 = arith.cmpi ult, %1, %c1 : index
      scf.if %30 {
        %31 = memref.load %alloca_9[%2] : memref<2xf16, #gpu.address_space<workgroup>>
        %32 = arith.addi %2, %c1 : index
        %33 = memref.load %alloca_9[%32] : memref<2xf16, #gpu.address_space<workgroup>>
        %34 = arith.maximumf %33, %31 : f16
        memref.store %34, %arg1[%0] : memref<4xf16>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown67_kernel(%arg0: memref<4x1000xf16>, %arg1: memref<4xf16>) kernel attributes {gpu.known_block_size = array<i32: 512, 1, 1>, gpu.known_grid_size = array<i32: 4, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1000 = arith.constant 1000 : index
      %c-1024 = arith.constant -1024 : index
      %c512 = arith.constant 512 : index
      %c-1 = arith.constant -1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %0 = gpu.block_id  x
      %subview = memref.subview %arg0[%0, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      %1 = gpu.thread_id  x
      %2 = arith.muli %1, %c2 : index
      %3 = arith.cmpi slt, %1, %c0 : index
      %4 = arith.subi %c-1, %1 : index
      %5 = arith.select %3, %4, %1 : index
      %6 = arith.divsi %5, %c512 : index
      %7 = arith.subi %c-1, %6 : index
      %8 = arith.select %3, %7, %6 : index
      %9 = arith.muli %8, %c-1024 : index
      %10 = arith.addi %2, %9 : index
      %11 = arith.cmpi slt, %10, %c1000 : index
      %12 = arith.select %11, %10, %c1000 : index
      %13 = arith.addi %10, %c2 : index
      %14 = arith.cmpi slt, %13, %c1000 : index
      %15 = arith.select %14, %13, %c1000 : index
      %16 = arith.subi %15, %12 : index
      %subview_0 = memref.subview %expand_shape[0, %12] [1, %16] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %17 = arith.cmpi ugt, %16, %c0 : index
      %18 = scf.if %17 -> (f16) {
        %34 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %34 : f16
      } else {
        scf.yield %cst : f16
      }
      %19 = math.exp %18 : f16
      %20 = arith.addf %19, %cst : f16
      %21 = arith.cmpi ugt, %16, %c1 : index
      %22 = scf.if %21 -> (f16) {
        %34 = memref.load %expand_shape_1[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %34 : f16
      } else {
        scf.yield %cst : f16
      }
      %23 = math.exp %22 : f16
      %24 = arith.addf %20, %23 : f16
      memref.store %24, %alloca[%1] : memref<512xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      %25 = arith.cmpi ult, %1, %c256 : index
      scf.if %25 {
        %34 = memref.load %alloca[%2] : memref<512xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca[%36] : memref<512xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_2[%1] : memref<256xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      %26 = arith.cmpi ult, %1, %c128 : index
      scf.if %26 {
        %34 = memref.load %alloca_2[%2] : memref<256xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca_2[%36] : memref<256xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_3[%1] : memref<128xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %27 = arith.cmpi ult, %1, %c64 : index
      scf.if %27 {
        %34 = memref.load %alloca_3[%2] : memref<128xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca_3[%36] : memref<128xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_4[%1] : memref<64xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %28 = arith.cmpi ult, %1, %c32 : index
      scf.if %28 {
        %34 = memref.load %alloca_4[%2] : memref<64xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca_4[%36] : memref<64xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_5[%1] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %29 = arith.cmpi ult, %1, %c16 : index
      scf.if %29 {
        %34 = memref.load %alloca_5[%2] : memref<32xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca_5[%36] : memref<32xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_6[%1] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_7 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %30 = arith.cmpi ult, %1, %c8 : index
      scf.if %30 {
        %34 = memref.load %alloca_6[%2] : memref<16xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca_6[%36] : memref<16xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_7[%1] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_8 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %31 = arith.cmpi ult, %1, %c4 : index
      scf.if %31 {
        %34 = memref.load %alloca_7[%2] : memref<8xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca_7[%36] : memref<8xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_8[%1] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_9 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %32 = arith.cmpi ult, %1, %c2 : index
      scf.if %32 {
        %34 = memref.load %alloca_8[%2] : memref<4xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca_8[%36] : memref<4xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_9[%1] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %33 = arith.cmpi ult, %1, %c1 : index
      scf.if %33 {
        %34 = memref.load %alloca_9[%2] : memref<2xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %2, %c1 : index
        %37 = memref.load %alloca_9[%36] : memref<2xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %arg1[%0] : memref<4xf16>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown147_kernel(%arg0: memref<32x125xf16>, %arg1: memref<32x125xf32>, %arg2: memref<32xf32>) kernel attributes {gpu.known_block_size = array<i32: 128, 1, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 0.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c125 = arith.constant 125 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %subview = memref.subview %arg0[%0, 0] [1, 125] [1, 1] : memref<32x125xf16> to memref<125xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<125xf16, strided<[1], offset: ?>> into memref<1x125xf16, strided<[125, 1], offset: ?>>
      %subview_1 = memref.subview %arg1[%0, 0] [1, 125] [1, 1] : memref<32x125xf32> to memref<125xf32, strided<[1], offset: ?>>
      %expand_shape_2 = memref.expand_shape %subview_1 [[0, 1]] : memref<125xf32, strided<[1], offset: ?>> into memref<1x125xf32, strided<[125, 1], offset: ?>>
      %alloca = memref.alloca() : memref<128xf32, #gpu.address_space<workgroup>>
      %1 = gpu.thread_id  x
      %2 = arith.remsi %1, %c128 : index
      %3 = arith.cmpi slt, %2, %c0 : index
      %4 = arith.addi %2, %c128 : index
      %5 = arith.select %3, %4, %2 : index
      %6 = arith.cmpi slt, %5, %c125 : index
      %7 = arith.select %6, %5, %c125 : index
      %8 = arith.addi %5, %c1 : index
      %9 = arith.cmpi slt, %8, %c125 : index
      %10 = arith.select %9, %8, %c125 : index
      %11 = arith.subi %10, %7 : index
      %subview_3 = memref.subview %expand_shape[0, %7] [1, %11] [1, 1] : memref<1x125xf16, strided<[125, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_4 = memref.expand_shape %subview_3 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %expand_shape_2[0, %7] [1, %11] [1, 1] : memref<1x125xf32, strided<[125, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %expand_shape_6 = memref.expand_shape %subview_5 [[0, 1]] : memref<?xf32, strided<[1], offset: ?>> into memref<1x?xf32, strided<[?, 1], offset: ?>>
      %12 = arith.cmpi ugt, %11, %c0 : index
      %13:2 = scf.if %12 -> (f16, f32) {
        %24 = memref.load %expand_shape_4[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        %25 = memref.load %expand_shape_6[%c0, %c0] : memref<1x?xf32, strided<[?, 1], offset: ?>>
        scf.yield %24, %25 : f16, f32
      } else {
        scf.yield %cst_0, %cst : f16, f32
      }
      %14 = arith.extf %13#0 : f16 to f32
      %15 = arith.mulf %14, %13#1 : f32
      %16 = arith.addf %15, %cst : f32
      memref.store %16, %alloca[%1] : memref<128xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_7 = memref.alloca() : memref<64xf32, #gpu.address_space<workgroup>>
      %17 = arith.cmpi ult, %1, %c64 : index
      scf.if %17 {
        %24 = arith.muli %1, %c2 : index
        %25 = memref.load %alloca[%24] : memref<128xf32, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %cst : f32
        %27 = arith.addi %24, %c1 : index
        %28 = memref.load %alloca[%27] : memref<128xf32, #gpu.address_space<workgroup>>
        %29 = arith.addf %28, %26 : f32
        memref.store %29, %alloca_7[%1] : memref<64xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_8 = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      %18 = arith.cmpi ult, %1, %c32 : index
      scf.if %18 {
        %24 = arith.muli %1, %c2 : index
        %25 = memref.load %alloca_7[%24] : memref<64xf32, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %cst : f32
        %27 = arith.addi %24, %c1 : index
        %28 = memref.load %alloca_7[%27] : memref<64xf32, #gpu.address_space<workgroup>>
        %29 = arith.addf %28, %26 : f32
        memref.store %29, %alloca_8[%1] : memref<32xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_9 = memref.alloca() : memref<16xf32, #gpu.address_space<workgroup>>
      %19 = arith.cmpi ult, %1, %c16 : index
      scf.if %19 {
        %24 = arith.muli %1, %c2 : index
        %25 = memref.load %alloca_8[%24] : memref<32xf32, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %cst : f32
        %27 = arith.addi %24, %c1 : index
        %28 = memref.load %alloca_8[%27] : memref<32xf32, #gpu.address_space<workgroup>>
        %29 = arith.addf %28, %26 : f32
        memref.store %29, %alloca_9[%1] : memref<16xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_10 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
      %20 = arith.cmpi ult, %1, %c8 : index
      scf.if %20 {
        %24 = arith.muli %1, %c2 : index
        %25 = memref.load %alloca_9[%24] : memref<16xf32, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %cst : f32
        %27 = arith.addi %24, %c1 : index
        %28 = memref.load %alloca_9[%27] : memref<16xf32, #gpu.address_space<workgroup>>
        %29 = arith.addf %28, %26 : f32
        memref.store %29, %alloca_10[%1] : memref<8xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_11 = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
      %21 = arith.cmpi ult, %1, %c4 : index
      scf.if %21 {
        %24 = arith.muli %1, %c2 : index
        %25 = memref.load %alloca_10[%24] : memref<8xf32, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %cst : f32
        %27 = arith.addi %24, %c1 : index
        %28 = memref.load %alloca_10[%27] : memref<8xf32, #gpu.address_space<workgroup>>
        %29 = arith.addf %28, %26 : f32
        memref.store %29, %alloca_11[%1] : memref<4xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_12 = memref.alloca() : memref<2xf32, #gpu.address_space<workgroup>>
      %22 = arith.cmpi ult, %1, %c2 : index
      scf.if %22 {
        %24 = arith.muli %1, %c2 : index
        %25 = memref.load %alloca_11[%24] : memref<4xf32, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %cst : f32
        %27 = arith.addi %24, %c1 : index
        %28 = memref.load %alloca_11[%27] : memref<4xf32, #gpu.address_space<workgroup>>
        %29 = arith.addf %28, %26 : f32
        memref.store %29, %alloca_12[%1] : memref<2xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %23 = arith.cmpi ult, %1, %c1 : index
      scf.if %23 {
        %24 = arith.muli %1, %c2 : index
        %25 = memref.load %alloca_12[%24] : memref<2xf32, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %cst : f32
        %27 = arith.addi %24, %c1 : index
        %28 = memref.load %alloca_12[%27] : memref<2xf32, #gpu.address_space<workgroup>>
        %29 = arith.addf %28, %26 : f32
        memref.store %29, %arg2[%0] : memref<32xf32>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown147_kernel_0(%arg0: memref<32xf32>, %arg1: memref<f32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %0 = gpu.block_id  x
      %alloca = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      %1 = gpu.thread_id  x
      %2 = arith.muli %0, %c32 : index
      %3 = arith.addi %2, %1 : index
      %4 = memref.load %arg0[%3] : memref<32xf32>
      %5 = arith.addf %4, %cst : f32
      memref.store %5, %alloca[%1] : memref<32xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_0 = memref.alloca() : memref<16xf32, #gpu.address_space<workgroup>>
      %6 = arith.cmpi ult, %1, %c16 : index
      scf.if %6 {
        %11 = arith.muli %1, %c2 : index
        %12 = memref.load %alloca[%11] : memref<32xf32, #gpu.address_space<workgroup>>
        %13 = arith.addf %12, %cst : f32
        %14 = arith.addi %11, %c1 : index
        %15 = memref.load %alloca[%14] : memref<32xf32, #gpu.address_space<workgroup>>
        %16 = arith.addf %15, %13 : f32
        memref.store %16, %alloca_0[%1] : memref<16xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_1 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
      %7 = arith.cmpi ult, %1, %c8 : index
      scf.if %7 {
        %11 = arith.muli %1, %c2 : index
        %12 = memref.load %alloca_0[%11] : memref<16xf32, #gpu.address_space<workgroup>>
        %13 = arith.addf %12, %cst : f32
        %14 = arith.addi %11, %c1 : index
        %15 = memref.load %alloca_0[%14] : memref<16xf32, #gpu.address_space<workgroup>>
        %16 = arith.addf %15, %13 : f32
        memref.store %16, %alloca_1[%1] : memref<8xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
      %8 = arith.cmpi ult, %1, %c4 : index
      scf.if %8 {
        %11 = arith.muli %1, %c2 : index
        %12 = memref.load %alloca_1[%11] : memref<8xf32, #gpu.address_space<workgroup>>
        %13 = arith.addf %12, %cst : f32
        %14 = arith.addi %11, %c1 : index
        %15 = memref.load %alloca_1[%14] : memref<8xf32, #gpu.address_space<workgroup>>
        %16 = arith.addf %15, %13 : f32
        memref.store %16, %alloca_2[%1] : memref<4xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<2xf32, #gpu.address_space<workgroup>>
      %9 = arith.cmpi ult, %1, %c2 : index
      scf.if %9 {
        %11 = arith.muli %1, %c2 : index
        %12 = memref.load %alloca_2[%11] : memref<4xf32, #gpu.address_space<workgroup>>
        %13 = arith.addf %12, %cst : f32
        %14 = arith.addi %11, %c1 : index
        %15 = memref.load %alloca_2[%14] : memref<4xf32, #gpu.address_space<workgroup>>
        %16 = arith.addf %15, %13 : f32
        memref.store %16, %alloca_3[%1] : memref<2xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %10 = arith.cmpi ult, %1, %c1 : index
      scf.if %10 {
        %11 = arith.muli %1, %c2 : index
        %12 = memref.load %alloca_3[%11] : memref<2xf32, #gpu.address_space<workgroup>>
        %13 = arith.addf %12, %cst : f32
        %14 = arith.addi %11, %c1 : index
        %15 = memref.load %alloca_3[%14] : memref<2xf32, #gpu.address_space<workgroup>>
        %16 = arith.addf %15, %13 : f32
        memref.store %16, %arg1[] : memref<f32>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown171_kernel(%arg0: memref<4x1000xf16>, %arg1: memref<1000xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 2, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>} {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 0.000000e+00 : f16
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c1000 = arith.constant 1000 : index
      %c-32 = arith.constant -32 : index
      %0 = gpu.block_id  x
      %1 = arith.muli %0, %c-32 : index
      %2 = arith.addi %1, %c1000 : index
      %3 = arith.cmpi slt, %2, %c32 : index
      %4 = arith.select %3, %2, %c32 : index
      %5 = arith.muli %0, %c32 : index
      %alloca = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      %alloca_1 = memref.alloca() : memref<2x32xf32, #gpu.address_space<workgroup>>
      %6 = gpu.thread_id  x
      %7 = gpu.thread_id  y
      %8 = arith.cmpi slt, %4, %6 : index
      %9 = arith.select %8, %4, %6 : index
      %10 = arith.addi %6, %c1 : index
      %11 = arith.cmpi slt, %4, %10 : index
      %12 = arith.select %11, %4, %10 : index
      %13 = arith.subi %12, %9 : index
      %14 = arith.cmpi ugt, %13, %c0 : index
      %15 = scf.if %14 -> (f16) {
        %22 = arith.muli %7, %c2 : index
        %23 = arith.addi %5, %9 : index
        %24 = memref.load %arg0[%22, %23] : memref<4x1000xf16>
        scf.yield %24 : f16
      } else {
        scf.yield %cst_0 : f16
      }
      %16 = arith.extf %15 : f16 to f32
      %17 = arith.addf %16, %cst : f32
      %18 = scf.if %14 -> (f16) {
        %22 = arith.muli %7, %c2 : index
        %23 = arith.addi %22, %c1 : index
        %24 = arith.addi %5, %9 : index
        %25 = memref.load %arg0[%23, %24] : memref<4x1000xf16>
        scf.yield %25 : f16
      } else {
        scf.yield %cst_0 : f16
      }
      %19 = arith.extf %18 : f16 to f32
      %20 = arith.addf %17, %19 : f32
      memref.store %20, %alloca_1[%7, %6] : memref<2x32xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %21 = arith.cmpi ult, %7, %c1 : index
      scf.if %21 {
        %22 = memref.load %alloca_1[%c0, %6] : memref<2x32xf32, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f32
        %24 = memref.load %alloca_1[%c1, %6] : memref<2x32xf32, #gpu.address_space<workgroup>>
        %25 = arith.addf %24, %23 : f32
        memref.store %25, %alloca[%6] : memref<32xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %subview = memref.subview %alloca[0] [%4] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<?xf32, strided<[1]>, #gpu.address_space<workgroup>>
      %subview_2 = memref.subview %arg1[%5] [%4] [1] : memref<1000xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview, %subview_2 : memref<?xf32, strided<[1]>, #gpu.address_space<workgroup>> to memref<?xf32, strided<[1], offset: ?>>
      gpu.return
    }
  }
  func.func private @Unknown0(memref<4x3x224x224xf32, "cuda">) -> memref<4x3x224x224xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 588 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown1(memref<64x3x7x7xf32, "cuda">) -> memref<64x3x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 10 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown1", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown3(memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 36 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown3", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown7(memref<128x64x1x1xf32, "cuda">) -> memref<128x64x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown7", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown8(memref<128x64x3x3xf32, "cuda">) -> memref<128x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 72 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown8", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown9(memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 144 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown9", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown12(memref<256x128x1x1xf32, "cuda">) -> memref<256x128x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown12", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown13(memref<256x128x3x3xf32, "cuda">) -> memref<256x128x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown13", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown14(memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown14", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown17(memref<512x256x1x1xf32, "cuda">) -> memref<512x256x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 128 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown17", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown18(memref<512x256x3x3xf32, "cuda">) -> memref<512x256x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown18", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown19(memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown19", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown22(memref<4x1000xf32, "cuda">) -> memref<4x1000xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown22", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown23(memref<1000x512xf32, "cuda">) -> memref<1000x512xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 500 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown23", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown24(memref<1000xf32, "cuda">) -> memref<1000xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown24", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown26(memref<4x64x112x112xf16, "cuda">) -> (memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown26", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown28(memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown28", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown30(memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown30", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown37(memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown37", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown39(memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown39", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown46(memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown46", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown48(memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown48", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown55(memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown55", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown57(memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown57", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown63(memref<4x512xf16, "cuda">) -> memref<4x512xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown63", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown64(memref<1000xf16, "cuda">, memref<4x1000xf16, "cuda">) -> memref<4x1000xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown64", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown66(memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">) -> memref<4x1000xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown66", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown68(memref<4xf16, "cuda">) -> memref<4xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown68", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown69(memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">) -> (memref<4x1000xf16, "cuda">, memref<4x1000xf16, "cuda">) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown69", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown70(memref<4x512xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [2 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown70", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown74(memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown74", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown78(memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) -> memref<4x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown78", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown89(memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown89", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown93(memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown93", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown108(memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown108", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown112(memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown112", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown127(memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown127", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown131(memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown131", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown143(memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown143", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown144(memref<4x64x112x112xi1, "cuda">, memref<4x64x112x112xf16, "cuda">) -> memref<4x64x112x112xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown144", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown148(memref<f32, "cuda">) -> memref<f32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [0 : i32, 0 : i32], __byre__kernel_name = "Unknown148", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown149(memref<64x3x7x7xf16, "cuda">) -> memref<64x3x7x7xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 10 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown149", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown150(memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 36 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown150", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown154(memref<128x64x3x3xf16, "cuda">) -> memref<128x64x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 72 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown154", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown155(memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 144 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown155", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown156(memref<128x64x1x1xf16, "cuda">) -> memref<128x64x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown156", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown159(memref<256x128x3x3xf16, "cuda">) -> memref<256x128x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown159", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown160(memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown160", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown161(memref<256x128x1x1xf16, "cuda">) -> memref<256x128x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown161", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown164(memref<512x256x3x3xf16, "cuda">) -> memref<512x256x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown164", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown165(memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown165", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown166(memref<512x256x1x1xf16, "cuda">) -> memref<512x256x1x1xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 128 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown166", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown170(memref<1000x512xf16, "cuda">) -> memref<1000x512xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 500 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown170", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown172(memref<1000xf32, "cuda">) -> memref<1000xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown172", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func @main(%arg0: memref<4x3x224x224xf32, "cuda">, %arg1: memref<4x1000xf32, "cuda">, %arg2: memref<64x3x7x7xf32, "cuda">, %arg3: memref<64xf32, "cuda">, %arg4: memref<64xf32, "cuda">, %arg5: memref<64xf32, "cuda">, %arg6: memref<64xf32, "cuda">, %arg7: memref<64x64x3x3xf32, "cuda">, %arg8: memref<64xf32, "cuda">, %arg9: memref<64xf32, "cuda">, %arg10: memref<64xf32, "cuda">, %arg11: memref<64xf32, "cuda">, %arg12: memref<64x64x3x3xf32, "cuda">, %arg13: memref<64xf32, "cuda">, %arg14: memref<64xf32, "cuda">, %arg15: memref<64xf32, "cuda">, %arg16: memref<64xf32, "cuda">, %arg17: memref<64x64x3x3xf32, "cuda">, %arg18: memref<64xf32, "cuda">, %arg19: memref<64xf32, "cuda">, %arg20: memref<64xf32, "cuda">, %arg21: memref<64xf32, "cuda">, %arg22: memref<64x64x3x3xf32, "cuda">, %arg23: memref<64xf32, "cuda">, %arg24: memref<64xf32, "cuda">, %arg25: memref<64xf32, "cuda">, %arg26: memref<64xf32, "cuda">, %arg27: memref<128x64x3x3xf32, "cuda">, %arg28: memref<128xf32, "cuda">, %arg29: memref<128xf32, "cuda">, %arg30: memref<128xf32, "cuda">, %arg31: memref<128xf32, "cuda">, %arg32: memref<128x128x3x3xf32, "cuda">, %arg33: memref<128xf32, "cuda">, %arg34: memref<128xf32, "cuda">, %arg35: memref<128xf32, "cuda">, %arg36: memref<128xf32, "cuda">, %arg37: memref<128x64x1x1xf32, "cuda">, %arg38: memref<128xf32, "cuda">, %arg39: memref<128xf32, "cuda">, %arg40: memref<128xf32, "cuda">, %arg41: memref<128xf32, "cuda">, %arg42: memref<128x128x3x3xf32, "cuda">, %arg43: memref<128xf32, "cuda">, %arg44: memref<128xf32, "cuda">, %arg45: memref<128xf32, "cuda">, %arg46: memref<128xf32, "cuda">, %arg47: memref<128x128x3x3xf32, "cuda">, %arg48: memref<128xf32, "cuda">, %arg49: memref<128xf32, "cuda">, %arg50: memref<128xf32, "cuda">, %arg51: memref<128xf32, "cuda">, %arg52: memref<256x128x3x3xf32, "cuda">, %arg53: memref<256xf32, "cuda">, %arg54: memref<256xf32, "cuda">, %arg55: memref<256xf32, "cuda">, %arg56: memref<256xf32, "cuda">, %arg57: memref<256x256x3x3xf32, "cuda">, %arg58: memref<256xf32, "cuda">, %arg59: memref<256xf32, "cuda">, %arg60: memref<256xf32, "cuda">, %arg61: memref<256xf32, "cuda">, %arg62: memref<256x128x1x1xf32, "cuda">, %arg63: memref<256xf32, "cuda">, %arg64: memref<256xf32, "cuda">, %arg65: memref<256xf32, "cuda">, %arg66: memref<256xf32, "cuda">, %arg67: memref<256x256x3x3xf32, "cuda">, %arg68: memref<256xf32, "cuda">, %arg69: memref<256xf32, "cuda">, %arg70: memref<256xf32, "cuda">, %arg71: memref<256xf32, "cuda">, %arg72: memref<256x256x3x3xf32, "cuda">, %arg73: memref<256xf32, "cuda">, %arg74: memref<256xf32, "cuda">, %arg75: memref<256xf32, "cuda">, %arg76: memref<256xf32, "cuda">, %arg77: memref<512x256x3x3xf32, "cuda">, %arg78: memref<512xf32, "cuda">, %arg79: memref<512xf32, "cuda">, %arg80: memref<512xf32, "cuda">, %arg81: memref<512xf32, "cuda">, %arg82: memref<512x512x3x3xf32, "cuda">, %arg83: memref<512xf32, "cuda">, %arg84: memref<512xf32, "cuda">, %arg85: memref<512xf32, "cuda">, %arg86: memref<512xf32, "cuda">, %arg87: memref<512x256x1x1xf32, "cuda">, %arg88: memref<512xf32, "cuda">, %arg89: memref<512xf32, "cuda">, %arg90: memref<512xf32, "cuda">, %arg91: memref<512xf32, "cuda">, %arg92: memref<512x512x3x3xf32, "cuda">, %arg93: memref<512xf32, "cuda">, %arg94: memref<512xf32, "cuda">, %arg95: memref<512xf32, "cuda">, %arg96: memref<512xf32, "cuda">, %arg97: memref<512x512x3x3xf32, "cuda">, %arg98: memref<512xf32, "cuda">, %arg99: memref<512xf32, "cuda">, %arg100: memref<512xf32, "cuda">, %arg101: memref<512xf32, "cuda">, %arg102: memref<1000x512xf32, "cuda">, %arg103: memref<1000xf32, "cuda">) -> (memref<f32, "cuda">, memref<64x3x7x7xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<128x64x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x64x1x1xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<256x128x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x128x1x1xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<512x256x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x256x1x1xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1000x512xf32, "cuda">, memref<1000xf32, "cuda">) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg0) : (memref<4x3x224x224xf32, "cuda">) -> memref<4x3x224x224xf16, "cuda">
    %1 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32, "cuda">) -> memref<64x3x7x7xf16, "cuda">
    %alloc = memref.alloc() : memref<4x64x112x112xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%0, %1, %alloc) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16, "cuda">, memref<64x3x7x7xf16, "cuda">, memref<4x64x112x112xf16, "cuda">
    %alloc_0 = memref.alloc() : memref<4x64x112x112xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc, %arg3, %arg4, %alloc_0) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x112x112xf16, "cuda">
    %2 = call @Unknown3(%arg7) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %3 = call @Unknown3(%arg12) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %4 = call @Unknown3(%arg17) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %5 = call @Unknown3(%arg22) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %6 = call @Unknown7(%arg37) : (memref<128x64x1x1xf32, "cuda">) -> memref<128x64x1x1xf16, "cuda">
    %7 = call @Unknown8(%arg27) : (memref<128x64x3x3xf32, "cuda">) -> memref<128x64x3x3xf16, "cuda">
    %8 = call @Unknown9(%arg32) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %9 = call @Unknown9(%arg42) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %10 = call @Unknown9(%arg47) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %11 = call @Unknown12(%arg62) : (memref<256x128x1x1xf32, "cuda">) -> memref<256x128x1x1xf16, "cuda">
    %12 = call @Unknown13(%arg52) : (memref<256x128x3x3xf32, "cuda">) -> memref<256x128x3x3xf16, "cuda">
    %13 = call @Unknown14(%arg57) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %14 = call @Unknown14(%arg67) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %15 = call @Unknown14(%arg72) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %16 = call @Unknown17(%arg87) : (memref<512x256x1x1xf32, "cuda">) -> memref<512x256x1x1xf16, "cuda">
    %17 = call @Unknown18(%arg77) : (memref<512x256x3x3xf32, "cuda">) -> memref<512x256x3x3xf16, "cuda">
    %18 = call @Unknown19(%arg82) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %19 = call @Unknown19(%arg92) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %20 = call @Unknown19(%arg97) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %21 = call @Unknown22(%arg1) : (memref<4x1000xf32, "cuda">) -> memref<4x1000xf16, "cuda">
    %22 = call @Unknown23(%arg102) : (memref<1000x512xf32, "cuda">) -> memref<1000x512xf16, "cuda">
    %23 = call @Unknown24(%arg103) : (memref<1000xf32, "cuda">) -> memref<1000xf16, "cuda">
    %alloc_1 = memref.alloc() : memref<4xf16, "cuda">
    byre.compute @PTXOp(%21, %alloc_1) {BlockSize.x = 512 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 4 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown25_kernel"} : memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">
    %24:2 = call @Unknown26(%alloc_0) : (memref<4x64x112x112xf16, "cuda">) -> (memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xi1, "cuda">)
    %alloc_2 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @PoolMaxOp_f16_f16(%24#0, %alloc_2) {base_dilations = dense<1> : tensor<4xi64>, device = "cuda", memory_effects = [1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_3 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%alloc_2, %2, %alloc_3) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_4 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_3, %arg8, %arg9, %alloc_4) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">
    %25:2 = call @Unknown28(%alloc_4) : (memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">)
    %alloc_5 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%25#0, %3, %alloc_5) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_6 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_5, %arg13, %arg14, %alloc_6) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">
    %26:2 = call @Unknown30(%alloc_6, %alloc_2) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">)
    %alloc_7 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%26#0, %4, %alloc_7) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_8 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_7, %arg18, %arg19, %alloc_8) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">
    %27:2 = call @Unknown28(%alloc_8) : (memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">)
    %alloc_9 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%27#0, %5, %alloc_9) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_10 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_9, %arg23, %arg24, %alloc_10) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">
    %28:2 = call @Unknown30(%alloc_10, %26#0) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">)
    %alloc_11 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%28#0, %6, %alloc_11) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_12 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_11, %arg38, %arg39, %alloc_12) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_13 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%28#0, %7, %alloc_13) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_14 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_13, %arg28, %arg29, %alloc_14) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %29:2 = call @Unknown37(%alloc_14) : (memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">)
    %alloc_15 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%29#0, %8, %alloc_15) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_16 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_15, %arg33, %arg34, %alloc_16) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %30:2 = call @Unknown39(%alloc_16, %alloc_12) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">)
    %alloc_17 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%30#0, %9, %alloc_17) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_18 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_17, %arg43, %arg44, %alloc_18) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %31:2 = call @Unknown37(%alloc_18) : (memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">)
    %alloc_19 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%31#0, %10, %alloc_19) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_20 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_19, %arg48, %arg49, %alloc_20) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %32:2 = call @Unknown39(%alloc_20, %30#0) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">) -> (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">)
    %alloc_21 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%32#0, %11, %alloc_21) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_22 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_21, %arg63, %arg64, %alloc_22) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_23 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%32#0, %12, %alloc_23) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_24 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_23, %arg53, %arg54, %alloc_24) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %33:2 = call @Unknown46(%alloc_24) : (memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">)
    %alloc_25 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%33#0, %13, %alloc_25) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_26 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_25, %arg58, %arg59, %alloc_26) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %34:2 = call @Unknown48(%alloc_26, %alloc_22) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">)
    %alloc_27 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%34#0, %14, %alloc_27) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_28 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_27, %arg68, %arg69, %alloc_28) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %35:2 = call @Unknown46(%alloc_28) : (memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">)
    %alloc_29 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%35#0, %15, %alloc_29) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_30 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_29, %arg73, %arg74, %alloc_30) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %36:2 = call @Unknown48(%alloc_30, %34#0) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">) -> (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">)
    %alloc_31 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%36#0, %16, %alloc_31) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_32 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_31, %arg88, %arg89, %alloc_32) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_33 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%36#0, %17, %alloc_33) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_34 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_33, %arg78, %arg79, %alloc_34) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %37:2 = call @Unknown55(%alloc_34) : (memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">)
    %alloc_35 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%37#0, %18, %alloc_35) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_36 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_35, %arg83, %arg84, %alloc_36) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %38:2 = call @Unknown57(%alloc_36, %alloc_32) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">)
    %alloc_37 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%38#0, %19, %alloc_37) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_38 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_37, %arg93, %arg94, %alloc_38) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %39:2 = call @Unknown55(%alloc_38) : (memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">)
    %alloc_39 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%39#0, %20, %alloc_39) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_40 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%alloc_39, %arg98, %arg99, %alloc_40) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %40:2 = call @Unknown57(%alloc_40, %38#0) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">) -> (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">)
    %collapse_shape = memref.collapse_shape %40#0 [[0, 1], [2, 3]] : memref<4x512x7x7xf16, "cuda"> into memref<2048x49xf16, "cuda">
    %alloc_41 = memref.alloc() : memref<2048xf16, "cuda">
    byre.compute @PTXOp(%collapse_shape, %alloc_41) {BlockSize.x = 64 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 2048 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown62_kernel"} : memref<2048x49xf16, "cuda">, memref<2048xf16, "cuda">
    %expand_shape = memref.expand_shape %alloc_41 [[0, 1]] : memref<2048xf16, "cuda"> into memref<4x512xf16, "cuda">
    %41 = call @Unknown63(%expand_shape) : (memref<4x512xf16, "cuda">) -> memref<4x512xf16, "cuda">
    %alloc_42 = memref.alloc() : memref<4x1000xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%41, %22, %alloc_42) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<4x512xf16, "cuda">, memref<1000x512xf16, "cuda">, memref<4x1000xf16, "cuda">
    %42 = call @Unknown64(%23, %alloc_42) : (memref<1000xf16, "cuda">, memref<4x1000xf16, "cuda">) -> memref<4x1000xf16, "cuda">
    %alloc_43 = memref.alloc() : memref<4xf16, "cuda">
    byre.compute @PTXOp(%42, %alloc_43) {BlockSize.x = 512 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 4 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown65_kernel"} : memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">
    %43 = call @Unknown66(%alloc_43, %42) : (memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">) -> memref<4x1000xf16, "cuda">
    %alloc_44 = memref.alloc() : memref<4xf16, "cuda">
    byre.compute @PTXOp(%43, %alloc_44) {BlockSize.x = 512 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 4 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown67_kernel"} : memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">
    %44 = call @Unknown68(%alloc_44) : (memref<4xf16, "cuda">) -> memref<4xf16, "cuda">
    %45:2 = call @Unknown69(%44, %43, %alloc_1, %21) : (memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">) -> (memref<4x1000xf16, "cuda">, memref<4x1000xf16, "cuda">)
    %alloc_45 = memref.alloc() : memref<4x512xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%45#1, %22, %alloc_45) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<4x1000xf16, "cuda">, memref<1000x512xf16, "cuda">, memref<4x512xf16, "cuda">
    %46 = call @Unknown70(%alloc_45, %40#1) : (memref<4x512xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %alloc_46 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    %alloc_47 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_48 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_39, %arg98, %46, %alloc_46, %alloc_47, %alloc_48) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_49 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_46, %20, %alloc_49) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_50 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%39#0, %alloc_46, %alloc_50) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    %47 = call @Unknown74(%39#1, %alloc_49) : (memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %alloc_51 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    %alloc_52 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_53 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_37, %arg93, %47, %alloc_51, %alloc_52, %alloc_53) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_54 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_51, %19, %alloc_54) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_55 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%38#0, %alloc_51, %alloc_55) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    %48 = call @Unknown78(%46, %alloc_54, %38#1) : (memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %alloc_56 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    %alloc_57 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_58 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_35, %arg83, %48, %alloc_56, %alloc_57, %alloc_58) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_59 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_56, %18, %alloc_59) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %alloc_60 = memref.alloc() : memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%37#0, %alloc_56, %alloc_60) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    %49 = call @Unknown74(%37#1, %alloc_59) : (memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %alloc_61 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    %alloc_62 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_63 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_33, %arg78, %49, %alloc_61, %alloc_62, %alloc_63) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_64 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_61, %17, %alloc_64) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_65 = memref.alloc() : memref<512x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%36#0, %alloc_61, %alloc_65) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">
    %alloc_66 = memref.alloc() : memref<4x512x7x7xf16, "cuda">
    %alloc_67 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_68 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_31, %arg88, %48, %alloc_66, %alloc_67, %alloc_68) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %alloc_69 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_66, %16, %alloc_69) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_70 = memref.alloc() : memref<512x256x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%36#0, %alloc_66, %alloc_70) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">
    %50 = call @Unknown89(%alloc_69, %alloc_64, %36#1) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %alloc_71 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    %alloc_72 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_73 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_29, %arg73, %50, %alloc_71, %alloc_72, %alloc_73) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_74 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_71, %15, %alloc_74) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_75 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%35#0, %alloc_71, %alloc_75) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    %51 = call @Unknown93(%35#1, %alloc_74) : (memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %alloc_76 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    %alloc_77 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_78 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_27, %arg68, %51, %alloc_76, %alloc_77, %alloc_78) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_79 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_76, %14, %alloc_79) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_80 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%34#0, %alloc_76, %alloc_80) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    %52 = call @Unknown89(%50, %alloc_79, %34#1) : (memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %alloc_81 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    %alloc_82 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_83 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_25, %arg58, %52, %alloc_81, %alloc_82, %alloc_83) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_84 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_81, %13, %alloc_84) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %alloc_85 = memref.alloc() : memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%33#0, %alloc_81, %alloc_85) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    %53 = call @Unknown93(%33#1, %alloc_84) : (memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %alloc_86 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    %alloc_87 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_88 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_23, %arg53, %53, %alloc_86, %alloc_87, %alloc_88) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_89 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_86, %12, %alloc_89) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_90 = memref.alloc() : memref<256x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%32#0, %alloc_86, %alloc_90) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">
    %alloc_91 = memref.alloc() : memref<4x256x14x14xf16, "cuda">
    %alloc_92 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_93 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_21, %arg63, %52, %alloc_91, %alloc_92, %alloc_93) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %alloc_94 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_91, %11, %alloc_94) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_95 = memref.alloc() : memref<256x128x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%32#0, %alloc_91, %alloc_95) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">
    %54 = call @Unknown108(%alloc_94, %alloc_89, %32#1) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %alloc_96 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    %alloc_97 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_98 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_19, %arg48, %54, %alloc_96, %alloc_97, %alloc_98) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_99 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_96, %10, %alloc_99) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_100 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%31#0, %alloc_96, %alloc_100) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    %55 = call @Unknown112(%31#1, %alloc_99) : (memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %alloc_101 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    %alloc_102 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_103 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_17, %arg43, %55, %alloc_101, %alloc_102, %alloc_103) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_104 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_101, %9, %alloc_104) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_105 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%30#0, %alloc_101, %alloc_105) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    %56 = call @Unknown108(%54, %alloc_104, %30#1) : (memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %alloc_106 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    %alloc_107 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_108 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_15, %arg33, %56, %alloc_106, %alloc_107, %alloc_108) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_109 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_106, %8, %alloc_109) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %alloc_110 = memref.alloc() : memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%29#0, %alloc_106, %alloc_110) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    %57 = call @Unknown112(%29#1, %alloc_109) : (memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %alloc_111 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    %alloc_112 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_113 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_13, %arg28, %57, %alloc_111, %alloc_112, %alloc_113) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_114 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_111, %7, %alloc_114) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_115 = memref.alloc() : memref<128x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%28#0, %alloc_111, %alloc_115) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">
    %alloc_116 = memref.alloc() : memref<4x128x28x28xf16, "cuda">
    %alloc_117 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_118 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_11, %arg38, %56, %alloc_116, %alloc_117, %alloc_118) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %alloc_119 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_116, %6, %alloc_119) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_120 = memref.alloc() : memref<128x64x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%28#0, %alloc_116, %alloc_120) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">
    %58 = call @Unknown127(%alloc_119, %alloc_114, %28#1) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %alloc_121 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    %alloc_122 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_123 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_9, %arg23, %58, %alloc_121, %alloc_122, %alloc_123) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_124 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_121, %5, %alloc_124) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_125 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%27#0, %alloc_121, %alloc_125) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    %59 = call @Unknown131(%27#1, %alloc_124) : (memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %alloc_126 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    %alloc_127 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_128 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_7, %arg18, %59, %alloc_126, %alloc_127, %alloc_128) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_129 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_126, %4, %alloc_129) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_130 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%26#0, %alloc_126, %alloc_130) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    %60 = call @Unknown127(%58, %alloc_129, %26#1) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %alloc_131 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    %alloc_132 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_133 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_5, %arg13, %60, %alloc_131, %alloc_132, %alloc_133) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_134 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_131, %3, %alloc_134) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_135 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%25#0, %alloc_131, %alloc_135) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    %61 = call @Unknown131(%25#1, %alloc_134) : (memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %alloc_136 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    %alloc_137 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_138 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc_3, %arg8, %61, %alloc_136, %alloc_137, %alloc_138) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_139 = memref.alloc() : memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%alloc_136, %2, %alloc_139) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %alloc_140 = memref.alloc() : memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%alloc_2, %alloc_136, %alloc_140) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    %62 = call @Unknown143(%60, %alloc_139) : (memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %alloc_141 = memref.alloc() : memref<4x64x112x112xf16, "cuda">
    byre.compute @PoolMaxGradOp_f16f16_f16(%24#0, %62, %alloc_141) {device = "cuda", memory_effects = [1 : i32, 1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x112x112xf16, "cuda">
    %63 = call @Unknown144(%24#1, %alloc_141) : (memref<4x64x112x112xi1, "cuda">, memref<4x64x112x112xf16, "cuda">) -> memref<4x64x112x112xf16, "cuda">
    %alloc_142 = memref.alloc() : memref<4x64x112x112xf16, "cuda">
    %alloc_143 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_144 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%alloc, %arg3, %63, %alloc_142, %alloc_143, %alloc_144) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %alloc_145 = memref.alloc() : memref<64x3x7x7xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%0, %alloc_142, %alloc_145) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<64x3x7x7xf16, "cuda">
    %alloc_146 = memref.alloc() : memref<f32, "cuda">
    %collapse_shape_147 = memref.collapse_shape %45#0 [[0, 1]] : memref<4x1000xf16, "cuda"> into memref<4000xf16, "cuda">
    %collapse_shape_148 = memref.collapse_shape %arg1 [[0, 1]] : memref<4x1000xf32, "cuda"> into memref<4000xf32, "cuda">
    %expand_shape_149 = memref.expand_shape %collapse_shape_147 [[0, 1]] : memref<4000xf16, "cuda"> into memref<32x125xf16, "cuda">
    %expand_shape_150 = memref.expand_shape %collapse_shape_148 [[0, 1]] : memref<4000xf32, "cuda"> into memref<32x125xf32, "cuda">
    %alloc_151 = memref.alloc() : memref<32xf32, "cuda">
    byre.compute @PTXOp(%expand_shape_149, %expand_shape_150, %alloc_151) {BlockSize.x = 128 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 32 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown147_kernel"} : memref<32x125xf16, "cuda">, memref<32x125xf32, "cuda">, memref<32xf32, "cuda">
    byre.compute @PTXOp(%alloc_151, %alloc_146) {BlockSize.x = 32 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 1 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown147_kernel_0"} : memref<32xf32, "cuda">, memref<f32, "cuda">
    %64 = call @Unknown148(%alloc_146) : (memref<f32, "cuda">) -> memref<f32, "cuda">
    %65 = call @Unknown149(%alloc_145) : (memref<64x3x7x7xf16, "cuda">) -> memref<64x3x7x7xf32, "cuda">
    %66 = call @Unknown150(%alloc_140) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %67 = call @Unknown150(%alloc_135) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %68 = call @Unknown150(%alloc_130) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %69 = call @Unknown150(%alloc_125) : (memref<64x64x3x3xf16, "cuda">) -> memref<64x64x3x3xf32, "cuda">
    %70 = call @Unknown154(%alloc_115) : (memref<128x64x3x3xf16, "cuda">) -> memref<128x64x3x3xf32, "cuda">
    %71 = call @Unknown155(%alloc_110) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %72 = call @Unknown156(%alloc_120) : (memref<128x64x1x1xf16, "cuda">) -> memref<128x64x1x1xf32, "cuda">
    %73 = call @Unknown155(%alloc_105) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %74 = call @Unknown155(%alloc_100) : (memref<128x128x3x3xf16, "cuda">) -> memref<128x128x3x3xf32, "cuda">
    %75 = call @Unknown159(%alloc_90) : (memref<256x128x3x3xf16, "cuda">) -> memref<256x128x3x3xf32, "cuda">
    %76 = call @Unknown160(%alloc_85) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %77 = call @Unknown161(%alloc_95) : (memref<256x128x1x1xf16, "cuda">) -> memref<256x128x1x1xf32, "cuda">
    %78 = call @Unknown160(%alloc_80) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %79 = call @Unknown160(%alloc_75) : (memref<256x256x3x3xf16, "cuda">) -> memref<256x256x3x3xf32, "cuda">
    %80 = call @Unknown164(%alloc_65) : (memref<512x256x3x3xf16, "cuda">) -> memref<512x256x3x3xf32, "cuda">
    %81 = call @Unknown165(%alloc_60) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %82 = call @Unknown166(%alloc_70) : (memref<512x256x1x1xf16, "cuda">) -> memref<512x256x1x1xf32, "cuda">
    %83 = call @Unknown165(%alloc_55) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %84 = call @Unknown165(%alloc_50) : (memref<512x512x3x3xf16, "cuda">) -> memref<512x512x3x3xf32, "cuda">
    %alloc_152 = memref.alloc() : memref<1000x512xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%41, %45#1, %alloc_152) {device = "cuda", lhs_contracting_dimension = 0 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_transpose, rhs_contracting_dimension = 0 : i64} : memref<4x512xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<1000x512xf16, "cuda">
    %85 = call @Unknown170(%alloc_152) : (memref<1000x512xf16, "cuda">) -> memref<1000x512xf32, "cuda">
    %alloc_153 = memref.alloc() : memref<1000xf32, "cuda">
    byre.compute @PTXOp(%45#1, %alloc_153) {BlockSize.x = 32 : i32, BlockSize.y = 2 : i32, BlockSize.z = 1 : i32, GridSize.x = 32 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown171_kernel"} : memref<4x1000xf16, "cuda">, memref<1000xf32, "cuda">
    %86 = call @Unknown172(%alloc_153) : (memref<1000xf32, "cuda">) -> memref<1000xf32, "cuda">
    return %64, %65, %alloc_143, %alloc_144, %66, %alloc_137, %alloc_138, %67, %alloc_132, %alloc_133, %68, %alloc_127, %alloc_128, %69, %alloc_122, %alloc_123, %70, %alloc_112, %alloc_113, %71, %alloc_107, %alloc_108, %72, %alloc_117, %alloc_118, %73, %alloc_102, %alloc_103, %74, %alloc_97, %alloc_98, %75, %alloc_87, %alloc_88, %76, %alloc_82, %alloc_83, %77, %alloc_92, %alloc_93, %78, %alloc_77, %alloc_78, %79, %alloc_72, %alloc_73, %80, %alloc_62, %alloc_63, %81, %alloc_57, %alloc_58, %82, %alloc_67, %alloc_68, %83, %alloc_52, %alloc_53, %84, %alloc_47, %alloc_48, %85, %86 : memref<f32, "cuda">, memref<64x3x7x7xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64x64x3x3xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<128x64x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x64x1x1xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128x128x3x3xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<256x128x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x128x1x1xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256x256x3x3xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<512x256x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x256x1x1xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512x512x3x3xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1000x512xf32, "cuda">, memref<1000xf32, "cuda">
  }
}