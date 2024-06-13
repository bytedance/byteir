// RUN: byteir-opt %s -byre-host="device-file-name=your_file target=cuda entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

module @IrToMhlo.2452 attributes {byre.container_module, gpu.container_module} {
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
  func.func @main(%arg0: memref<4x3x224x224xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<4x1000xf32, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<64x3x7x7xf32, "cuda"> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<64xf32, "cuda"> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<64xf32, "cuda"> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<64xf32, "cuda"> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<64xf32, "cuda"> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<64xf32, "cuda"> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<64xf32, "cuda"> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<64xf32, "cuda"> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<64xf32, "cuda"> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<64xf32, "cuda"> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<64xf32, "cuda"> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<64xf32, "cuda"> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<64xf32, "cuda"> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<64xf32, "cuda"> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<64xf32, "cuda"> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<64xf32, "cuda"> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<64xf32, "cuda"> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<64xf32, "cuda"> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<64xf32, "cuda"> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<64xf32, "cuda"> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<64xf32, "cuda"> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x64x3x3xf32, "cuda"> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32, "cuda"> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128xf32, "cuda"> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32, "cuda"> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128xf32, "cuda"> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32, "cuda"> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32, "cuda"> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<128xf32, "cuda"> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<128xf32, "cuda"> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x64x1x1xf32, "cuda"> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32, "cuda"> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32, "cuda"> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32, "cuda"> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128xf32, "cuda"> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32, "cuda"> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32, "cuda"> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<128xf32, "cuda"> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<128xf32, "cuda"> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<128xf32, "cuda"> {byre.argname = "Input48", byre.argtype = 1 : i32}, %arg49: memref<128xf32, "cuda"> {byre.argname = "Input49", byre.argtype = 1 : i32}, %arg50: memref<128xf32, "cuda"> {byre.argname = "Input50", byre.argtype = 1 : i32}, %arg51: memref<128xf32, "cuda"> {byre.argname = "Input51", byre.argtype = 1 : i32}, %arg52: memref<256x128x3x3xf32, "cuda"> {byre.argname = "Input52", byre.argtype = 1 : i32}, %arg53: memref<256xf32, "cuda"> {byre.argname = "Input53", byre.argtype = 1 : i32}, %arg54: memref<256xf32, "cuda"> {byre.argname = "Input54", byre.argtype = 1 : i32}, %arg55: memref<256xf32, "cuda"> {byre.argname = "Input55", byre.argtype = 1 : i32}, %arg56: memref<256xf32, "cuda"> {byre.argname = "Input56", byre.argtype = 1 : i32}, %arg57: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Input57", byre.argtype = 1 : i32}, %arg58: memref<256xf32, "cuda"> {byre.argname = "Input58", byre.argtype = 1 : i32}, %arg59: memref<256xf32, "cuda"> {byre.argname = "Input59", byre.argtype = 1 : i32}, %arg60: memref<256xf32, "cuda"> {byre.argname = "Input60", byre.argtype = 1 : i32}, %arg61: memref<256xf32, "cuda"> {byre.argname = "Input61", byre.argtype = 1 : i32}, %arg62: memref<256x128x1x1xf32, "cuda"> {byre.argname = "Input62", byre.argtype = 1 : i32}, %arg63: memref<256xf32, "cuda"> {byre.argname = "Input63", byre.argtype = 1 : i32}, %arg64: memref<256xf32, "cuda"> {byre.argname = "Input64", byre.argtype = 1 : i32}, %arg65: memref<256xf32, "cuda"> {byre.argname = "Input65", byre.argtype = 1 : i32}, %arg66: memref<256xf32, "cuda"> {byre.argname = "Input66", byre.argtype = 1 : i32}, %arg67: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Input67", byre.argtype = 1 : i32}, %arg68: memref<256xf32, "cuda"> {byre.argname = "Input68", byre.argtype = 1 : i32}, %arg69: memref<256xf32, "cuda"> {byre.argname = "Input69", byre.argtype = 1 : i32}, %arg70: memref<256xf32, "cuda"> {byre.argname = "Input70", byre.argtype = 1 : i32}, %arg71: memref<256xf32, "cuda"> {byre.argname = "Input71", byre.argtype = 1 : i32}, %arg72: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Input72", byre.argtype = 1 : i32}, %arg73: memref<256xf32, "cuda"> {byre.argname = "Input73", byre.argtype = 1 : i32}, %arg74: memref<256xf32, "cuda"> {byre.argname = "Input74", byre.argtype = 1 : i32}, %arg75: memref<256xf32, "cuda"> {byre.argname = "Input75", byre.argtype = 1 : i32}, %arg76: memref<256xf32, "cuda"> {byre.argname = "Input76", byre.argtype = 1 : i32}, %arg77: memref<512x256x3x3xf32, "cuda"> {byre.argname = "Input77", byre.argtype = 1 : i32}, %arg78: memref<512xf32, "cuda"> {byre.argname = "Input78", byre.argtype = 1 : i32}, %arg79: memref<512xf32, "cuda"> {byre.argname = "Input79", byre.argtype = 1 : i32}, %arg80: memref<512xf32, "cuda"> {byre.argname = "Input80", byre.argtype = 1 : i32}, %arg81: memref<512xf32, "cuda"> {byre.argname = "Input81", byre.argtype = 1 : i32}, %arg82: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Input82", byre.argtype = 1 : i32}, %arg83: memref<512xf32, "cuda"> {byre.argname = "Input83", byre.argtype = 1 : i32}, %arg84: memref<512xf32, "cuda"> {byre.argname = "Input84", byre.argtype = 1 : i32}, %arg85: memref<512xf32, "cuda"> {byre.argname = "Input85", byre.argtype = 1 : i32}, %arg86: memref<512xf32, "cuda"> {byre.argname = "Input86", byre.argtype = 1 : i32}, %arg87: memref<512x256x1x1xf32, "cuda"> {byre.argname = "Input87", byre.argtype = 1 : i32}, %arg88: memref<512xf32, "cuda"> {byre.argname = "Input88", byre.argtype = 1 : i32}, %arg89: memref<512xf32, "cuda"> {byre.argname = "Input89", byre.argtype = 1 : i32}, %arg90: memref<512xf32, "cuda"> {byre.argname = "Input90", byre.argtype = 1 : i32}, %arg91: memref<512xf32, "cuda"> {byre.argname = "Input91", byre.argtype = 1 : i32}, %arg92: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Input92", byre.argtype = 1 : i32}, %arg93: memref<512xf32, "cuda"> {byre.argname = "Input93", byre.argtype = 1 : i32}, %arg94: memref<512xf32, "cuda"> {byre.argname = "Input94", byre.argtype = 1 : i32}, %arg95: memref<512xf32, "cuda"> {byre.argname = "Input95", byre.argtype = 1 : i32}, %arg96: memref<512xf32, "cuda"> {byre.argname = "Input96", byre.argtype = 1 : i32}, %arg97: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Input97", byre.argtype = 1 : i32}, %arg98: memref<512xf32, "cuda"> {byre.argname = "Input98", byre.argtype = 1 : i32}, %arg99: memref<512xf32, "cuda"> {byre.argname = "Input99", byre.argtype = 1 : i32}, %arg100: memref<512xf32, "cuda"> {byre.argname = "Input100", byre.argtype = 1 : i32}, %arg101: memref<512xf32, "cuda"> {byre.argname = "Input101", byre.argtype = 1 : i32}, %arg102: memref<1000x512xf32, "cuda"> {byre.argname = "Input102", byre.argtype = 1 : i32}, %arg103: memref<1000xf32, "cuda"> {byre.argname = "Input103", byre.argtype = 1 : i32}, %arg104: memref<f32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg105: memref<64x3x7x7xf32, "cuda"> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg106: memref<64xf32, "cuda"> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg107: memref<64xf32, "cuda"> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg108: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg109: memref<64xf32, "cuda"> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg110: memref<64xf32, "cuda"> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg111: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg112: memref<64xf32, "cuda"> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg113: memref<64xf32, "cuda"> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg114: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg115: memref<64xf32, "cuda"> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg116: memref<64xf32, "cuda"> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg117: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg118: memref<64xf32, "cuda"> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg119: memref<64xf32, "cuda"> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg120: memref<128x64x3x3xf32, "cuda"> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg121: memref<128xf32, "cuda"> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg122: memref<128xf32, "cuda"> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg123: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg124: memref<128xf32, "cuda"> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg125: memref<128xf32, "cuda"> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg126: memref<128x64x1x1xf32, "cuda"> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg127: memref<128xf32, "cuda"> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg128: memref<128xf32, "cuda"> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg129: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg130: memref<128xf32, "cuda"> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg131: memref<128xf32, "cuda"> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg132: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg133: memref<128xf32, "cuda"> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg134: memref<128xf32, "cuda"> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg135: memref<256x128x3x3xf32, "cuda"> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg136: memref<256xf32, "cuda"> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg137: memref<256xf32, "cuda"> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg138: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg139: memref<256xf32, "cuda"> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg140: memref<256xf32, "cuda"> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg141: memref<256x128x1x1xf32, "cuda"> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg142: memref<256xf32, "cuda"> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg143: memref<256xf32, "cuda"> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg144: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg145: memref<256xf32, "cuda"> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg146: memref<256xf32, "cuda"> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg147: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Output43", byre.argtype = 2 : i32}, %arg148: memref<256xf32, "cuda"> {byre.argname = "Output44", byre.argtype = 2 : i32}, %arg149: memref<256xf32, "cuda"> {byre.argname = "Output45", byre.argtype = 2 : i32}, %arg150: memref<512x256x3x3xf32, "cuda"> {byre.argname = "Output46", byre.argtype = 2 : i32}, %arg151: memref<512xf32, "cuda"> {byre.argname = "Output47", byre.argtype = 2 : i32}, %arg152: memref<512xf32, "cuda"> {byre.argname = "Output48", byre.argtype = 2 : i32}, %arg153: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Output49", byre.argtype = 2 : i32}, %arg154: memref<512xf32, "cuda"> {byre.argname = "Output50", byre.argtype = 2 : i32}, %arg155: memref<512xf32, "cuda"> {byre.argname = "Output51", byre.argtype = 2 : i32}, %arg156: memref<512x256x1x1xf32, "cuda"> {byre.argname = "Output52", byre.argtype = 2 : i32}, %arg157: memref<512xf32, "cuda"> {byre.argname = "Output53", byre.argtype = 2 : i32}, %arg158: memref<512xf32, "cuda"> {byre.argname = "Output54", byre.argtype = 2 : i32}, %arg159: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Output55", byre.argtype = 2 : i32}, %arg160: memref<512xf32, "cuda"> {byre.argname = "Output56", byre.argtype = 2 : i32}, %arg161: memref<512xf32, "cuda"> {byre.argname = "Output57", byre.argtype = 2 : i32}, %arg162: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Output58", byre.argtype = 2 : i32}, %arg163: memref<512xf32, "cuda"> {byre.argname = "Output59", byre.argtype = 2 : i32}, %arg164: memref<512xf32, "cuda"> {byre.argname = "Output60", byre.argtype = 2 : i32}, %arg165: memref<1000x512xf32, "cuda"> {byre.argname = "Output61", byre.argtype = 2 : i32}, %arg166: memref<1000xf32, "cuda"> {byre.argname = "Output62", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %alloc = memref.alloc() : memref<76533504xi8, "cuda">
    %0 = "byre.alias"(%alloc) <{offset = 75329280 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x3x224x224xf16, "cuda">
    byre.compute @PTXOp(%arg0, %0) {BlockSize.x = 256 : i32, GridSize.x = 588 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown0", memory_effects = [1 : i32, 2 : i32]} : memref<4x3x224x224xf32, "cuda">, memref<4x3x224x224xf16, "cuda">
    %1 = "byre.alias"(%alloc) <{offset = 62959360 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x3x7x7xf16, "cuda">
    byre.compute @PTXOp(%arg2, %1) {BlockSize.x = 256 : i32, GridSize.x = 10 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown1", memory_effects = [1 : i32, 2 : i32]} : memref<64x3x7x7xf32, "cuda">, memref<64x3x7x7xf16, "cuda">
    %2 = "byre.alias"(%alloc) <{offset = 49311488 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x112x112xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%0, %1, %2) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16, "cuda">, memref<64x3x7x7xf16, "cuda">, memref<4x64x112x112xf16, "cuda">
    %3 = "byre.alias"(%alloc) <{offset = 42888960 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x112x112xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%2, %arg3, %arg4, %3) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x112x112xf16, "cuda">
    %4 = "byre.alias"(%alloc) <{offset = 5545728 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg7, %4) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown3", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf16, "cuda">
    %5 = "byre.alias"(%alloc) <{offset = 5361664 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg12, %5) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown3", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf16, "cuda">
    %6 = "byre.alias"(%alloc) <{offset = 6283008 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg17, %6) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown3", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf16, "cuda">
    %7 = "byre.alias"(%alloc) <{offset = 6209280 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg22, %7) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown3", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf16, "cuda">
    %8 = "byre.alias"(%alloc) <{offset = 5463808 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x64x1x1xf16, "cuda">
    byre.compute @PTXOp(%arg37, %8) {BlockSize.x = 256 : i32, GridSize.x = 8 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown7", memory_effects = [1 : i32, 2 : i32]} : memref<128x64x1x1xf32, "cuda">, memref<128x64x1x1xf16, "cuda">
    %9 = "byre.alias"(%alloc) <{offset = 6557440 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg27, %9) {BlockSize.x = 256 : i32, GridSize.x = 72 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown8", memory_effects = [1 : i32, 2 : i32]} : memref<128x64x3x3xf32, "cuda">, memref<128x64x3x3xf16, "cuda">
    %10 = "byre.alias"(%alloc) <{offset = 2256896 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg32, %10) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown9", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf16, "cuda">
    %11 = "byre.alias"(%alloc) <{offset = 1761280 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg42, %11) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown9", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf16, "cuda">
    %12 = "byre.alias"(%alloc) <{offset = 0 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg47, %12) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown9", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf16, "cuda">
    %13 = "byre.alias"(%alloc) <{offset = 5480192 : i64}> : (memref<76533504xi8, "cuda">) -> memref<256x128x1x1xf16, "cuda">
    byre.compute @PTXOp(%arg62, %13) {BlockSize.x = 256 : i32, GridSize.x = 32 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown12", memory_effects = [1 : i32, 2 : i32]} : memref<256x128x1x1xf32, "cuda">, memref<256x128x1x1xf16, "cuda">
    %14 = "byre.alias"(%alloc) <{offset = 5619456 : i64}> : (memref<76533504xi8, "cuda">) -> memref<256x128x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg52, %14) {BlockSize.x = 256 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown13", memory_effects = [1 : i32, 2 : i32]} : memref<256x128x3x3xf32, "cuda">, memref<256x128x3x3xf16, "cuda">
    %15 = "byre.alias"(%alloc) <{offset = 74149632 : i64}> : (memref<76533504xi8, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg57, %15) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown14", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf16, "cuda">
    %16 = "byre.alias"(%alloc) <{offset = 21556992 : i64}> : (memref<76533504xi8, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg67, %16) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown14", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf16, "cuda">
    %17 = "byre.alias"(%alloc) <{offset = 8711936 : i64}> : (memref<76533504xi8, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg72, %17) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown14", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf16, "cuda">
    %18 = "byre.alias"(%alloc) <{offset = 4558848 : i64}> : (memref<76533504xi8, "cuda">) -> memref<512x256x1x1xf16, "cuda">
    byre.compute @PTXOp(%arg87, %18) {BlockSize.x = 256 : i32, GridSize.x = 128 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown17", memory_effects = [1 : i32, 2 : i32]} : memref<512x256x1x1xf32, "cuda">, memref<512x256x1x1xf16, "cuda">
    %19 = "byre.alias"(%alloc) <{offset = 23162624 : i64}> : (memref<76533504xi8, "cuda">) -> memref<512x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg77, %19) {BlockSize.x = 256 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown18", memory_effects = [1 : i32, 2 : i32]} : memref<512x256x3x3xf32, "cuda">, memref<512x256x3x3xf16, "cuda">
    %20 = "byre.alias"(%alloc) <{offset = 28733184 : i64}> : (memref<76533504xi8, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg82, %20) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown19", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf16, "cuda">
    %21 = "byre.alias"(%alloc) <{offset = 33451776 : i64}> : (memref<76533504xi8, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg92, %21) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown19", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf16, "cuda">
    %22 = "byre.alias"(%alloc) <{offset = 38170368 : i64}> : (memref<76533504xi8, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg97, %22) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown19", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf16, "cuda">
    %23 = "byre.alias"(%alloc) <{offset = 5439616 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x1000xf16, "cuda">
    byre.compute @PTXOp(%arg1, %23) {BlockSize.x = 256 : i32, GridSize.x = 4 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown22", memory_effects = [1 : i32, 2 : i32]} : memref<4x1000xf32, "cuda">, memref<4x1000xf16, "cuda">
    %24 = "byre.alias"(%alloc) <{offset = 72969984 : i64}> : (memref<76533504xi8, "cuda">) -> memref<1000x512xf16, "cuda">
    byre.compute @PTXOp(%arg102, %24) {BlockSize.x = 256 : i32, GridSize.x = 500 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown23", memory_effects = [1 : i32, 2 : i32]} : memref<1000x512xf32, "cuda">, memref<1000x512xf16, "cuda">
    %25 = "byre.alias"(%alloc) <{offset = 73993984 : i64}> : (memref<76533504xi8, "cuda">) -> memref<1000xf16, "cuda">
    byre.compute @PTXOp(%arg103, %25) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown24", memory_effects = [1 : i32, 2 : i32]} : memref<1000xf32, "cuda">, memref<1000xf16, "cuda">
    %26 = "byre.alias"(%alloc) <{offset = 5435392 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4xf16, "cuda">
    byre.compute @PTXOp(%23, %26) {BlockSize.x = 512 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 4 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown25_kernel"} : memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">
    %27 = "byre.alias"(%alloc) <{offset = 62959360 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x112x112xf16, "cuda">
    %28 = "byre.alias"(%alloc) <{offset = 25521920 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x112x112xi1, "cuda">
    byre.compute @PTXOp(%3, %27, %28) {BlockSize.x = 256 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown26", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xi1, "cuda">
    %29 = "byre.alias"(%alloc) <{offset = 15134464 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @PoolMaxOp_f16_f16(%27, %29) {base_dilations = dense<1> : tensor<4xi64>, device = "cuda", memory_effects = [1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %30 = "byre.alias"(%alloc) <{offset = 16740096 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%29, %4, %30) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %31 = "byre.alias"(%alloc) <{offset = 42888960 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%30, %arg8, %arg9, %31) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">
    %32 = "byre.alias"(%alloc) <{offset = 19951360 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %33 = "byre.alias"(%alloc) <{offset = 69381888 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xi1, "cuda">
    byre.compute @PTXOp(%31, %32, %33) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown28", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">
    %34 = "byre.alias"(%alloc) <{offset = 7106304 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%32, %5, %34) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%34, %arg13, %arg14, %31) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">
    %35 = "byre.alias"(%alloc) <{offset = 18345728 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %36 = "byre.alias"(%alloc) <{offset = 70987520 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xi1, "cuda">
    byre.compute @PTXOp(%31, %29, %35, %36) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown30", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">
    byre.compute @ConvOp_f16f16_f16(%35, %6, %31) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %37 = "byre.alias"(%alloc) <{offset = 44494592 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%31, %arg18, %arg19, %37) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">
    %38 = "byre.alias"(%alloc) <{offset = 13528832 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    %39 = "byre.alias"(%alloc) <{offset = 57339648 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xi1, "cuda">
    byre.compute @PTXOp(%37, %38, %39) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown28", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">
    %40 = "byre.alias"(%alloc) <{offset = 11923200 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%38, %7, %40) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %41 = "byre.alias"(%alloc) <{offset = 10317568 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%40, %arg23, %arg24, %41) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">
    %42 = "byre.alias"(%alloc) <{offset = 70184704 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xi1, "cuda">
    byre.compute @PTXOp(%41, %35, %37, %42) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown30", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">
    %43 = "byre.alias"(%alloc) <{offset = 58142464 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%37, %8, %43) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %44 = "byre.alias"(%alloc) <{offset = 10317568 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%43, %arg38, %arg39, %44) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %45 = "byre.alias"(%alloc) <{offset = 59748096 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%37, %9, %45) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %46 = "byre.alias"(%alloc) <{offset = 46100224 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%45, %arg28, %arg29, %46) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %47 = "byre.alias"(%alloc) <{offset = 60550912 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %48 = "byre.alias"(%alloc) <{offset = 3756032 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xi1, "cuda">
    byre.compute @PTXOp(%46, %47, %48) {BlockSize.x = 256 : i32, GridSize.x = 392 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown37", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">
    %49 = "byre.alias"(%alloc) <{offset = 62156544 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%47, %10, %49) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%49, %arg33, %arg34, %46) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %50 = "byre.alias"(%alloc) <{offset = 55734016 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %51 = "byre.alias"(%alloc) <{offset = 3354624 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xi1, "cuda">
    byre.compute @PTXOp(%46, %44, %50, %51) {BlockSize.x = 256 : i32, GridSize.x = 392 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown39", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">
    %52 = "byre.alias"(%alloc) <{offset = 61353728 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%50, %11, %52) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%52, %arg43, %arg44, %46) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %53 = "byre.alias"(%alloc) <{offset = 58945280 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    %54 = "byre.alias"(%alloc) <{offset = 2953216 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xi1, "cuda">
    byre.compute @PTXOp(%46, %53, %54) {BlockSize.x = 256 : i32, GridSize.x = 392 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown37", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">
    %55 = "byre.alias"(%alloc) <{offset = 56536832 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%53, %12, %55) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%55, %arg48, %arg49, %46) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">
    %56 = "byre.alias"(%alloc) <{offset = 4960256 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xi1, "cuda">
    byre.compute @PTXOp(%46, %50, %44, %56) {BlockSize.x = 256 : i32, GridSize.x = 392 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown39", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">
    %57 = "byre.alias"(%alloc) <{offset = 11120384 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%44, %13, %57) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %58 = "byre.alias"(%alloc) <{offset = 11521792 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%57, %arg63, %arg64, %58) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %59 = "byre.alias"(%alloc) <{offset = 46100224 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%44, %14, %59) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %60 = "byre.alias"(%alloc) <{offset = 46501632 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%59, %arg53, %arg54, %60) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %61 = "byre.alias"(%alloc) <{offset = 2551808 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %62 = "byre.alias"(%alloc) <{offset = 6704896 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xi1, "cuda">
    byre.compute @PTXOp(%60, %61, %62) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown46", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">
    byre.compute @ConvOp_f16f16_f16(%61, %15, %60) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %63 = "byre.alias"(%alloc) <{offset = 46903040 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%60, %arg58, %arg59, %63) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %64 = "byre.alias"(%alloc) <{offset = 4157440 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %65 = "byre.alias"(%alloc) <{offset = 6905600 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xi1, "cuda">
    byre.compute @PTXOp(%63, %58, %64, %65) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown48", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">
    %66 = "byre.alias"(%alloc) <{offset = 22736640 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%64, %16, %66) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%66, %arg68, %arg69, %63) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %67 = "byre.alias"(%alloc) <{offset = 71790336 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %68 = "byre.alias"(%alloc) <{offset = 2056192 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xi1, "cuda">
    byre.compute @PTXOp(%63, %67, %68) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown46", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">
    %69 = "byre.alias"(%alloc) <{offset = 9891584 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%67, %17, %69) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%69, %arg73, %arg74, %63) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">
    %70 = "byre.alias"(%alloc) <{offset = 72191744 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    %71 = "byre.alias"(%alloc) <{offset = 294912 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xi1, "cuda">
    byre.compute @PTXOp(%63, %64, %70, %71) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown48", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">
    %72 = "byre.alias"(%alloc) <{offset = 495616 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%70, %18, %72) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %73 = "byre.alias"(%alloc) <{offset = 11521792 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%72, %arg88, %arg89, %73) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %74 = "byre.alias"(%alloc) <{offset = 696320 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%70, %19, %74) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %75 = "byre.alias"(%alloc) <{offset = 46903040 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%74, %arg78, %arg79, %75) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %76 = "byre.alias"(%alloc) <{offset = 897024 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %77 = "byre.alias"(%alloc) <{offset = 6457088 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xi1, "cuda">
    byre.compute @PTXOp(%75, %76, %77) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown55", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">
    byre.compute @ConvOp_f16f16_f16(%76, %20, %75) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    %78 = "byre.alias"(%alloc) <{offset = 47103744 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%75, %arg83, %arg84, %78) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %79 = "byre.alias"(%alloc) <{offset = 1097728 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %80 = "byre.alias"(%alloc) <{offset = 4820992 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xi1, "cuda">
    byre.compute @PTXOp(%78, %73, %79, %80) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown57", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">
    %81 = "byre.alias"(%alloc) <{offset = 1298432 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%79, %21, %81) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%81, %arg93, %arg94, %78) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %82 = "byre.alias"(%alloc) <{offset = 1499136 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    %83 = "byre.alias"(%alloc) <{offset = 6356736 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xi1, "cuda">
    byre.compute @PTXOp(%78, %82, %83) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown55", memory_effects = [1 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">
    %84 = "byre.alias"(%alloc) <{offset = 72593152 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%82, %22, %84) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16(%84, %arg98, %arg99, %73) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">
    %85 = "byre.alias"(%alloc) <{offset = 72793856 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xi1, "cuda">
    byre.compute @PTXOp(%73, %79, %78, %85) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown57", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">
    %86 = "byre.alias"(%alloc) <{offset = 47103744 : i64}> : (memref<76533504xi8, "cuda">) -> memref<2048x49xf16, "cuda">
    %87 = "byre.alias"(%alloc) <{offset = 11521792 : i64}> : (memref<76533504xi8, "cuda">) -> memref<2048xf16, "cuda">
    byre.compute @PTXOp(%86, %87) {BlockSize.x = 64 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 2048 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown62_kernel"} : memref<2048x49xf16, "cuda">, memref<2048xf16, "cuda">
    %88 = "byre.alias"(%alloc) <{offset = 11521792 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512xf16, "cuda">
    %89 = "byre.alias"(%alloc) <{offset = 5435520 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512xf16, "cuda">
    byre.compute @PTXOp(%88, %89) {BlockSize.x = 256 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown63", memory_effects = [1 : i32, 2 : i32]} : memref<4x512xf16, "cuda">, memref<4x512xf16, "cuda">
    %90 = "byre.alias"(%alloc) <{offset = 47103744 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x1000xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%89, %24, %90) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<4x512xf16, "cuda">, memref<1000x512xf16, "cuda">, memref<4x1000xf16, "cuda">
    %91 = "byre.alias"(%alloc) <{offset = 11521792 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x1000xf16, "cuda">
    byre.compute @PTXOp(%25, %90, %91) {BlockSize.x = 256 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown64", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1000xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4x1000xf16, "cuda">
    %92 = "byre.alias"(%alloc) <{offset = 11529856 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4xf16, "cuda">
    byre.compute @PTXOp(%91, %92) {BlockSize.x = 512 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 4 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown65_kernel"} : memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">
    byre.compute @PTXOp(%92, %91, %90) {BlockSize.x = 256 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown66", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4x1000xf16, "cuda">
    %93 = "byre.alias"(%alloc) <{offset = 47111808 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4xf16, "cuda">
    byre.compute @PTXOp(%90, %93) {BlockSize.x = 512 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 4 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown67_kernel"} : memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">
    %94 = "byre.alias"(%alloc) <{offset = 11521792 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4xf16, "cuda">
    byre.compute @PTXOp(%93, %94) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown68", memory_effects = [1 : i32, 2 : i32]} : memref<4xf16, "cuda">, memref<4xf16, "cuda">
    %95 = "byre.alias"(%alloc) <{offset = 5447680 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x1000xf16, "cuda">
    %96 = "byre.alias"(%alloc) <{offset = 5455744 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x1000xf16, "cuda">
    byre.compute @PTXOp(%94, %90, %26, %23, %95, %96) {BlockSize.x = 256 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown69", memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<4x1000xf16, "cuda">
    %97 = "byre.alias"(%alloc) <{offset = 47103744 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%96, %24, %97) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<4x1000xf16, "cuda">, memref<1000x512xf16, "cuda">, memref<4x512xf16, "cuda">
    %98 = "byre.alias"(%alloc) <{offset = 72969984 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x512x7x7xf16, "cuda">
    byre.compute @PTXOp(%97, %85, %98) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [2 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown70", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x512xf16, "cuda">, memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%84, %arg98, %98, %78, %arg163, %arg164) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%78, %22, %73) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%82, %78, %22) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    byre.compute @PTXOp(%83, %73, %78) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown74", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%81, %arg93, %78, %73, %arg160, %arg161) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%73, %21, %78) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%79, %73, %21) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    byre.compute @PTXOp(%98, %78, %80, %84) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown78", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%75, %arg83, %84, %73, %arg154, %arg155) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%73, %20, %75) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%76, %73, %20) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    byre.compute @PTXOp(%77, %75, %73) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown74", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x512x7x7xi1, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%74, %arg78, %73, %75, %arg151, %arg152) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%75, %19, %58) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%70, %75, %19) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%72, %arg88, %84, %98, %arg157, %arg158) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%98, %18, %63) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %99 = "byre.alias"(%alloc) <{offset = 1499136 : i64}> : (memref<76533504xi8, "cuda">) -> memref<512x256x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%70, %98, %99) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">
    %100 = "byre.alias"(%alloc) <{offset = 4558848 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @PTXOp(%63, %58, %71, %100) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown89", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%69, %arg73, %100, %63, %arg148, %arg149) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%63, %17, %58) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %101 = "byre.alias"(%alloc) <{offset = 72969984 : i64}> : (memref<76533504xi8, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%67, %63, %101) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%68, %58, %63) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown93", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%66, %arg68, %63, %58, %arg145, %arg146) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%58, %16, %63) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    %102 = "byre.alias"(%alloc) <{offset = 71790336 : i64}> : (memref<76533504xi8, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%64, %58, %102) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    %103 = "byre.alias"(%alloc) <{offset = 21556992 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @PTXOp(%100, %63, %65, %103) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown89", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">
    %104 = "byre.alias"(%alloc) <{offset = 8711936 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%60, %arg58, %103, %104, %arg139, %arg140) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%104, %15, %58) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%61, %104, %15) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%62, %58, %60) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown93", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x256x14x14xi1, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%59, %arg53, %60, %58, %arg136, %arg137) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%58, %14, %46) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%44, %58, %14) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%57, %arg63, %103, %104, %arg142, %arg143) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %105 = "byre.alias"(%alloc) <{offset = 46903040 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%104, %13, %105) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %106 = "byre.alias"(%alloc) <{offset = 4558848 : i64}> : (memref<76533504xi8, "cuda">) -> memref<256x128x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%44, %104, %106) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">
    byre.compute @PTXOp(%105, %46, %56, %44) {BlockSize.x = 256 : i32, GridSize.x = 392 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown108", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%55, %arg48, %44, %46, %arg133, %arg134) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %107 = "byre.alias"(%alloc) <{offset = 11120384 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%46, %12, %107) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %108 = "byre.alias"(%alloc) <{offset = 56536832 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%53, %46, %108) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    byre.compute @PTXOp(%54, %107, %46) {BlockSize.x = 256 : i32, GridSize.x = 392 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown112", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%52, %arg43, %46, %107, %arg130, %arg131) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%107, %11, %46) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %109 = "byre.alias"(%alloc) <{offset = 61353728 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%50, %107, %109) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    %110 = "byre.alias"(%alloc) <{offset = 8711936 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @PTXOp(%44, %46, %51, %110) {BlockSize.x = 256 : i32, GridSize.x = 392 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown108", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%49, %arg33, %110, %46, %arg124, %arg125) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%46, %10, %44) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %111 = "byre.alias"(%alloc) <{offset = 55734016 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%47, %46, %111) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    byre.compute @PTXOp(%48, %44, %46) {BlockSize.x = 256 : i32, GridSize.x = 392 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown112", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x128x28x28xi1, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">
    %112 = "byre.alias"(%alloc) <{offset = 9514752 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%45, %arg28, %46, %112, %arg121, %arg122) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%112, %9, %41) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %113 = "byre.alias"(%alloc) <{offset = 56028928 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%37, %112, %113) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%43, %arg38, %110, %46, %arg127, %arg128) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %114 = "byre.alias"(%alloc) <{offset = 8711936 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%46, %8, %114) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %115 = "byre.alias"(%alloc) <{offset = 62156544 : i64}> : (memref<76533504xi8, "cuda">) -> memref<128x64x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%37, %46, %115) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">
    %116 = "byre.alias"(%alloc) <{offset = 21556992 : i64}> : (memref<76533504xi8, "cuda">) -> memref<4x64x56x56xf16, "cuda">
    byre.compute @PTXOp(%114, %41, %42, %116) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown127", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%40, %arg23, %116, %37, %arg118, %arg119) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%37, %7, %40) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %117 = "byre.alias"(%alloc) <{offset = 10317568 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%38, %37, %117) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%39, %40, %37) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown131", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%31, %arg18, %37, %38, %arg115, %arg116) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%38, %6, %31) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %118 = "byre.alias"(%alloc) <{offset = 11923200 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%35, %38, %118) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%116, %31, %36, %35) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown127", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%34, %arg13, %35, %31, %arg112, %arg113) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%31, %5, %34) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %119 = "byre.alias"(%alloc) <{offset = 13528832 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%32, %31, %119) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%33, %34, %31) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown131", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xi1, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%30, %arg8, %31, %34, %arg109, %arg110) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%34, %4, %31) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    %120 = "byre.alias"(%alloc) <{offset = 19951360 : i64}> : (memref<76533504xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%29, %34, %120) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%35, %31, %34) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown143", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x56x56xf16, "cuda">
    byre.compute @PoolMaxGradOp_f16f16_f16(%27, %34, %3) {device = "cuda", memory_effects = [1 : i32, 1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16, "cuda">, memref<4x64x56x56xf16, "cuda">, memref<4x64x112x112xf16, "cuda">
    byre.compute @PTXOp(%28, %3, %27) {BlockSize.x = 256 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown144", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<4x64x112x112xi1, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%2, %arg3, %27, %3, %arg106, %arg107) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%0, %3, %1) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16, "cuda">, memref<4x64x112x112xf16, "cuda">, memref<64x3x7x7xf16, "cuda">
    %121 = "byre.alias"(%alloc) <{offset = 62978176 : i64}> : (memref<76533504xi8, "cuda">) -> memref<f32, "cuda">
    %122 = "byre.alias"(%alloc) <{offset = 5447680 : i64}> : (memref<76533504xi8, "cuda">) -> memref<32x125xf16, "cuda">
    %123 = "byre.alias"(%arg1) <{offset = 0 : i64}> : (memref<4x1000xf32, "cuda">) -> memref<32x125xf32, "cuda">
    %124 = "byre.alias"(%alloc) <{offset = 49311488 : i64}> : (memref<76533504xi8, "cuda">) -> memref<32xf32, "cuda">
    byre.compute @PTXOp(%122, %123, %124) {BlockSize.x = 128 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 32 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown147_kernel"} : memref<32x125xf16, "cuda">, memref<32x125xf32, "cuda">, memref<32xf32, "cuda">
    byre.compute @PTXOp(%124, %121) {BlockSize.x = 32 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 1 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown147_kernel_0"} : memref<32xf32, "cuda">, memref<f32, "cuda">
    byre.compute @PTXOp(%121, %arg104) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown148", memory_effects = [1 : i32, 2 : i32]} : memref<f32, "cuda">, memref<f32, "cuda">
    byre.compute @PTXOp(%1, %arg105) {BlockSize.x = 256 : i32, GridSize.x = 10 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown149", memory_effects = [1 : i32, 2 : i32]} : memref<64x3x7x7xf16, "cuda">, memref<64x3x7x7xf32, "cuda">
    byre.compute @PTXOp(%120, %arg108) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown150", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf16, "cuda">, memref<64x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%119, %arg111) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown150", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf16, "cuda">, memref<64x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%118, %arg114) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown150", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf16, "cuda">, memref<64x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%117, %arg117) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown150", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf16, "cuda">, memref<64x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%113, %arg120) {BlockSize.x = 256 : i32, GridSize.x = 72 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown154", memory_effects = [1 : i32, 2 : i32]} : memref<128x64x3x3xf16, "cuda">, memref<128x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%111, %arg123) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown155", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf16, "cuda">, memref<128x128x3x3xf32, "cuda">
    byre.compute @PTXOp(%115, %arg126) {BlockSize.x = 256 : i32, GridSize.x = 8 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown156", memory_effects = [1 : i32, 2 : i32]} : memref<128x64x1x1xf16, "cuda">, memref<128x64x1x1xf32, "cuda">
    byre.compute @PTXOp(%109, %arg129) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown155", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf16, "cuda">, memref<128x128x3x3xf32, "cuda">
    byre.compute @PTXOp(%108, %arg132) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown155", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf16, "cuda">, memref<128x128x3x3xf32, "cuda">
    byre.compute @PTXOp(%14, %arg135) {BlockSize.x = 256 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown159", memory_effects = [1 : i32, 2 : i32]} : memref<256x128x3x3xf16, "cuda">, memref<256x128x3x3xf32, "cuda">
    byre.compute @PTXOp(%15, %arg138) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown160", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf16, "cuda">, memref<256x256x3x3xf32, "cuda">
    byre.compute @PTXOp(%106, %arg141) {BlockSize.x = 256 : i32, GridSize.x = 32 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown161", memory_effects = [1 : i32, 2 : i32]} : memref<256x128x1x1xf16, "cuda">, memref<256x128x1x1xf32, "cuda">
    byre.compute @PTXOp(%102, %arg144) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown160", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf16, "cuda">, memref<256x256x3x3xf32, "cuda">
    byre.compute @PTXOp(%101, %arg147) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown160", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf16, "cuda">, memref<256x256x3x3xf32, "cuda">
    byre.compute @PTXOp(%19, %arg150) {BlockSize.x = 256 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown164", memory_effects = [1 : i32, 2 : i32]} : memref<512x256x3x3xf16, "cuda">, memref<512x256x3x3xf32, "cuda">
    byre.compute @PTXOp(%20, %arg153) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown165", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf16, "cuda">, memref<512x512x3x3xf32, "cuda">
    byre.compute @PTXOp(%99, %arg156) {BlockSize.x = 256 : i32, GridSize.x = 128 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown166", memory_effects = [1 : i32, 2 : i32]} : memref<512x256x1x1xf16, "cuda">, memref<512x256x1x1xf32, "cuda">
    byre.compute @PTXOp(%21, %arg159) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown165", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf16, "cuda">, memref<512x512x3x3xf32, "cuda">
    byre.compute @PTXOp(%22, %arg162) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown165", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf16, "cuda">, memref<512x512x3x3xf32, "cuda">
    %125 = "byre.alias"(%alloc) <{offset = 62959360 : i64}> : (memref<76533504xi8, "cuda">) -> memref<1000x512xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%89, %96, %125) {device = "cuda", lhs_contracting_dimension = 0 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_transpose, rhs_contracting_dimension = 0 : i64} : memref<4x512xf16, "cuda">, memref<4x1000xf16, "cuda">, memref<1000x512xf16, "cuda">
    byre.compute @PTXOp(%125, %arg165) {BlockSize.x = 256 : i32, GridSize.x = 500 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown170", memory_effects = [1 : i32, 2 : i32]} : memref<1000x512xf16, "cuda">, memref<1000x512xf32, "cuda">
    %126 = "byre.alias"(%alloc) <{offset = 62959360 : i64}> : (memref<76533504xi8, "cuda">) -> memref<1000xf32, "cuda">
    byre.compute @PTXOp(%96, %126) {BlockSize.x = 32 : i32, BlockSize.y = 2 : i32, BlockSize.z = 1 : i32, GridSize.x = 32 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown171_kernel"} : memref<4x1000xf16, "cuda">, memref<1000xf32, "cuda">
    byre.compute @PTXOp(%126, %arg166) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown172", memory_effects = [1 : i32, 2 : i32]} : memref<1000xf32, "cuda">, memref<1000xf32, "cuda">
    return
  }
}