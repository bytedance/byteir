// RUN: byteir-opt %s -remove-func-body="anchor-attr=__byteir_elementwise_fusion__" -inline -gpu-launch-func-to-byre -set-op-space="entry-func=main space=cuda" -set-arg-space="entry-func=main all-space=cuda" | FileCheck %s

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
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %c-1024 = arith.constant -1024 : index
      %c1000 = arith.constant 1000 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %12 = gpu.block_id  x
      %subview = memref.subview %arg0[%12, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      %13 = gpu.thread_id  x
      %14 = arith.muli %13, %c2 : index
      %15 = arith.cmpi slt, %13, %c0 : index
      %16 = arith.subi %c-1, %13 : index
      %17 = arith.select %15, %16, %13 : index
      %18 = arith.divsi %17, %c512 : index
      %19 = arith.subi %c-1, %18 : index
      %20 = arith.select %15, %19, %18 : index
      %21 = arith.muli %20, %c-1024 : index
      %22 = arith.addi %14, %21 : index
      %23 = arith.cmpi slt, %22, %c1000 : index
      %24 = arith.select %23, %22, %c1000 : index
      %25 = arith.addi %22, %c2 : index
      %26 = arith.cmpi slt, %25, %c1000 : index
      %27 = arith.select %26, %25, %c1000 : index
      %28 = arith.subi %27, %24 : index
      %subview_0 = memref.subview %expand_shape[0, %24] [1, %28] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %29 = arith.cmpi ugt, %28, %c0 : index
      %30 = scf.if %29 -> (f16) {
        %44 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %44 : f16
      } else {
        scf.yield %cst : f16
      }
      %31 = arith.addf %30, %cst : f16
      %32 = arith.cmpi ugt, %28, %c1 : index
      %33 = scf.if %32 -> (f16) {
        %44 = memref.load %expand_shape_1[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %44 : f16
      } else {
        scf.yield %cst : f16
      }
      %34 = arith.addf %31, %33 : f16
      memref.store %34, %alloca[%13] : memref<512xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      %35 = arith.cmpi ult, %13, %c256 : index
      scf.if %35 {
        %44 = memref.load %alloca[%14] : memref<512xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca[%46] : memref<512xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %alloca_2[%13] : memref<256xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      %36 = arith.cmpi ult, %13, %c128 : index
      scf.if %36 {
        %44 = memref.load %alloca_2[%14] : memref<256xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca_2[%46] : memref<256xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %alloca_3[%13] : memref<128xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %37 = arith.cmpi ult, %13, %c64 : index
      scf.if %37 {
        %44 = memref.load %alloca_3[%14] : memref<128xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca_3[%46] : memref<128xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %alloca_4[%13] : memref<64xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %38 = arith.cmpi ult, %13, %c32 : index
      scf.if %38 {
        %44 = memref.load %alloca_4[%14] : memref<64xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca_4[%46] : memref<64xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %alloca_5[%13] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %39 = arith.cmpi ult, %13, %c16 : index
      scf.if %39 {
        %44 = memref.load %alloca_5[%14] : memref<32xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca_5[%46] : memref<32xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %alloca_6[%13] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_7 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %40 = arith.cmpi ult, %13, %c8 : index
      scf.if %40 {
        %44 = memref.load %alloca_6[%14] : memref<16xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca_6[%46] : memref<16xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %alloca_7[%13] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_8 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %41 = arith.cmpi ult, %13, %c4 : index
      scf.if %41 {
        %44 = memref.load %alloca_7[%14] : memref<8xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca_7[%46] : memref<8xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %alloca_8[%13] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_9 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %42 = arith.cmpi ult, %13, %c2 : index
      scf.if %42 {
        %44 = memref.load %alloca_8[%14] : memref<4xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca_8[%46] : memref<4xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %alloca_9[%13] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %43 = arith.cmpi ult, %13, %c1 : index
      scf.if %43 {
        %44 = memref.load %alloca_9[%14] : memref<2xf16, #gpu.address_space<workgroup>>
        %45 = arith.addf %44, %cst : f16
        %46 = arith.addi %14, %c1 : index
        %47 = memref.load %alloca_9[%46] : memref<2xf16, #gpu.address_space<workgroup>>
        %48 = arith.addf %47, %45 : f16
        memref.store %48, %arg1[%12] : memref<4xf16>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown62_kernel(%arg0: memref<2048x49xf16>, %arg1: memref<2048xf16>) kernel attributes {gpu.known_block_size = array<i32: 64, 1, 1>, gpu.known_grid_size = array<i32: 2048, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c49 = arith.constant 49 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %12 = gpu.block_id  x
      %subview = memref.subview %arg0[%12, 0] [1, 49] [1, 1] : memref<2048x49xf16> to memref<49xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<49xf16, strided<[1], offset: ?>> into memref<1x49xf16, strided<[49, 1], offset: ?>>
      %alloca = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %13 = gpu.thread_id  x
      %14 = arith.remsi %13, %c64 : index
      %15 = arith.cmpi slt, %14, %c0 : index
      %16 = arith.addi %14, %c64 : index
      %17 = arith.select %15, %16, %14 : index
      %18 = arith.cmpi slt, %17, %c49 : index
      %19 = arith.select %18, %17, %c49 : index
      %20 = arith.addi %17, %c1 : index
      %21 = arith.cmpi slt, %20, %c49 : index
      %22 = arith.select %21, %20, %c49 : index
      %23 = arith.subi %22, %19 : index
      %subview_0 = memref.subview %expand_shape[0, %19] [1, %23] [1, 1] : memref<1x49xf16, strided<[49, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %24 = arith.cmpi ugt, %23, %c0 : index
      %25 = scf.if %24 -> (f16) {
        %33 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %33 : f16
      } else {
        scf.yield %cst : f16
      }
      %26 = arith.addf %25, %cst : f16
      memref.store %26, %alloca[%13] : memref<64xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %27 = arith.cmpi ult, %13, %c32 : index
      scf.if %27 {
        %33 = arith.muli %13, %c2 : index
        %34 = memref.load %alloca[%33] : memref<64xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %33, %c1 : index
        %37 = memref.load %alloca[%36] : memref<64xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_2[%13] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %28 = arith.cmpi ult, %13, %c16 : index
      scf.if %28 {
        %33 = arith.muli %13, %c2 : index
        %34 = memref.load %alloca_2[%33] : memref<32xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %33, %c1 : index
        %37 = memref.load %alloca_2[%36] : memref<32xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_3[%13] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %29 = arith.cmpi ult, %13, %c8 : index
      scf.if %29 {
        %33 = arith.muli %13, %c2 : index
        %34 = memref.load %alloca_3[%33] : memref<16xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %33, %c1 : index
        %37 = memref.load %alloca_3[%36] : memref<16xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_4[%13] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %30 = arith.cmpi ult, %13, %c4 : index
      scf.if %30 {
        %33 = arith.muli %13, %c2 : index
        %34 = memref.load %alloca_4[%33] : memref<8xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %33, %c1 : index
        %37 = memref.load %alloca_4[%36] : memref<8xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_5[%13] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %31 = arith.cmpi ult, %13, %c2 : index
      scf.if %31 {
        %33 = arith.muli %13, %c2 : index
        %34 = memref.load %alloca_5[%33] : memref<4xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %33, %c1 : index
        %37 = memref.load %alloca_5[%36] : memref<4xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %alloca_6[%13] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %32 = arith.cmpi ult, %13, %c1 : index
      scf.if %32 {
        %33 = arith.muli %13, %c2 : index
        %34 = memref.load %alloca_6[%33] : memref<2xf16, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst : f16
        %36 = arith.addi %33, %c1 : index
        %37 = memref.load %alloca_6[%36] : memref<2xf16, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %35 : f16
        memref.store %38, %arg1[%12] : memref<2048xf16>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown65_kernel(%arg0: memref<4x1000xf16>, %arg1: memref<4xf16>) kernel attributes {gpu.known_block_size = array<i32: 512, 1, 1>, gpu.known_grid_size = array<i32: 4, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %c-1024 = arith.constant -1024 : index
      %c1000 = arith.constant 1000 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %12 = gpu.block_id  x
      %subview = memref.subview %arg0[%12, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      %13 = gpu.thread_id  x
      %14 = arith.muli %13, %c2 : index
      %15 = arith.cmpi slt, %13, %c0 : index
      %16 = arith.subi %c-1, %13 : index
      %17 = arith.select %15, %16, %13 : index
      %18 = arith.divsi %17, %c512 : index
      %19 = arith.subi %c-1, %18 : index
      %20 = arith.select %15, %19, %18 : index
      %21 = arith.muli %20, %c-1024 : index
      %22 = arith.addi %14, %21 : index
      %23 = arith.cmpi slt, %22, %c1000 : index
      %24 = arith.select %23, %22, %c1000 : index
      %25 = arith.addi %22, %c2 : index
      %26 = arith.cmpi slt, %25, %c1000 : index
      %27 = arith.select %26, %25, %c1000 : index
      %28 = arith.subi %27, %24 : index
      %subview_0 = memref.subview %expand_shape[0, %24] [1, %28] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %29 = arith.cmpi ugt, %28, %c0 : index
      %30 = scf.if %29 -> (f16) {
        %43 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %43 : f16
      } else {
        scf.yield %cst : f16
      }
      %31 = arith.cmpi ugt, %28, %c1 : index
      %32 = scf.if %31 -> (f16) {
        %43 = memref.load %expand_shape_1[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %43 : f16
      } else {
        scf.yield %cst : f16
      }
      %33 = arith.maximumf %30, %32 : f16
      memref.store %33, %alloca[%13] : memref<512xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      %34 = arith.cmpi ult, %13, %c256 : index
      scf.if %34 {
        %43 = memref.load %alloca[%14] : memref<512xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca[%44] : memref<512xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %alloca_2[%13] : memref<256xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      %35 = arith.cmpi ult, %13, %c128 : index
      scf.if %35 {
        %43 = memref.load %alloca_2[%14] : memref<256xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca_2[%44] : memref<256xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %alloca_3[%13] : memref<128xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %36 = arith.cmpi ult, %13, %c64 : index
      scf.if %36 {
        %43 = memref.load %alloca_3[%14] : memref<128xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca_3[%44] : memref<128xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %alloca_4[%13] : memref<64xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %37 = arith.cmpi ult, %13, %c32 : index
      scf.if %37 {
        %43 = memref.load %alloca_4[%14] : memref<64xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca_4[%44] : memref<64xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %alloca_5[%13] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %38 = arith.cmpi ult, %13, %c16 : index
      scf.if %38 {
        %43 = memref.load %alloca_5[%14] : memref<32xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca_5[%44] : memref<32xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %alloca_6[%13] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_7 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %39 = arith.cmpi ult, %13, %c8 : index
      scf.if %39 {
        %43 = memref.load %alloca_6[%14] : memref<16xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca_6[%44] : memref<16xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %alloca_7[%13] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_8 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %40 = arith.cmpi ult, %13, %c4 : index
      scf.if %40 {
        %43 = memref.load %alloca_7[%14] : memref<8xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca_7[%44] : memref<8xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %alloca_8[%13] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_9 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %41 = arith.cmpi ult, %13, %c2 : index
      scf.if %41 {
        %43 = memref.load %alloca_8[%14] : memref<4xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca_8[%44] : memref<4xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %alloca_9[%13] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %42 = arith.cmpi ult, %13, %c1 : index
      scf.if %42 {
        %43 = memref.load %alloca_9[%14] : memref<2xf16, #gpu.address_space<workgroup>>
        %44 = arith.addi %14, %c1 : index
        %45 = memref.load %alloca_9[%44] : memref<2xf16, #gpu.address_space<workgroup>>
        %46 = arith.maximumf %45, %43 : f16
        memref.store %46, %arg1[%12] : memref<4xf16>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown67_kernel(%arg0: memref<4x1000xf16>, %arg1: memref<4xf16>) kernel attributes {gpu.known_block_size = array<i32: 512, 1, 1>, gpu.known_grid_size = array<i32: 4, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c-1 = arith.constant -1 : index
      %c512 = arith.constant 512 : index
      %c-1024 = arith.constant -1024 : index
      %c1000 = arith.constant 1000 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %12 = gpu.block_id  x
      %subview = memref.subview %arg0[%12, 0] [1, 1000] [1, 1] : memref<4x1000xf16> to memref<1000xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<1000xf16, strided<[1], offset: ?>> into memref<1x1000xf16, strided<[1000, 1], offset: ?>>
      %alloca = memref.alloca() : memref<512xf16, #gpu.address_space<workgroup>>
      %13 = gpu.thread_id  x
      %14 = arith.muli %13, %c2 : index
      %15 = arith.cmpi slt, %13, %c0 : index
      %16 = arith.subi %c-1, %13 : index
      %17 = arith.select %15, %16, %13 : index
      %18 = arith.divsi %17, %c512 : index
      %19 = arith.subi %c-1, %18 : index
      %20 = arith.select %15, %19, %18 : index
      %21 = arith.muli %20, %c-1024 : index
      %22 = arith.addi %14, %21 : index
      %23 = arith.cmpi slt, %22, %c1000 : index
      %24 = arith.select %23, %22, %c1000 : index
      %25 = arith.addi %22, %c2 : index
      %26 = arith.cmpi slt, %25, %c1000 : index
      %27 = arith.select %26, %25, %c1000 : index
      %28 = arith.subi %27, %24 : index
      %subview_0 = memref.subview %expand_shape[0, %24] [1, %28] [1, 1] : memref<1x1000xf16, strided<[1000, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %29 = arith.cmpi ugt, %28, %c0 : index
      %30 = scf.if %29 -> (f16) {
        %46 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %46 : f16
      } else {
        scf.yield %cst : f16
      }
      %31 = math.exp %30 : f16
      %32 = arith.addf %31, %cst : f16
      %33 = arith.cmpi ugt, %28, %c1 : index
      %34 = scf.if %33 -> (f16) {
        %46 = memref.load %expand_shape_1[%c0, %c1] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %46 : f16
      } else {
        scf.yield %cst : f16
      }
      %35 = math.exp %34 : f16
      %36 = arith.addf %32, %35 : f16
      memref.store %36, %alloca[%13] : memref<512xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<256xf16, #gpu.address_space<workgroup>>
      %37 = arith.cmpi ult, %13, %c256 : index
      scf.if %37 {
        %46 = memref.load %alloca[%14] : memref<512xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca[%48] : memref<512xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %alloca_2[%13] : memref<256xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<128xf16, #gpu.address_space<workgroup>>
      %38 = arith.cmpi ult, %13, %c128 : index
      scf.if %38 {
        %46 = memref.load %alloca_2[%14] : memref<256xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca_2[%48] : memref<256xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %alloca_3[%13] : memref<128xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %39 = arith.cmpi ult, %13, %c64 : index
      scf.if %39 {
        %46 = memref.load %alloca_3[%14] : memref<128xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca_3[%48] : memref<128xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %alloca_4[%13] : memref<64xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %40 = arith.cmpi ult, %13, %c32 : index
      scf.if %40 {
        %46 = memref.load %alloca_4[%14] : memref<64xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca_4[%48] : memref<64xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %alloca_5[%13] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %41 = arith.cmpi ult, %13, %c16 : index
      scf.if %41 {
        %46 = memref.load %alloca_5[%14] : memref<32xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca_5[%48] : memref<32xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %alloca_6[%13] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_7 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %42 = arith.cmpi ult, %13, %c8 : index
      scf.if %42 {
        %46 = memref.load %alloca_6[%14] : memref<16xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca_6[%48] : memref<16xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %alloca_7[%13] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_8 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %43 = arith.cmpi ult, %13, %c4 : index
      scf.if %43 {
        %46 = memref.load %alloca_7[%14] : memref<8xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca_7[%48] : memref<8xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %alloca_8[%13] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_9 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %44 = arith.cmpi ult, %13, %c2 : index
      scf.if %44 {
        %46 = memref.load %alloca_8[%14] : memref<4xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca_8[%48] : memref<4xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %alloca_9[%13] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %45 = arith.cmpi ult, %13, %c1 : index
      scf.if %45 {
        %46 = memref.load %alloca_9[%14] : memref<2xf16, #gpu.address_space<workgroup>>
        %47 = arith.addf %46, %cst : f16
        %48 = arith.addi %14, %c1 : index
        %49 = memref.load %alloca_9[%48] : memref<2xf16, #gpu.address_space<workgroup>>
        %50 = arith.addf %49, %47 : f16
        memref.store %50, %arg1[%12] : memref<4xf16>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown147_kernel(%arg0: memref<32x125xf16>, %arg1: memref<32x125xf32>, %arg2: memref<32xf32>) kernel attributes {gpu.known_block_size = array<i32: 128, 1, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c125 = arith.constant 125 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %12 = gpu.block_id  x
      %subview = memref.subview %arg0[%12, 0] [1, 125] [1, 1] : memref<32x125xf16> to memref<125xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<125xf16, strided<[1], offset: ?>> into memref<1x125xf16, strided<[125, 1], offset: ?>>
      %subview_1 = memref.subview %arg1[%12, 0] [1, 125] [1, 1] : memref<32x125xf32> to memref<125xf32, strided<[1], offset: ?>>
      %expand_shape_2 = memref.expand_shape %subview_1 [[0, 1]] : memref<125xf32, strided<[1], offset: ?>> into memref<1x125xf32, strided<[125, 1], offset: ?>>
      %alloca = memref.alloca() : memref<128xf32, #gpu.address_space<workgroup>>
      %13 = gpu.thread_id  x
      %14 = arith.remsi %13, %c128 : index
      %15 = arith.cmpi slt, %14, %c0 : index
      %16 = arith.addi %14, %c128 : index
      %17 = arith.select %15, %16, %14 : index
      %18 = arith.cmpi slt, %17, %c125 : index
      %19 = arith.select %18, %17, %c125 : index
      %20 = arith.addi %17, %c1 : index
      %21 = arith.cmpi slt, %20, %c125 : index
      %22 = arith.select %21, %20, %c125 : index
      %23 = arith.subi %22, %19 : index
      %subview_3 = memref.subview %expand_shape[0, %19] [1, %23] [1, 1] : memref<1x125xf16, strided<[125, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_4 = memref.expand_shape %subview_3 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %expand_shape_2[0, %19] [1, %23] [1, 1] : memref<1x125xf32, strided<[125, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %expand_shape_6 = memref.expand_shape %subview_5 [[0, 1]] : memref<?xf32, strided<[1], offset: ?>> into memref<1x?xf32, strided<[?, 1], offset: ?>>
      %24 = arith.cmpi ugt, %23, %c0 : index
      %25:2 = scf.if %24 -> (f16, f32) {
        %36 = memref.load %expand_shape_4[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        %37 = memref.load %expand_shape_6[%c0, %c0] : memref<1x?xf32, strided<[?, 1], offset: ?>>
        scf.yield %36, %37 : f16, f32
      } else {
        scf.yield %cst, %cst_0 : f16, f32
      }
      %26 = arith.extf %25#0 : f16 to f32
      %27 = arith.mulf %26, %25#1 : f32
      %28 = arith.addf %27, %cst_0 : f32
      memref.store %28, %alloca[%13] : memref<128xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_7 = memref.alloca() : memref<64xf32, #gpu.address_space<workgroup>>
      %29 = arith.cmpi ult, %13, %c64 : index
      scf.if %29 {
        %36 = arith.muli %13, %c2 : index
        %37 = memref.load %alloca[%36] : memref<128xf32, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %cst_0 : f32
        %39 = arith.addi %36, %c1 : index
        %40 = memref.load %alloca[%39] : memref<128xf32, #gpu.address_space<workgroup>>
        %41 = arith.addf %40, %38 : f32
        memref.store %41, %alloca_7[%13] : memref<64xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_8 = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      %30 = arith.cmpi ult, %13, %c32 : index
      scf.if %30 {
        %36 = arith.muli %13, %c2 : index
        %37 = memref.load %alloca_7[%36] : memref<64xf32, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %cst_0 : f32
        %39 = arith.addi %36, %c1 : index
        %40 = memref.load %alloca_7[%39] : memref<64xf32, #gpu.address_space<workgroup>>
        %41 = arith.addf %40, %38 : f32
        memref.store %41, %alloca_8[%13] : memref<32xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_9 = memref.alloca() : memref<16xf32, #gpu.address_space<workgroup>>
      %31 = arith.cmpi ult, %13, %c16 : index
      scf.if %31 {
        %36 = arith.muli %13, %c2 : index
        %37 = memref.load %alloca_8[%36] : memref<32xf32, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %cst_0 : f32
        %39 = arith.addi %36, %c1 : index
        %40 = memref.load %alloca_8[%39] : memref<32xf32, #gpu.address_space<workgroup>>
        %41 = arith.addf %40, %38 : f32
        memref.store %41, %alloca_9[%13] : memref<16xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_10 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
      %32 = arith.cmpi ult, %13, %c8 : index
      scf.if %32 {
        %36 = arith.muli %13, %c2 : index
        %37 = memref.load %alloca_9[%36] : memref<16xf32, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %cst_0 : f32
        %39 = arith.addi %36, %c1 : index
        %40 = memref.load %alloca_9[%39] : memref<16xf32, #gpu.address_space<workgroup>>
        %41 = arith.addf %40, %38 : f32
        memref.store %41, %alloca_10[%13] : memref<8xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_11 = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
      %33 = arith.cmpi ult, %13, %c4 : index
      scf.if %33 {
        %36 = arith.muli %13, %c2 : index
        %37 = memref.load %alloca_10[%36] : memref<8xf32, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %cst_0 : f32
        %39 = arith.addi %36, %c1 : index
        %40 = memref.load %alloca_10[%39] : memref<8xf32, #gpu.address_space<workgroup>>
        %41 = arith.addf %40, %38 : f32
        memref.store %41, %alloca_11[%13] : memref<4xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_12 = memref.alloca() : memref<2xf32, #gpu.address_space<workgroup>>
      %34 = arith.cmpi ult, %13, %c2 : index
      scf.if %34 {
        %36 = arith.muli %13, %c2 : index
        %37 = memref.load %alloca_11[%36] : memref<4xf32, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %cst_0 : f32
        %39 = arith.addi %36, %c1 : index
        %40 = memref.load %alloca_11[%39] : memref<4xf32, #gpu.address_space<workgroup>>
        %41 = arith.addf %40, %38 : f32
        memref.store %41, %alloca_12[%13] : memref<2xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %35 = arith.cmpi ult, %13, %c1 : index
      scf.if %35 {
        %36 = arith.muli %13, %c2 : index
        %37 = memref.load %alloca_12[%36] : memref<2xf32, #gpu.address_space<workgroup>>
        %38 = arith.addf %37, %cst_0 : f32
        %39 = arith.addi %36, %c1 : index
        %40 = memref.load %alloca_12[%39] : memref<2xf32, #gpu.address_space<workgroup>>
        %41 = arith.addf %40, %38 : f32
        memref.store %41, %arg2[%12] : memref<32xf32>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown147_kernel_0(%arg0: memref<32xf32>, %arg1: memref<f32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c32 = arith.constant 32 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %12 = gpu.block_id  x
      %alloca = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      %13 = gpu.thread_id  x
      %14 = arith.muli %12, %c32 : index
      %15 = arith.addi %14, %13 : index
      %16 = memref.load %arg0[%15] : memref<32xf32>
      %17 = arith.addf %16, %cst : f32
      memref.store %17, %alloca[%13] : memref<32xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_0 = memref.alloca() : memref<16xf32, #gpu.address_space<workgroup>>
      %18 = arith.cmpi ult, %13, %c16 : index
      scf.if %18 {
        %23 = arith.muli %13, %c2 : index
        %24 = memref.load %alloca[%23] : memref<32xf32, #gpu.address_space<workgroup>>
        %25 = arith.addf %24, %cst : f32
        %26 = arith.addi %23, %c1 : index
        %27 = memref.load %alloca[%26] : memref<32xf32, #gpu.address_space<workgroup>>
        %28 = arith.addf %27, %25 : f32
        memref.store %28, %alloca_0[%13] : memref<16xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_1 = memref.alloca() : memref<8xf32, #gpu.address_space<workgroup>>
      %19 = arith.cmpi ult, %13, %c8 : index
      scf.if %19 {
        %23 = arith.muli %13, %c2 : index
        %24 = memref.load %alloca_0[%23] : memref<16xf32, #gpu.address_space<workgroup>>
        %25 = arith.addf %24, %cst : f32
        %26 = arith.addi %23, %c1 : index
        %27 = memref.load %alloca_0[%26] : memref<16xf32, #gpu.address_space<workgroup>>
        %28 = arith.addf %27, %25 : f32
        memref.store %28, %alloca_1[%13] : memref<8xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
      %20 = arith.cmpi ult, %13, %c4 : index
      scf.if %20 {
        %23 = arith.muli %13, %c2 : index
        %24 = memref.load %alloca_1[%23] : memref<8xf32, #gpu.address_space<workgroup>>
        %25 = arith.addf %24, %cst : f32
        %26 = arith.addi %23, %c1 : index
        %27 = memref.load %alloca_1[%26] : memref<8xf32, #gpu.address_space<workgroup>>
        %28 = arith.addf %27, %25 : f32
        memref.store %28, %alloca_2[%13] : memref<4xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<2xf32, #gpu.address_space<workgroup>>
      %21 = arith.cmpi ult, %13, %c2 : index
      scf.if %21 {
        %23 = arith.muli %13, %c2 : index
        %24 = memref.load %alloca_2[%23] : memref<4xf32, #gpu.address_space<workgroup>>
        %25 = arith.addf %24, %cst : f32
        %26 = arith.addi %23, %c1 : index
        %27 = memref.load %alloca_2[%26] : memref<4xf32, #gpu.address_space<workgroup>>
        %28 = arith.addf %27, %25 : f32
        memref.store %28, %alloca_3[%13] : memref<2xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %22 = arith.cmpi ult, %13, %c1 : index
      scf.if %22 {
        %23 = arith.muli %13, %c2 : index
        %24 = memref.load %alloca_3[%23] : memref<2xf32, #gpu.address_space<workgroup>>
        %25 = arith.addf %24, %cst : f32
        %26 = arith.addi %23, %c1 : index
        %27 = memref.load %alloca_3[%26] : memref<2xf32, #gpu.address_space<workgroup>>
        %28 = arith.addf %27, %25 : f32
        memref.store %28, %arg1[] : memref<f32>
      }
      gpu.barrier
      gpu.return
    }
    gpu.func @Unknown171_kernel(%arg0: memref<4x1000xf16>, %arg1: memref<1000xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 2, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c-32 = arith.constant -32 : index
      %c1000 = arith.constant 1000 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f32
      %12 = gpu.block_id  x
      %13 = arith.muli %12, %c-32 : index
      %14 = arith.addi %13, %c1000 : index
      %15 = arith.cmpi slt, %14, %c32 : index
      %16 = arith.select %15, %14, %c32 : index
      %17 = arith.muli %12, %c32 : index
      %alloca = memref.alloca() : memref<32xf32, #gpu.address_space<workgroup>>
      %alloca_1 = memref.alloca() : memref<2x32xf32, #gpu.address_space<workgroup>>
      %18 = gpu.thread_id  x
      %19 = gpu.thread_id  y
      %20 = arith.cmpi slt, %16, %18 : index
      %21 = arith.select %20, %16, %18 : index
      %22 = arith.addi %18, %c1 : index
      %23 = arith.cmpi slt, %16, %22 : index
      %24 = arith.select %23, %16, %22 : index
      %25 = arith.subi %24, %21 : index
      %26 = arith.cmpi ugt, %25, %c0 : index
      %27 = scf.if %26 -> (f16) {
        %34 = arith.muli %19, %c2 : index
        %35 = arith.addi %17, %21 : index
        %36 = memref.load %arg0[%34, %35] : memref<4x1000xf16>
        scf.yield %36 : f16
      } else {
        scf.yield %cst : f16
      }
      %28 = arith.extf %27 : f16 to f32
      %29 = arith.addf %28, %cst_0 : f32
      %30 = scf.if %26 -> (f16) {
        %34 = arith.muli %19, %c2 : index
        %35 = arith.addi %34, %c1 : index
        %36 = arith.addi %17, %21 : index
        %37 = memref.load %arg0[%35, %36] : memref<4x1000xf16>
        scf.yield %37 : f16
      } else {
        scf.yield %cst : f16
      }
      %31 = arith.extf %30 : f16 to f32
      %32 = arith.addf %29, %31 : f32
      memref.store %32, %alloca_1[%19, %18] : memref<2x32xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %33 = arith.cmpi ult, %19, %c1 : index
      scf.if %33 {
        %34 = memref.load %alloca_1[%c0, %18] : memref<2x32xf32, #gpu.address_space<workgroup>>
        %35 = arith.addf %34, %cst_0 : f32
        %36 = memref.load %alloca_1[%c1, %18] : memref<2x32xf32, #gpu.address_space<workgroup>>
        %37 = arith.addf %36, %35 : f32
        memref.store %37, %alloca[%18] : memref<32xf32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %subview = memref.subview %alloca[0] [%16] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<?xf32, strided<[1]>, #gpu.address_space<workgroup>>
      %subview_2 = memref.subview %arg1[%17] [%16] [1] : memref<1000xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview, %subview_2 : memref<?xf32, strided<[1]>, #gpu.address_space<workgroup>> to memref<?xf32, strided<[1], offset: ?>>
      gpu.return
    }
  }
  func.func private @Unknown0(%arg0: memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 588 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c588 = arith.constant 588 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x3x224x224xf16>
    gpu.launch_func  @unified::@Unknown0 blocks in (%c588, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x3x224x224xf32>, %alloc : memref<4x3x224x224xf16>)
    return %alloc : memref<4x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 10 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown1", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf16>
    gpu.launch_func  @unified::@Unknown1 blocks in (%c10, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<64x3x7x7xf32>, %alloc : memref<64x3x7x7xf16>)
    return %alloc : memref<64x3x7x7xf16>
  }
  func.func private @Unknown3(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 36 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown3", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c36 = arith.constant 36 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf16>
    gpu.launch_func  @unified::@Unknown3 blocks in (%c36, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<64x64x3x3xf32>, %alloc : memref<64x64x3x3xf16>)
    return %alloc : memref<64x64x3x3xf16>
  }
  func.func private @Unknown7(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown7", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf16>
    gpu.launch_func  @unified::@Unknown7 blocks in (%c8, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<128x64x1x1xf32>, %alloc : memref<128x64x1x1xf16>)
    return %alloc : memref<128x64x1x1xf16>
  }
  func.func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 72 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown8", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c72 = arith.constant 72 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf16>
    gpu.launch_func  @unified::@Unknown8 blocks in (%c72, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<128x64x3x3xf32>, %alloc : memref<128x64x3x3xf16>)
    return %alloc : memref<128x64x3x3xf16>
  }
  func.func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 144 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown9", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c144 = arith.constant 144 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf16>
    gpu.launch_func  @unified::@Unknown9 blocks in (%c144, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<128x128x3x3xf32>, %alloc : memref<128x128x3x3xf16>)
    return %alloc : memref<128x128x3x3xf16>
  }
  func.func private @Unknown12(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown12", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf16>
    gpu.launch_func  @unified::@Unknown12 blocks in (%c32, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<256x128x1x1xf32>, %alloc : memref<256x128x1x1xf16>)
    return %alloc : memref<256x128x1x1xf16>
  }
  func.func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown13", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c288 = arith.constant 288 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf16>
    gpu.launch_func  @unified::@Unknown13 blocks in (%c288, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<256x128x3x3xf32>, %alloc : memref<256x128x3x3xf16>)
    return %alloc : memref<256x128x3x3xf16>
  }
  func.func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown14", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c576 = arith.constant 576 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf16>
    gpu.launch_func  @unified::@Unknown14 blocks in (%c576, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<256x256x3x3xf32>, %alloc : memref<256x256x3x3xf16>)
    return %alloc : memref<256x256x3x3xf16>
  }
  func.func private @Unknown17(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 128 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown17", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf16>
    gpu.launch_func  @unified::@Unknown17 blocks in (%c128, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<512x256x1x1xf32>, %alloc : memref<512x256x1x1xf16>)
    return %alloc : memref<512x256x1x1xf16>
  }
  func.func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown18", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf16>
    gpu.launch_func  @unified::@Unknown18 blocks in (%c1152, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<512x256x3x3xf32>, %alloc : memref<512x256x3x3xf16>)
    return %alloc : memref<512x256x3x3xf16>
  }
  func.func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown19", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c2304 = arith.constant 2304 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf16>
    gpu.launch_func  @unified::@Unknown19 blocks in (%c2304, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<512x512x3x3xf32>, %alloc : memref<512x512x3x3xf16>)
    return %alloc : memref<512x512x3x3xf16>
  }
  func.func private @Unknown22(%arg0: memref<4x1000xf32>) -> memref<4x1000xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown22", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    gpu.launch_func  @unified::@Unknown22 blocks in (%c4, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x1000xf32>, %alloc : memref<4x1000xf16>)
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown23(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 500 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown23", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c500 = arith.constant 500 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<1000x512xf16>
    gpu.launch_func  @unified::@Unknown23 blocks in (%c500, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<1000x512xf32>, %alloc : memref<1000x512xf16>)
    return %alloc : memref<1000x512xf16>
  }
  func.func private @Unknown24(%arg0: memref<1000xf32>) -> memref<1000xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown24", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<1000xf16>
    gpu.launch_func  @unified::@Unknown24 blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<1000xf32>, %alloc : memref<1000xf16>)
    return %alloc : memref<1000xf16>
  }
  func.func private @Unknown25(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c1000 = arith.constant 1000 : index
    %c-1024 = arith.constant -1024 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4xf16>
    gpu.launch_func  @unified::@Unknown25_kernel blocks in (%c4, %c1, %c1) threads in (%c512, %c1, %c1)  args(%arg0 : memref<4x1000xf16>, %alloc : memref<4xf16>)
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown26(%arg0: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown26", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c3136 = arith.constant 3136 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x64x112x112xi1>
    %alloc_0 = memref.alloc() : memref<4x64x112x112xf16>
    gpu.launch_func  @unified::@Unknown26 blocks in (%c3136, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x64x112x112xf16>, %alloc_0 : memref<4x64x112x112xf16>, %alloc : memref<4x64x112x112xi1>)
    return %alloc_0, %alloc : memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
  }
  func.func private @Unknown28(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown28", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x64x56x56xi1>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown28 blocks in (%c784, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x64x56x56xf16>, %alloc_0 : memref<4x64x56x56xf16>, %alloc : memref<4x64x56x56xi1>)
    return %alloc_0, %alloc : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown30(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown30", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x64x56x56xi1>
    %alloc_0 = memref.alloc() : memref<4x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown30 blocks in (%c784, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x64x56x56xf16>, %arg1 : memref<4x64x56x56xf16>, %alloc_0 : memref<4x64x56x56xf16>, %alloc : memref<4x64x56x56xi1>)
    return %alloc_0, %alloc : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func.func private @Unknown37(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown37", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c392 = arith.constant 392 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x128x28x28xi1>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf16>
    gpu.launch_func  @unified::@Unknown37 blocks in (%c392, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x128x28x28xf16>, %alloc_0 : memref<4x128x28x28xf16>, %alloc : memref<4x128x28x28xi1>)
    return %alloc_0, %alloc : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown39(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown39", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c392 = arith.constant 392 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x128x28x28xi1>
    %alloc_0 = memref.alloc() : memref<4x128x28x28xf16>
    gpu.launch_func  @unified::@Unknown39 blocks in (%c392, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x128x28x28xf16>, %arg1 : memref<4x128x28x28xf16>, %alloc_0 : memref<4x128x28x28xf16>, %alloc : memref<4x128x28x28xi1>)
    return %alloc_0, %alloc : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func.func private @Unknown46(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown46", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c196 = arith.constant 196 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xi1>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf16>
    gpu.launch_func  @unified::@Unknown46 blocks in (%c196, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x256x14x14xf16>, %alloc_0 : memref<4x256x14x14xf16>, %alloc : memref<4x256x14x14xi1>)
    return %alloc_0, %alloc : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown48(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown48", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c196 = arith.constant 196 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xi1>
    %alloc_0 = memref.alloc() : memref<4x256x14x14xf16>
    gpu.launch_func  @unified::@Unknown48 blocks in (%c196, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x256x14x14xf16>, %arg1 : memref<4x256x14x14xf16>, %alloc_0 : memref<4x256x14x14xf16>, %alloc : memref<4x256x14x14xi1>)
    return %alloc_0, %alloc : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func.func private @Unknown55(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown55", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c98 = arith.constant 98 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x512x7x7xi1>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown55 blocks in (%c98, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x512x7x7xf16>, %alloc_0 : memref<4x512x7x7xf16>, %alloc : memref<4x512x7x7xi1>)
    return %alloc_0, %alloc : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown57(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown57", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c98 = arith.constant 98 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x512x7x7xi1>
    %alloc_0 = memref.alloc() : memref<4x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown57 blocks in (%c98, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x512x7x7xf16>, %arg1 : memref<4x512x7x7xf16>, %alloc_0 : memref<4x512x7x7xf16>, %alloc : memref<4x512x7x7xi1>)
    return %alloc_0, %alloc : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func.func private @Unknown62(%arg0: memref<4x512x7x7xf16>) -> memref<4x512xf16> attributes {__byteir_reduction_fusion__} {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c2048 = arith.constant 2048 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c49 = arith.constant 49 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2, 3]] : memref<4x512x7x7xf16> into memref<2048x49xf16>
    %alloc = memref.alloc() : memref<2048xf16>
    gpu.launch_func  @unified::@Unknown62_kernel blocks in (%c2048, %c1, %c1) threads in (%c64, %c1, %c1)  args(%collapse_shape : memref<2048x49xf16>, %alloc : memref<2048xf16>)
    %expand_shape = memref.expand_shape %alloc [[0, 1]] : memref<2048xf16> into memref<4x512xf16>
    return %expand_shape : memref<4x512xf16>
  }
  func.func private @Unknown63(%arg0: memref<4x512xf16>) -> memref<4x512xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown63", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x512xf16>
    gpu.launch_func  @unified::@Unknown63 blocks in (%c2, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x512xf16>, %alloc : memref<4x512xf16>)
    return %alloc : memref<4x512xf16>
  }
  func.func private @Unknown64(%arg0: memref<1000xf16>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown64", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    gpu.launch_func  @unified::@Unknown64 blocks in (%c4, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<1000xf16>, %arg1 : memref<4x1000xf16>, %alloc : memref<4x1000xf16>)
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown65(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c1000 = arith.constant 1000 : index
    %c-1024 = arith.constant -1024 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4xf16>
    gpu.launch_func  @unified::@Unknown65_kernel blocks in (%c4, %c1, %c1) threads in (%c512, %c1, %c1)  args(%arg0 : memref<4x1000xf16>, %alloc : memref<4xf16>)
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown66(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown66", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    gpu.launch_func  @unified::@Unknown66 blocks in (%c4, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4xf16>, %arg1 : memref<4x1000xf16>, %alloc : memref<4x1000xf16>)
    return %alloc : memref<4x1000xf16>
  }
  func.func private @Unknown67(%arg0: memref<4x1000xf16>) -> memref<4xf16> attributes {__byteir_reduction_fusion__} {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c1000 = arith.constant 1000 : index
    %c-1024 = arith.constant -1024 : index
    %c-1 = arith.constant -1 : index
    %c512 = arith.constant 512 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4xf16>
    gpu.launch_func  @unified::@Unknown67_kernel blocks in (%c4, %c1, %c1) threads in (%c512, %c1, %c1)  args(%arg0 : memref<4x1000xf16>, %alloc : memref<4xf16>)
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown68(%arg0: memref<4xf16>) -> memref<4xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown68", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4xf16>
    gpu.launch_func  @unified::@Unknown68 blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4xf16>, %alloc : memref<4xf16>)
    return %alloc : memref<4xf16>
  }
  func.func private @Unknown69(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>) attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown69", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x1000xf16>
    %alloc_0 = memref.alloc() : memref<4x1000xf16>
    gpu.launch_func  @unified::@Unknown69 blocks in (%c4, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4xf16>, %arg1 : memref<4x1000xf16>, %arg2 : memref<4xf16>, %arg3 : memref<4x1000xf16>, %alloc_0 : memref<4x1000xf16>, %alloc : memref<4x1000xf16>)
    return %alloc_0, %alloc : memref<4x1000xf16>, memref<4x1000xf16>
  }
  func.func private @Unknown70(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [2 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown70", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c98 = arith.constant 98 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown70 blocks in (%c98, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x512xf16>, %arg1 : memref<4x512x7x7xi1>, %alloc : memref<4x512x7x7xf16>)
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown74(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown74", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c98 = arith.constant 98 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown74 blocks in (%c98, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x512x7x7xi1>, %arg1 : memref<4x512x7x7xf16>, %alloc : memref<4x512x7x7xf16>)
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown78(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown78", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c98 = arith.constant 98 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x512x7x7xf16>
    gpu.launch_func  @unified::@Unknown78 blocks in (%c98, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x512x7x7xf16>, %arg1 : memref<4x512x7x7xf16>, %arg2 : memref<4x512x7x7xi1>, %alloc : memref<4x512x7x7xf16>)
    return %alloc : memref<4x512x7x7xf16>
  }
  func.func private @Unknown89(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown89", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c196 = arith.constant 196 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    gpu.launch_func  @unified::@Unknown89 blocks in (%c196, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x256x14x14xf16>, %arg1 : memref<4x256x14x14xf16>, %arg2 : memref<4x256x14x14xi1>, %alloc : memref<4x256x14x14xf16>)
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown93(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown93", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c196 = arith.constant 196 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x256x14x14xf16>
    gpu.launch_func  @unified::@Unknown93 blocks in (%c196, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x256x14x14xi1>, %arg1 : memref<4x256x14x14xf16>, %alloc : memref<4x256x14x14xf16>)
    return %alloc : memref<4x256x14x14xf16>
  }
  func.func private @Unknown108(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown108", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c392 = arith.constant 392 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    gpu.launch_func  @unified::@Unknown108 blocks in (%c392, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x128x28x28xf16>, %arg1 : memref<4x128x28x28xf16>, %arg2 : memref<4x128x28x28xi1>, %alloc : memref<4x128x28x28xf16>)
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown112(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown112", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c392 = arith.constant 392 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x128x28x28xf16>
    gpu.launch_func  @unified::@Unknown112 blocks in (%c392, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x128x28x28xi1>, %arg1 : memref<4x128x28x28xf16>, %alloc : memref<4x128x28x28xf16>)
    return %alloc : memref<4x128x28x28xf16>
  }
  func.func private @Unknown127(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown127", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown127 blocks in (%c784, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x64x56x56xf16>, %arg1 : memref<4x64x56x56xf16>, %arg2 : memref<4x64x56x56xi1>, %alloc : memref<4x64x56x56xf16>)
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown131(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown131", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown131 blocks in (%c784, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x64x56x56xi1>, %arg1 : memref<4x64x56x56xf16>, %alloc : memref<4x64x56x56xf16>)
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown143(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown143", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x64x56x56xf16>
    gpu.launch_func  @unified::@Unknown143 blocks in (%c784, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x64x56x56xf16>, %arg1 : memref<4x64x56x56xf16>, %alloc : memref<4x64x56x56xf16>)
    return %alloc : memref<4x64x56x56xf16>
  }
  func.func private @Unknown144(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 3136 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown144", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c3136 = arith.constant 3136 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<4x64x112x112xf16>
    gpu.launch_func  @unified::@Unknown144 blocks in (%c3136, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4x64x112x112xi1>, %arg1 : memref<4x64x112x112xf16>, %alloc : memref<4x64x112x112xf16>)
    return %alloc : memref<4x64x112x112xf16>
  }
  func.func private @Unknown147(%arg0: memref<4x1000xf16>, %arg1: memref<4x1000xf32>) -> memref<f32> attributes {__byteir_reduction_fusion__} {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c125 = arith.constant 125 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<f32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<4x1000xf16> into memref<4000xf16>
    %collapse_shape_1 = memref.collapse_shape %arg1 [[0, 1]] : memref<4x1000xf32> into memref<4000xf32>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] : memref<4000xf16> into memref<32x125xf16>
    %expand_shape_2 = memref.expand_shape %collapse_shape_1 [[0, 1]] : memref<4000xf32> into memref<32x125xf32>
    %alloc_3 = memref.alloc() : memref<32xf32>
    gpu.launch_func  @unified::@Unknown147_kernel blocks in (%c32, %c1, %c1) threads in (%c128, %c1, %c1)  args(%expand_shape : memref<32x125xf16>, %expand_shape_2 : memref<32x125xf32>, %alloc_3 : memref<32xf32>)
    gpu.launch_func  @unified::@Unknown147_kernel_0 blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1)  args(%alloc_3 : memref<32xf32>, %alloc : memref<f32>)
    return %alloc : memref<f32>
  }
  func.func private @Unknown148(%arg0: memref<f32>) -> memref<f32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [0 : i32, 0 : i32], __byre__kernel_name = "Unknown148", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<f32>
    gpu.launch_func  @unified::@Unknown148 blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<f32>, %alloc : memref<f32>)
    return %alloc : memref<f32>
  }
  func.func private @Unknown149(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 10 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown149", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<64x3x7x7xf32>
    gpu.launch_func  @unified::@Unknown149 blocks in (%c10, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<64x3x7x7xf16>, %alloc : memref<64x3x7x7xf32>)
    return %alloc : memref<64x3x7x7xf32>
  }
  func.func private @Unknown150(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 36 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown150", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c36 = arith.constant 36 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @unified::@Unknown150 blocks in (%c36, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<64x64x3x3xf16>, %alloc : memref<64x64x3x3xf32>)
    return %alloc : memref<64x64x3x3xf32>
  }
  func.func private @Unknown154(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 72 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown154", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c72 = arith.constant 72 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<128x64x3x3xf32>
    gpu.launch_func  @unified::@Unknown154 blocks in (%c72, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<128x64x3x3xf16>, %alloc : memref<128x64x3x3xf32>)
    return %alloc : memref<128x64x3x3xf32>
  }
  func.func private @Unknown155(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 144 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown155", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c144 = arith.constant 144 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<128x128x3x3xf32>
    gpu.launch_func  @unified::@Unknown155 blocks in (%c144, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<128x128x3x3xf16>, %alloc : memref<128x128x3x3xf32>)
    return %alloc : memref<128x128x3x3xf32>
  }
  func.func private @Unknown156(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown156", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<128x64x1x1xf32>
    gpu.launch_func  @unified::@Unknown156 blocks in (%c8, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<128x64x1x1xf16>, %alloc : memref<128x64x1x1xf32>)
    return %alloc : memref<128x64x1x1xf32>
  }
  func.func private @Unknown159(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown159", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c288 = arith.constant 288 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x128x3x3xf32>
    gpu.launch_func  @unified::@Unknown159 blocks in (%c288, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<256x128x3x3xf16>, %alloc : memref<256x128x3x3xf32>)
    return %alloc : memref<256x128x3x3xf32>
  }
  func.func private @Unknown160(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown160", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c576 = arith.constant 576 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x256x3x3xf32>
    gpu.launch_func  @unified::@Unknown160 blocks in (%c576, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<256x256x3x3xf16>, %alloc : memref<256x256x3x3xf32>)
    return %alloc : memref<256x256x3x3xf32>
  }
  func.func private @Unknown161(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown161", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x128x1x1xf32>
    gpu.launch_func  @unified::@Unknown161 blocks in (%c32, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<256x128x1x1xf16>, %alloc : memref<256x128x1x1xf32>)
    return %alloc : memref<256x128x1x1xf32>
  }
  func.func private @Unknown164(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown164", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x256x3x3xf32>
    gpu.launch_func  @unified::@Unknown164 blocks in (%c1152, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<512x256x3x3xf16>, %alloc : memref<512x256x3x3xf32>)
    return %alloc : memref<512x256x3x3xf32>
  }
  func.func private @Unknown165(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown165", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c2304 = arith.constant 2304 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x512x3x3xf32>
    gpu.launch_func  @unified::@Unknown165 blocks in (%c2304, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<512x512x3x3xf16>, %alloc : memref<512x512x3x3xf32>)
    return %alloc : memref<512x512x3x3xf32>
  }
  func.func private @Unknown166(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 128 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown166", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<512x256x1x1xf32>
    gpu.launch_func  @unified::@Unknown166 blocks in (%c128, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<512x256x1x1xf16>, %alloc : memref<512x256x1x1xf32>)
    return %alloc : memref<512x256x1x1xf32>
  }
  func.func private @Unknown170(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 500 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown170", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c500 = arith.constant 500 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<1000x512xf32>
    gpu.launch_func  @unified::@Unknown170 blocks in (%c500, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<1000x512xf16>, %alloc : memref<1000x512xf32>)
    return %alloc : memref<1000x512xf32>
  }
  func.func private @Unknown171(%arg0: memref<4x1000xf16>) -> memref<1000xf32> attributes {__byteir_reduction_fusion__} {
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c1000 = arith.constant 1000 : index
    %c-32 = arith.constant -32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1000xf32>
    gpu.launch_func  @unified::@Unknown171_kernel blocks in (%c32, %c1, %c1) threads in (%c32, %c2, %c1)  args(%arg0 : memref<4x1000xf16>, %alloc : memref<1000xf32>)
    return %alloc : memref<1000xf32>
  }
  func.func private @Unknown172(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown172", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<1000xf32>
    gpu.launch_func  @unified::@Unknown172 blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<1000xf32>, %alloc : memref<1000xf32>)
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