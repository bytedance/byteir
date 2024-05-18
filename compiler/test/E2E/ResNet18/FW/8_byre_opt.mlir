// RUN: byteir-opt %s -byre-opt="append-arg-types entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown92(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %cst = arith.constant 1.000000e-01 : f32
      %cst_0 = arith.constant 0.899999976 : f32
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c512 step %6 {
        %7 = memref.load %arg1[%arg3] : memref<512xf32>
        %8 = memref.load %arg0[%arg3] : memref<512xf32>
        %9 = arith.mulf %7, %cst_0 : f32
        %10 = arith.mulf %8, %cst : f32
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg2[%arg3] : memref<512xf32>
      }
      gpu.return
    }
    gpu.func @Unknown82(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %cst = arith.constant 1.000000e-01 : f32
      %cst_0 = arith.constant 0.899999976 : f32
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c256 step %6 {
        %7 = memref.load %arg1[%arg3] : memref<256xf32>
        %8 = memref.load %arg0[%arg3] : memref<256xf32>
        %9 = arith.mulf %7, %cst_0 : f32
        %10 = arith.mulf %8, %cst : f32
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg2[%arg3] : memref<256xf32>
      }
      gpu.return
    }
    gpu.func @Unknown72(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %cst = arith.constant 1.000000e-01 : f32
      %cst_0 = arith.constant 0.899999976 : f32
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c128 step %6 {
        %7 = memref.load %arg1[%arg3] : memref<128xf32>
        %8 = memref.load %arg0[%arg3] : memref<128xf32>
        %9 = arith.mulf %7, %cst_0 : f32
        %10 = arith.mulf %8, %cst : f32
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg2[%arg3] : memref<128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown62(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %cst = arith.constant 1.000000e-01 : f32
      %cst_0 = arith.constant 0.899999976 : f32
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c64 step %6 {
        %7 = memref.load %arg1[%arg3] : memref<64xf32>
        %8 = memref.load %arg0[%arg3] : memref<64xf32>
        %9 = arith.mulf %7, %cst_0 : f32
        %10 = arith.mulf %8, %cst : f32
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg2[%arg3] : memref<64xf32>
      }
      gpu.return
    }
    gpu.func @Unknown61(%arg0: memref<1000xf32>, %arg1: memref<1x1000xf16>, %arg2: memref<1x1000xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c1000 step %6 {
        %7 = memref.load %arg0[%arg3] : memref<1000xf32>
        %8 = memref.load %arg1[%c0, %arg3] : memref<1x1000xf16>
        %9 = arith.truncf %7 : f32 to f16
        %10 = arith.addf %8, %9 : f16
        memref.store %10, %arg2[%c0, %arg3] : memref<1x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown60(%arg0: memref<1000x512xf32>, %arg1: memref<1000x512xf16>) kernel {
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
    gpu.func @Unknown59(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) kernel {
      %cst = arith.constant 2.040100e-02 : f16
      %c0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c512 step %6 {
        %7 = memref.load %arg0[%c0, %arg2] : memref<1x512xf16>
        %8 = arith.mulf %7, %cst : f16
        memref.store %8, %arg1[%c0, %arg2] : memref<1x512xf16>
      }
      gpu.return
    }
    gpu.func @Unknown51(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
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
        %13 = arith.addf %11, %12 : f16
        %14 = arith.maximumf %13, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown49(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
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
    gpu.func @Unknown48(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) kernel {
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
      scf.for %arg2 = %4 to %c25088 step %6 {
        %7 = arith.remsi %arg2, %c7 : index
        %8 = arith.divsi %arg2, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown46(%arg0: memref<512x256x3x3xf32>, %arg1: memref<512x256x3x3xf16>) kernel {
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
    gpu.func @Unknown44(%arg0: memref<512x256x1x1xf32>, %arg1: memref<512x256x1x1xf16>) kernel {
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
    gpu.func @Unknown37(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
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
        %13 = arith.addf %11, %12 : f16
        %14 = arith.maximumf %13, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown35(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
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
    gpu.func @Unknown34(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) kernel {
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
      scf.for %arg2 = %4 to %c50176 step %6 {
        %7 = arith.remsi %arg2, %c14 : index
        %8 = arith.divsi %arg2, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown32(%arg0: memref<256x128x3x3xf32>, %arg1: memref<256x128x3x3xf16>) kernel {
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
    gpu.func @Unknown30(%arg0: memref<256x128x1x1xf32>, %arg1: memref<256x128x1x1xf16>) kernel {
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
    gpu.func @Unknown23(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
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
        %13 = arith.addf %11, %12 : f16
        %14 = arith.maximumf %13, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown21(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
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
    gpu.func @Unknown20(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) kernel {
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
      scf.for %arg2 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg2, %c28 : index
        %8 = arith.divsi %arg2, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown18(%arg0: memref<128x64x3x3xf32>, %arg1: memref<128x64x3x3xf16>) kernel {
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
    gpu.func @Unknown16(%arg0: memref<128x64x1x1xf32>, %arg1: memref<128x64x1x1xf16>) kernel {
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
    gpu.func @Unknown9(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
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
        %13 = arith.addf %11, %12 : f16
        %14 = arith.maximumf %13, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown6(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) kernel {
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
      scf.for %arg2 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg2, %c56 : index
        %8 = arith.divsi %arg2, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
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
    gpu.func @Unknown3(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) kernel {
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
      scf.for %arg2 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg2, %c112 : index
        %8 = arith.divsi %arg2, %c112 : index
        %9 = arith.remsi %8, %c112 : index
        %10 = arith.divsi %8, %c112 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
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
    gpu.func @Unknown0(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x3x224x224xf16>) kernel {
      %c150528 = arith.constant 150528 : index
      %c0 = arith.constant 0 : index
      %c224 = arith.constant 224 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c150528 step %6 {
        %7 = arith.remsi %arg2, %c224 : index
        %8 = arith.divsi %arg2, %c224 : index
        %9 = arith.remsi %8, %c224 : index
        %10 = arith.divsi %8, %c224 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x3x224x224xf32>
        %12 = arith.truncf %11 : f32 to f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x3x224x224xf16>
      }
      gpu.return
    }
    gpu.func @Unknown58_kernel(%arg0: memref<512x49xf16>, %arg1: memref<512xf16>) kernel attributes {gpu.known_block_size = array<i32: 64, 1, 1>, gpu.known_grid_size = array<i32: 512, 1, 1>} {
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
      %subview = memref.subview %arg0[%0, 0] [1, 49] [1, 1] : memref<512x49xf16> to memref<49xf16, strided<[1], offset: ?>>
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
        memref.store %26, %arg1[%0] : memref<512xf16>
      }
      gpu.barrier
      gpu.return
    }
  }
  func.func private @Unknown0(memref<1x3x224x224xf32, "cuda">) -> memref<1x3x224x224xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 147 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown1(memref<64x3x7x7xf32, "cuda">) -> memref<64x3x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 10 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown1", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown3(memref<1x64x112x112xf16, "cuda">) -> memref<1x64x112x112xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown3", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown4(memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 36 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown4", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown6(memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown6", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown9(memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown9", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown16(memref<128x64x1x1xf32, "cuda">) -> memref<128x64x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown16", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown18(memref<128x64x3x3xf32, "cuda">) -> memref<128x64x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 72 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown18", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown20(memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown20", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown21(memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 144 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown21", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown23(memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 98 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown23", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown30(memref<256x128x1x1xf32, "cuda">) -> memref<256x128x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown30", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown32(memref<256x128x3x3xf32, "cuda">) -> memref<256x128x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown32", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown34(memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 49 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown34", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown35(memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown35", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown37(memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 49 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown37", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown44(memref<512x256x1x1xf32, "cuda">) -> memref<512x256x1x1xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 128 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown44", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown46(memref<512x256x3x3xf32, "cuda">) -> memref<512x256x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown46", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown48(memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 25 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown48", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown49(memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown49", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown51(memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 25 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown51", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown59(memref<1x512xf16, "cuda">) -> memref<1x512xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown59", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown60(memref<1000x512xf32, "cuda">) -> memref<1000x512xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 500 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown60", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown61(memref<1000xf32, "cuda">, memref<1x1000xf16, "cuda">) -> memref<1x1000xf16, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown61", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown62(memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown62", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown72(memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown72", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown82(memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown82", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func private @Unknown92(memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown92", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func @main(%arg0: memref<64xf32, "cuda">, %arg1: memref<64xf32, "cuda">, %arg2: memref<64x3x7x7xf32, "cuda">, %arg3: memref<1000xf32, "cuda">, %arg4: memref<1000x512xf32, "cuda">, %arg5: memref<64xf32, "cuda">, %arg6: memref<64xf32, "cuda">, %arg7: memref<64xf32, "cuda">, %arg8: memref<64xf32, "cuda">, %arg9: memref<64x64x3x3xf32, "cuda">, %arg10: memref<64x64x3x3xf32, "cuda">, %arg11: memref<64xf32, "cuda">, %arg12: memref<64xf32, "cuda">, %arg13: memref<64xf32, "cuda">, %arg14: memref<64xf32, "cuda">, %arg15: memref<64x64x3x3xf32, "cuda">, %arg16: memref<64x64x3x3xf32, "cuda">, %arg17: memref<128xf32, "cuda">, %arg18: memref<128xf32, "cuda">, %arg19: memref<128xf32, "cuda">, %arg20: memref<128xf32, "cuda">, %arg21: memref<128x64x3x3xf32, "cuda">, %arg22: memref<128x128x3x3xf32, "cuda">, %arg23: memref<128x64x1x1xf32, "cuda">, %arg24: memref<128xf32, "cuda">, %arg25: memref<128xf32, "cuda">, %arg26: memref<128xf32, "cuda">, %arg27: memref<128xf32, "cuda">, %arg28: memref<128xf32, "cuda">, %arg29: memref<128xf32, "cuda">, %arg30: memref<128x128x3x3xf32, "cuda">, %arg31: memref<128x128x3x3xf32, "cuda">, %arg32: memref<256xf32, "cuda">, %arg33: memref<256xf32, "cuda">, %arg34: memref<256xf32, "cuda">, %arg35: memref<256xf32, "cuda">, %arg36: memref<256x128x3x3xf32, "cuda">, %arg37: memref<256x256x3x3xf32, "cuda">, %arg38: memref<256x128x1x1xf32, "cuda">, %arg39: memref<256xf32, "cuda">, %arg40: memref<256xf32, "cuda">, %arg41: memref<256xf32, "cuda">, %arg42: memref<256xf32, "cuda">, %arg43: memref<256xf32, "cuda">, %arg44: memref<256xf32, "cuda">, %arg45: memref<256x256x3x3xf32, "cuda">, %arg46: memref<256x256x3x3xf32, "cuda">, %arg47: memref<512xf32, "cuda">, %arg48: memref<512xf32, "cuda">, %arg49: memref<512xf32, "cuda">, %arg50: memref<512xf32, "cuda">, %arg51: memref<512x256x3x3xf32, "cuda">, %arg52: memref<512x512x3x3xf32, "cuda">, %arg53: memref<512x256x1x1xf32, "cuda">, %arg54: memref<512xf32, "cuda">, %arg55: memref<512xf32, "cuda">, %arg56: memref<512xf32, "cuda">, %arg57: memref<512xf32, "cuda">, %arg58: memref<512xf32, "cuda">, %arg59: memref<512xf32, "cuda">, %arg60: memref<512x512x3x3xf32, "cuda">, %arg61: memref<512x512x3x3xf32, "cuda">, %arg62: memref<i64, "cuda">, %arg63: memref<64xf32, "cuda">, %arg64: memref<64xf32, "cuda">, %arg65: memref<i64, "cuda">, %arg66: memref<64xf32, "cuda">, %arg67: memref<64xf32, "cuda">, %arg68: memref<i64, "cuda">, %arg69: memref<64xf32, "cuda">, %arg70: memref<64xf32, "cuda">, %arg71: memref<i64, "cuda">, %arg72: memref<64xf32, "cuda">, %arg73: memref<64xf32, "cuda">, %arg74: memref<i64, "cuda">, %arg75: memref<64xf32, "cuda">, %arg76: memref<64xf32, "cuda">, %arg77: memref<i64, "cuda">, %arg78: memref<128xf32, "cuda">, %arg79: memref<128xf32, "cuda">, %arg80: memref<i64, "cuda">, %arg81: memref<128xf32, "cuda">, %arg82: memref<128xf32, "cuda">, %arg83: memref<i64, "cuda">, %arg84: memref<128xf32, "cuda">, %arg85: memref<128xf32, "cuda">, %arg86: memref<i64, "cuda">, %arg87: memref<128xf32, "cuda">, %arg88: memref<128xf32, "cuda">, %arg89: memref<i64, "cuda">, %arg90: memref<128xf32, "cuda">, %arg91: memref<128xf32, "cuda">, %arg92: memref<i64, "cuda">, %arg93: memref<256xf32, "cuda">, %arg94: memref<256xf32, "cuda">, %arg95: memref<i64, "cuda">, %arg96: memref<256xf32, "cuda">, %arg97: memref<256xf32, "cuda">, %arg98: memref<i64, "cuda">, %arg99: memref<256xf32, "cuda">, %arg100: memref<256xf32, "cuda">, %arg101: memref<i64, "cuda">, %arg102: memref<256xf32, "cuda">, %arg103: memref<256xf32, "cuda">, %arg104: memref<i64, "cuda">, %arg105: memref<256xf32, "cuda">, %arg106: memref<256xf32, "cuda">, %arg107: memref<i64, "cuda">, %arg108: memref<512xf32, "cuda">, %arg109: memref<512xf32, "cuda">, %arg110: memref<i64, "cuda">, %arg111: memref<512xf32, "cuda">, %arg112: memref<512xf32, "cuda">, %arg113: memref<i64, "cuda">, %arg114: memref<512xf32, "cuda">, %arg115: memref<512xf32, "cuda">, %arg116: memref<i64, "cuda">, %arg117: memref<512xf32, "cuda">, %arg118: memref<512xf32, "cuda">, %arg119: memref<i64, "cuda">, %arg120: memref<512xf32, "cuda">, %arg121: memref<512xf32, "cuda">, %arg122: memref<1x3x224x224xf32, "cuda">) -> (memref<1x1000xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<64x3x7x7xf16, "cuda">, memref<1x3x224x224xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512xf16, "cuda">, memref<512x1000xf16, "cuda">) attributes {__placeholder__byre.entry_point} {
    %0 = call @Unknown0(%arg122) : (memref<1x3x224x224xf32, "cuda">) -> memref<1x3x224x224xf16, "cuda">
    %1 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32, "cuda">) -> memref<64x3x7x7xf16, "cuda">
    %alloc = memref.alloc() : memref<1x64x112x112xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%0, %1, %alloc) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x3x224x224xf16, "cuda">, memref<64x3x7x7xf16, "cuda">, memref<1x64x112x112xf16, "cuda">
    %alloc_0 = memref.alloc() : memref<1x64x112x112xf16, "cuda">
    %alloc_1 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_2 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc, %arg1, %arg0, %alloc_0, %alloc_1, %alloc_2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %2 = call @Unknown3(%alloc_0) : (memref<1x64x112x112xf16, "cuda">) -> memref<1x64x112x112xf16, "cuda">
    %alloc_3 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @PoolMaxOp_f16_f16(%2, %alloc_3) {base_dilations = dense<1> : tensor<4xi64>, device = "cuda", memory_effects = [1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<1x64x112x112xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %3 = call @Unknown4(%arg9) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %alloc_4 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%alloc_3, %3, %alloc_4) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_5 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    %alloc_6 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_7 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_4, %arg6, %arg5, %alloc_5, %alloc_6, %alloc_7) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %4 = call @Unknown6(%alloc_5) : (memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %5 = call @Unknown4(%arg10) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %alloc_8 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%4, %5, %alloc_8) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_9 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    %alloc_10 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_11 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_8, %arg8, %arg7, %alloc_9, %alloc_10, %alloc_11) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %6 = call @Unknown9(%alloc_9, %alloc_3) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %7 = call @Unknown4(%arg15) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %alloc_12 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%6, %7, %alloc_12) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_13 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    %alloc_14 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_15 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_12, %arg12, %arg11, %alloc_13, %alloc_14, %alloc_15) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %8 = call @Unknown6(%alloc_13) : (memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %9 = call @Unknown4(%arg16) : (memref<64x64x3x3xf32, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    %alloc_16 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%8, %9, %alloc_16) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %alloc_17 = memref.alloc() : memref<1x64x56x56xf16, "cuda">
    %alloc_18 = memref.alloc() : memref<64xf32, "cuda">
    %alloc_19 = memref.alloc() : memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_16, %arg14, %arg13, %alloc_17, %alloc_18, %alloc_19) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %10 = call @Unknown9(%alloc_17, %6) : (memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %11 = call @Unknown16(%arg23) : (memref<128x64x1x1xf32, "cuda">) -> memref<128x64x1x1xf16, "cuda">
    %alloc_20 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%10, %11, %alloc_20) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_21 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_22 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_23 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_20, %arg25, %arg24, %alloc_21, %alloc_22, %alloc_23) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %12 = call @Unknown18(%arg21) : (memref<128x64x3x3xf32, "cuda">) -> memref<128x64x3x3xf16, "cuda">
    %alloc_24 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%10, %12, %alloc_24) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_25 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_26 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_27 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_24, %arg18, %arg17, %alloc_25, %alloc_26, %alloc_27) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %13 = call @Unknown20(%alloc_25) : (memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %14 = call @Unknown21(%arg22) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %alloc_28 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%13, %14, %alloc_28) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_29 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_30 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_31 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_28, %arg20, %arg19, %alloc_29, %alloc_30, %alloc_31) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %15 = call @Unknown23(%alloc_29, %alloc_21) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %16 = call @Unknown21(%arg30) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %alloc_32 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%15, %16, %alloc_32) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_33 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_34 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_35 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_32, %arg27, %arg26, %alloc_33, %alloc_34, %alloc_35) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %17 = call @Unknown20(%alloc_33) : (memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %18 = call @Unknown21(%arg31) : (memref<128x128x3x3xf32, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    %alloc_36 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%17, %18, %alloc_36) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %alloc_37 = memref.alloc() : memref<1x128x28x28xf16, "cuda">
    %alloc_38 = memref.alloc() : memref<128xf32, "cuda">
    %alloc_39 = memref.alloc() : memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_36, %arg29, %arg28, %alloc_37, %alloc_38, %alloc_39) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %19 = call @Unknown23(%alloc_37, %15) : (memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %20 = call @Unknown30(%arg38) : (memref<256x128x1x1xf32, "cuda">) -> memref<256x128x1x1xf16, "cuda">
    %alloc_40 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%19, %20, %alloc_40) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_41 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_42 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_43 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_40, %arg40, %arg39, %alloc_41, %alloc_42, %alloc_43) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %21 = call @Unknown32(%arg36) : (memref<256x128x3x3xf32, "cuda">) -> memref<256x128x3x3xf16, "cuda">
    %alloc_44 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%19, %21, %alloc_44) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_45 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_46 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_47 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_44, %arg33, %arg32, %alloc_45, %alloc_46, %alloc_47) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %22 = call @Unknown34(%alloc_45) : (memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %23 = call @Unknown35(%arg37) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %alloc_48 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%22, %23, %alloc_48) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_49 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_50 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_51 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_48, %arg35, %arg34, %alloc_49, %alloc_50, %alloc_51) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %24 = call @Unknown37(%alloc_49, %alloc_41) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %25 = call @Unknown35(%arg45) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %alloc_52 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%24, %25, %alloc_52) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_53 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_54 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_55 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_52, %arg42, %arg41, %alloc_53, %alloc_54, %alloc_55) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %26 = call @Unknown34(%alloc_53) : (memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %27 = call @Unknown35(%arg46) : (memref<256x256x3x3xf32, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    %alloc_56 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%26, %27, %alloc_56) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %alloc_57 = memref.alloc() : memref<1x256x14x14xf16, "cuda">
    %alloc_58 = memref.alloc() : memref<256xf32, "cuda">
    %alloc_59 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_56, %arg44, %arg43, %alloc_57, %alloc_58, %alloc_59) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %28 = call @Unknown37(%alloc_57, %24) : (memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %29 = call @Unknown44(%arg53) : (memref<512x256x1x1xf32, "cuda">) -> memref<512x256x1x1xf16, "cuda">
    %alloc_60 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%28, %29, %alloc_60) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %alloc_61 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_62 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_63 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_60, %arg55, %arg54, %alloc_61, %alloc_62, %alloc_63) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %30 = call @Unknown46(%arg51) : (memref<512x256x3x3xf32, "cuda">) -> memref<512x256x3x3xf16, "cuda">
    %alloc_64 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%28, %30, %alloc_64) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %alloc_65 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_66 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_67 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_64, %arg48, %arg47, %alloc_65, %alloc_66, %alloc_67) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %31 = call @Unknown48(%alloc_65) : (memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %32 = call @Unknown49(%arg52) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %alloc_68 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%31, %32, %alloc_68) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %alloc_69 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_70 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_71 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_68, %arg50, %arg49, %alloc_69, %alloc_70, %alloc_71) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %33 = call @Unknown51(%alloc_69, %alloc_61) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %34 = call @Unknown49(%arg60) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %alloc_72 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%33, %34, %alloc_72) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %alloc_73 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_74 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_75 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_72, %arg57, %arg56, %alloc_73, %alloc_74, %alloc_75) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %35 = call @Unknown48(%alloc_73) : (memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %36 = call @Unknown49(%arg61) : (memref<512x512x3x3xf32, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    %alloc_76 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%35, %36, %alloc_76) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %alloc_77 = memref.alloc() : memref<1x512x7x7xf16, "cuda">
    %alloc_78 = memref.alloc() : memref<512xf32, "cuda">
    %alloc_79 = memref.alloc() : memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%alloc_76, %arg59, %arg58, %alloc_77, %alloc_78, %alloc_79) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %37 = call @Unknown51(%alloc_77, %33) : (memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %collapse_shape = memref.collapse_shape %37 [[0, 1], [2, 3]] : memref<1x512x7x7xf16, "cuda"> into memref<512x49xf16, "cuda">
    %alloc_80 = memref.alloc() : memref<512xf16, "cuda">
    byre.compute @PTXOp(%collapse_shape, %alloc_80) {BlockSize.x = 64 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 512 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown58_kernel"} : memref<512x49xf16, "cuda">, memref<512xf16, "cuda">
    %expand_shape = memref.expand_shape %alloc_80 [[0, 1]] : memref<512xf16, "cuda"> into memref<1x512xf16, "cuda">
    %38 = call @Unknown59(%expand_shape) : (memref<1x512xf16, "cuda">) -> memref<1x512xf16, "cuda">
    %39 = call @Unknown60(%arg4) : (memref<1000x512xf32, "cuda">) -> memref<1000x512xf16, "cuda">
    %alloc_81 = memref.alloc() : memref<512x1000xf16, "cuda">
    byre.compute @TransposeOp_f16_f16(%39, %alloc_81) {device = "cuda", memory_effects = [1 : i32, 2 : i32], minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : memref<1000x512xf16, "cuda">, memref<512x1000xf16, "cuda">
    %alloc_82 = memref.alloc() : memref<1x1000xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%38, %39, %alloc_82) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<1x512xf16, "cuda">, memref<1000x512xf16, "cuda">, memref<1x1000xf16, "cuda">
    %40 = call @Unknown61(%arg3, %alloc_82) : (memref<1000xf32, "cuda">, memref<1x1000xf16, "cuda">) -> memref<1x1000xf16, "cuda">
    %41 = call @Unknown62(%alloc_1, %arg63) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %42 = call @Unknown62(%alloc_2, %arg64) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %43 = call @Unknown62(%alloc_6, %arg66) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %44 = call @Unknown62(%alloc_7, %arg67) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %45 = call @Unknown62(%alloc_10, %arg69) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %46 = call @Unknown62(%alloc_11, %arg70) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %47 = call @Unknown62(%alloc_14, %arg72) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %48 = call @Unknown62(%alloc_15, %arg73) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %49 = call @Unknown62(%alloc_18, %arg75) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %50 = call @Unknown62(%alloc_19, %arg76) : (memref<64xf32, "cuda">, memref<64xf32, "cuda">) -> memref<64xf32, "cuda">
    %51 = call @Unknown72(%alloc_26, %arg78) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %52 = call @Unknown72(%alloc_27, %arg79) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %53 = call @Unknown72(%alloc_30, %arg81) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %54 = call @Unknown72(%alloc_31, %arg82) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %55 = call @Unknown72(%alloc_22, %arg84) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %56 = call @Unknown72(%alloc_23, %arg85) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %57 = call @Unknown72(%alloc_34, %arg87) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %58 = call @Unknown72(%alloc_35, %arg88) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %59 = call @Unknown72(%alloc_38, %arg90) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %60 = call @Unknown72(%alloc_39, %arg91) : (memref<128xf32, "cuda">, memref<128xf32, "cuda">) -> memref<128xf32, "cuda">
    %61 = call @Unknown82(%alloc_46, %arg93) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %62 = call @Unknown82(%alloc_47, %arg94) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %63 = call @Unknown82(%alloc_50, %arg96) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %64 = call @Unknown82(%alloc_51, %arg97) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %65 = call @Unknown82(%alloc_42, %arg99) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %66 = call @Unknown82(%alloc_43, %arg100) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %67 = call @Unknown82(%alloc_54, %arg102) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %68 = call @Unknown82(%alloc_55, %arg103) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %69 = call @Unknown82(%alloc_58, %arg105) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %70 = call @Unknown82(%alloc_59, %arg106) : (memref<256xf32, "cuda">, memref<256xf32, "cuda">) -> memref<256xf32, "cuda">
    %71 = call @Unknown92(%alloc_66, %arg108) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %72 = call @Unknown92(%alloc_67, %arg109) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %73 = call @Unknown92(%alloc_70, %arg111) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %74 = call @Unknown92(%alloc_71, %arg112) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %75 = call @Unknown92(%alloc_62, %arg114) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %76 = call @Unknown92(%alloc_63, %arg115) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %77 = call @Unknown92(%alloc_74, %arg117) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %78 = call @Unknown92(%alloc_75, %arg118) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %79 = call @Unknown92(%alloc_78, %arg120) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    %80 = call @Unknown92(%alloc_79, %arg121) : (memref<512xf32, "cuda">, memref<512xf32, "cuda">) -> memref<512xf32, "cuda">
    return %40, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %1, %0, %alloc, %2, %alloc_3, %3, %alloc_4, %4, %5, %alloc_8, %6, %7, %alloc_12, %8, %9, %alloc_16, %10, %12, %alloc_24, %13, %14, %alloc_28, %11, %alloc_20, %15, %16, %alloc_32, %17, %18, %alloc_36, %19, %21, %alloc_44, %22, %23, %alloc_48, %20, %alloc_40, %24, %25, %alloc_52, %26, %27, %alloc_56, %28, %30, %alloc_64, %31, %32, %alloc_68, %29, %alloc_60, %33, %34, %alloc_72, %35, %36, %alloc_76, %37, %38, %alloc_81 : memref<1x1000xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<64x3x7x7xf16, "cuda">, memref<1x3x224x224xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512xf16, "cuda">, memref<512x1000xf16, "cuda">
  }
}