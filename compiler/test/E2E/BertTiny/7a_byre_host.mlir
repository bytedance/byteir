// RUN: byteir-opt %s -byre-host="device-file-name=your_file target=cuda" | FileCheck %s

// CHECK-LABEL: func.func @main
module attributes {byre.container_module, gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown19(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c16384 : index
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
        %16 = memref.load %arg0[%15] : memref<128xi1>
        %17 = memref.load %arg1[%15, %9] : memref<128x128xf32>
        %18 = arith.select %16, %17, %cst : f32
        memref.store %18, %arg2[%15, %9] : memref<128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown18(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
      %cst = arith.constant 0.000000e+00 : f32
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
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.muli %25, %c128 : index
        %27 = arith.addi %26, %19 : index
        %28 = memref.load %arg2[%27] : memref<256xi1>
        %29 = memref.load %arg0[%27] : memref<256xi1>
        %30 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %31 = arith.select %29, %30, %cst : f32
        %32 = arith.select %28, %30, %cst : f32
        memref.store %31, %arg3[%25, %19, %9] : memref<2x128x128xf32>
        memref.store %32, %arg4[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown17(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
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
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = memref.load %arg2[%25, %19, %9] : memref<2x128x128xf32>
        %29 = memref.load %arg3[%25, %19, %9] : memref<2x128x128xf32>
        %30 = arith.addf %26, %27 : f32
        %31 = arith.addf %30, %28 : f32
        %32 = arith.addf %31, %29 : f32
        memref.store %32, %arg4[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown16(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>) kernel {
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
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = arith.addf %26, %27 : f32
        memref.store %28, %arg2[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown15(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
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
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = memref.load %arg2[%25, %19, %9] : memref<2x128x128xf32>
        %29 = memref.load %arg3[%25, %19, %9] : memref<2x128x128xf32>
        %30 = arith.addf %26, %27 : f32
        %31 = arith.addf %30, %28 : f32
        %32 = arith.addf %31, %29 : f32
        memref.store %32, %arg4[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown14(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>) kernel {
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
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = arith.addf %26, %27 : f32
        memref.store %28, %arg2[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>, %arg3: memref<2x128x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.muli %25, %c128 : index
        %27 = arith.addi %26, %19 : index
        %28 = memref.load %arg2[%27, %9] : memref<256x30522xf32>
        %29 = memref.load %arg1[%27, %9] : memref<256x30522xf32>
        %30 = memref.load %arg0[%27] : memref<256xf32>
        %31 = arith.mulf %29, %30 : f32
        %32 = arith.subf %28, %31 : f32
        memref.store %32, %arg3[%25, %19, %9] : memref<2x128x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown11(%arg0: memref<f32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg1[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg0[] : memref<f32>
        %18 = arith.divf %16, %17 : f32
        memref.store %18, %arg2[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown10(%arg0: memref<f32>, %arg1: memref<f32>) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1 : index
      scf.if %5 {
        %6 = memref.load %arg0[] : memref<f32>
        %7 = arith.cmpf une, %6, %cst : f32
        %8 = arith.select %7, %6, %cst_0 : f32
        memref.store %8, %arg1[] : memref<f32>
      }
      gpu.return
    }
    gpu.func @Unknown9(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) kernel {
      %c1 = arith.constant 1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1 : index
      scf.if %5 {
        %6 = memref.load %arg0[] : memref<f32>
        %7 = memref.load %arg1[] : memref<f32>
        %8 = arith.divf %6, %7 : f32
        memref.store %8, %arg2[] : memref<f32>
      }
      gpu.return
    }
    gpu.func @Unknown8(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>, %arg4: memref<256x30522xf32>, %arg5: memref<256x30522xf32>, %arg6: memref<256x30522xf32>, %arg7: memref<256x30522xf32>) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg3[%15] : memref<256xi1>
        %17 = memref.load %arg2[%15] : memref<256xi64>
        %18 = memref.load %arg1[%15, %9] : memref<256x30522xf32>
        %19 = memref.load %arg0[%15] : memref<256xf32>
        %20 = arith.index_cast %9 : index to i64
        %21 = arith.cmpi eq, %17, %20 : i64
        %22 = arith.select %21, %cst_0, %cst : f32
        %23 = arith.select %16, %cst_0, %cst : f32
        %24 = arith.mulf %23, %22 : f32
        %25 = arith.subf %18, %19 : f32
        %26 = arith.negf %22 : f32
        %27 = arith.mulf %26, %25 : f32
        %28 = arith.cmpf une, %22, %cst_0 : f32
        %29 = arith.select %28, %cst, %27 : f32
        %30 = arith.mulf %29, %24 : f32
        %31 = arith.mulf %26, %24 : f32
        %32 = math.exp %25 : f32
        memref.store %24, %arg4[%15, %9] : memref<256x30522xf32>
        memref.store %30, %arg5[%15, %9] : memref<256x30522xf32>
        memref.store %31, %arg6[%15, %9] : memref<256x30522xf32>
        memref.store %32, %arg7[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown7(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = math.log %6 : f32
        memref.store %7, %arg1[%4] : memref<256xf32>
      }
      gpu.return
    }
    gpu.func @Unknown6(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>, %arg3: memref<256x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg1[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg0[%15] : memref<256xf32>
        %18 = arith.subf %16, %17 : f32
        %19 = math.exp %18 : f32
        memref.store %18, %arg2[%15, %9] : memref<256x30522xf32>
        memref.store %19, %arg3[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>, %arg2: memref<2x128x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.muli %25, %c128 : index
        %27 = arith.addi %26, %19 : index
        %28 = memref.load %arg0[%27, %9] : memref<256x30522xf32>
        %29 = memref.load %arg1[%9] : memref<30522xf32>
        %30 = arith.addf %28, %29 : f32
        memref.store %30, %arg2[%25, %19, %9] : memref<2x128x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>, %arg3: memref<2x128x128xf32>) kernel {
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
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = arith.muli %25, %c128 : index
        %27 = arith.addi %26, %19 : index
        %28 = memref.load %arg0[%27, %9] : memref<256x128xf32>
        %29 = memref.load %arg1[%27, %9] : memref<256x128xf32>
        %30 = memref.load %arg2[%19, %9] : memref<128x128xf32>
        %31 = arith.addf %28, %29 : f32
        %32 = arith.addf %31, %30 : f32
        memref.store %32, %arg3[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown3(%arg0: memref<1x128xi64>, %arg1: memref<1x128xui32>, %arg2: memref<1x128xi64>, %arg3: memref<1x128xi1>) kernel {
      %c0_i64 = arith.constant 0 : i64
      %c512_i64 = arith.constant 512 : i64
      %c-1_i64 = arith.constant -1 : i64
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = arith.cmpi slt, %4, %c0 : index
        %7 = arith.addi %4, %c128 : index
        %8 = arith.select %6, %7, %4 : index
        %9 = memref.load %arg0[%c0, %8] : memref<1x128xi64>
        %10 = arith.trunci %9 : i64 to i32
        %11 = builtin.unrealized_conversion_cast %10 : i32 to ui32
        %12 = arith.addi %9, %c512_i64 : i64
        %13 = arith.cmpi slt, %9, %c0_i64 : i64
        %14 = arith.select %13, %12, %9 : i64
        %15 = arith.cmpi ne, %9, %c-1_i64 : i64
        memref.store %11, %arg1[%c0, %8] : memref<1x128xui32>
        memref.store %14, %arg2[%c0, %8] : memref<1x128xi64>
        memref.store %15, %arg3[%c0, %8] : memref<1x128xi1>
      }
      gpu.return
    }
    gpu.func @Unknown2(%arg0: memref<128xi64>, %arg1: memref<2x128xui32>, %arg2: memref<2x128xi64>, %arg3: memref<2x128xi1>) kernel {
      %c0_i64 = arith.constant 0 : i64
      %c2_i64 = arith.constant 2 : i64
      %c-1_i64 = arith.constant -1 : i64
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c256 : index
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
        %16 = memref.load %arg0[%9] : memref<128xi64>
        %17 = arith.addi %16, %c2_i64 : i64
        %18 = arith.trunci %16 : i64 to i32
        %19 = builtin.unrealized_conversion_cast %18 : i32 to ui32
        %20 = arith.cmpi slt, %16, %c0_i64 : i64
        %21 = arith.select %20, %17, %16 : i64
        %22 = arith.cmpi ne, %16, %c-1_i64 : i64
        memref.store %19, %arg1[%15, %9] : memref<2x128xui32>
        memref.store %21, %arg2[%15, %9] : memref<2x128xi64>
        memref.store %22, %arg3[%15, %9] : memref<2x128xi1>
      }
      gpu.return
    }
    gpu.func @Unknown1(%arg0: memref<2x128xi64>, %arg1: memref<2x128xui32>, %arg2: memref<2x128xi64>, %arg3: memref<2x128xi1>) kernel {
      %c0_i64 = arith.constant 0 : i64
      %c30522_i64 = arith.constant 30522 : i64
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c256 : index
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
        %16 = memref.load %arg0[%15, %9] : memref<2x128xi64>
        %17 = arith.trunci %16 : i64 to i32
        %18 = builtin.unrealized_conversion_cast %17 : i32 to ui32
        %19 = arith.addi %16, %c30522_i64 : i64
        %20 = arith.cmpi slt, %16, %c0_i64 : i64
        %21 = arith.select %20, %19, %16 : i64
        %22 = arith.cmpi ne, %16, %c0_i64 : i64
        memref.store %18, %arg1[%15, %9] : memref<2x128xui32>
        memref.store %21, %arg2[%15, %9] : memref<2x128xi64>
        memref.store %22, %arg3[%15, %9] : memref<2x128xi1>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<2x128xi64>, %arg1: memref<2x128xi1>) kernel {
      %c-100_i64 = arith.constant -100 : i64
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c256 : index
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
        %16 = memref.load %arg0[%15, %9] : memref<2x128xi64>
        %17 = arith.cmpi ne, %16, %c-100_i64 : i64
        memref.store %17, %arg1[%15, %9] : memref<2x128xi1>
      }
      gpu.return
    }
  }
  func.func @main(%arg0: memref<2x128xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<2x128xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x512xi64> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<30522x128xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<2x128xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<512x128xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128x128xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128x128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128x128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512x128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128x128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<30522xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<2x128x30522xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg47: memref<f32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg48: memref<30522x128xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg49: memref<2x128xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg50: memref<512x128xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg51: memref<128xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg52: memref<128xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg53: memref<128x128xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg54: memref<128xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg55: memref<128x128xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg56: memref<128xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg57: memref<128x128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg58: memref<128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg59: memref<128x128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg60: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg61: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg62: memref<128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg63: memref<512x128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg64: memref<512xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg65: memref<128x512xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg66: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg67: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg68: memref<128xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg69: memref<128x128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg70: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg71: memref<128x128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg72: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg73: memref<128x128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg74: memref<128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg75: memref<128x128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg76: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg77: memref<128xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg78: memref<128xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg79: memref<512x128xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg80: memref<512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg81: memref<128x512xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg82: memref<128xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg83: memref<128xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg84: memref<128xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg85: memref<128x128xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg86: memref<128xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg87: memref<128xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg88: memref<128xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg89: memref<30522xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %alloc = memref.alloc() : memref<0xi8>
    %alloc_0 = memref.alloc() : memref<0xi8>
    %alloc_1 = memref.alloc() : memref<0xi8>
    %alloc_2 = memref.alloc() : memref<16xi8>
    %alloc_3 = memref.alloc() : memref<32xi8>
    %alloc_4 = memref.alloc() : memref<32xi8>
    %alloc_5 = memref.alloc() : memref<32xi8>
    %alloc_6 = memref.alloc() : memref<1024xi8>
    %alloc_7 = memref.alloc() : memref<1024xi8>
    %alloc_8 = memref.alloc() : memref<1024xi8>
    %alloc_9 = memref.alloc() : memref<1024xi8>
    %alloc_10 = memref.alloc() : memref<1024xi8>
    %alloc_11 = memref.alloc() : memref<1024xi8>
    %alloc_12 = memref.alloc() : memref<1024xi8>
    %alloc_13 = memref.alloc() : memref<1024xi8>
    %alloc_14 = memref.alloc() : memref<1024xi8>
    %alloc_15 = memref.alloc() : memref<1024xi8>
    %alloc_16 = memref.alloc() : memref<1024xi8>
    %alloc_17 = memref.alloc() : memref<1024xi8>
    %alloc_18 = memref.alloc() : memref<1024xi8>
    %alloc_19 = memref.alloc() : memref<1024xi8>
    %alloc_20 = memref.alloc() : memref<2048xi8>
    %alloc_21 = memref.alloc() : memref<2048xi8>
    %alloc_22 = memref.alloc() : memref<65536xi8>
    %alloc_23 = memref.alloc() : memref<65536xi8>
    %alloc_24 = memref.alloc() : memref<131072xi8>
    %alloc_25 = memref.alloc() : memref<131072xi8>
    %alloc_26 = memref.alloc() : memref<131072xi8>
    %alloc_27 = memref.alloc() : memref<131072xi8>
    %alloc_28 = memref.alloc() : memref<131072xi8>
    %alloc_29 = memref.alloc() : memref<131072xi8>
    %alloc_30 = memref.alloc() : memref<131072xi8>
    %alloc_31 = memref.alloc() : memref<131072xi8>
    %alloc_32 = memref.alloc() : memref<131072xi8>
    %alloc_33 = memref.alloc() : memref<131072xi8>
    %alloc_34 = memref.alloc() : memref<131072xi8>
    %alloc_35 = memref.alloc() : memref<131072xi8>
    %alloc_36 = memref.alloc() : memref<131072xi8>
    %alloc_37 = memref.alloc() : memref<131072xi8>
    %alloc_38 = memref.alloc() : memref<131072xi8>
    %alloc_39 = memref.alloc() : memref<131072xi8>
    %alloc_40 = memref.alloc() : memref<131072xi8>
    %alloc_41 = memref.alloc() : memref<131072xi8>
    %alloc_42 = memref.alloc() : memref<131072xi8>
    %alloc_43 = memref.alloc() : memref<131072xi8>
    %alloc_44 = memref.alloc() : memref<131072xi8>
    %alloc_45 = memref.alloc() : memref<262144xi8>
    %alloc_46 = memref.alloc() : memref<262144xi8>
    %alloc_47 = memref.alloc() : memref<524288xi8>
    %alloc_48 = memref.alloc() : memref<524288xi8>
    %alloc_49 = memref.alloc() : memref<524288xi8>
    %alloc_50 = memref.alloc() : memref<524288xi8>
    %alloc_51 = memref.alloc() : memref<31254528xi8>
    %alloc_52 = memref.alloc() : memref<31254528xi8>
    %alloc_53 = memref.alloc() : memref<31254528xi8>
    %alloc_54 = memref.alloc() : memref<31254528xi8>
    %alloc_55 = memref.alloc() : memref<31254528xi8>
    %alloc_56 = memref.alloc() : memref<512x128xf32>
    byre.compute @FillOp(%alloc_56) {memory_effects = [2 : i32], value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32>
    %alloc_57 = memref.alloc() : memref<2x128xf32>
    byre.compute @FillOp(%alloc_57) {memory_effects = [2 : i32], value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32>
    %alloc_58 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @FillOp(%alloc_58) {memory_effects = [2 : i32], value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : memref<2x128x128xf32>
    %0 = "byre.alias"(%arg2) {offset = 0 : i64} : (memref<1x512xi64>) -> memref<128xi64>
    %1 = "byre.alias"(%arg3) {offset = 0 : i64} : (memref<1x512xi64>) -> memref<1x128xi64>
    %2 = "byre.alias"(%alloc_3) {offset = 0 : i64} : (memref<32xi8>) -> memref<256xi1>
    byre.compute @PTXOp(%arg1, %2) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown0", memory_effects = [1 : i32, 2 : i32]} : memref<2x128xi64>, memref<256xi1>
    %3 = "byre.alias"(%arg1) {offset = 0 : i64} : (memref<2x128xi64>) -> memref<256xi64>
    %4 = "byre.alias"(%alloc_55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256xui32>
    %5 = "byre.alias"(%alloc_21) {offset = 0 : i64} : (memref<2048xi8>) -> memref<256x1xi64>
    %6 = "byre.alias"(%alloc_5) {offset = 0 : i64} : (memref<32xi8>) -> memref<256xi1>
    byre.compute @PTXOp(%arg0, %4, %5, %6) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown1", memory_effects = [1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %7 = "byre.alias"(%alloc_54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg4, %4, %7) {dim = 0 : i32, memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    %8 = "byre.alias"(%alloc_54) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<256xui32>
    %9 = "byre.alias"(%alloc_20) {offset = 0 : i64} : (memref<2048xi8>) -> memref<256x1xi64>
    %10 = "byre.alias"(%alloc_4) {offset = 0 : i64} : (memref<32xi8>) -> memref<256xi1>
    byre.compute @PTXOp(%0, %8, %9, %10) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown2", memory_effects = [1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %11 = "byre.alias"(%alloc_55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg5, %8, %11) {dim = 0 : i32, memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %12 = "byre.alias"(%alloc_54) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<128xui32>
    %13 = "byre.alias"(%alloc_19) {offset = 0 : i64} : (memref<1024xi8>) -> memref<128x1xi64>
    %14 = "byre.alias"(%alloc_2) {offset = 0 : i64} : (memref<16xi8>) -> memref<128xi1>
    byre.compute @PTXOp(%1, %12, %13, %14) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown3", memory_effects = [1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128xi64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    %15 = "byre.alias"(%alloc_55) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<128x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg6, %12, %15) {dim = 0 : i32, memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    %16 = "byre.alias"(%alloc_34) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @PTXOp(%7, %11, %15, %16) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown4", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>
    %17 = "byre.alias"(%alloc_35) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %18 = "byre.alias"(%alloc_18) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %19 = "byre.alias"(%alloc_17) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    byre.compute @ftv4.layernorm(%16, %arg7, %arg8, %17, %18, %19) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %20 = "byre.alias"(%alloc_36) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%17, %arg9, %arg10, %20) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %21 = "byre.alias"(%alloc_37) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%17, %arg11, %arg12, %21) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %22 = "byre.alias"(%alloc_54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul(%20, %21, %22) {memory_effects = [1 : i32, 1 : i32, 2 : i32], scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %23 = "byre.alias"(%alloc_46) {offset = 0 : i64} : (memref<262144xi8>) -> memref<2x2x128x128xf32>
    %24 = "byre.alias"(%alloc_55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x128xf32>
    %25 = "byre.alias"(%alloc_23) {offset = 0 : i64} : (memref<65536xi8>) -> memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%22, %alloc_58, %23, %24, %25) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %26 = "byre.alias"(%alloc_38) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%17, %arg13, %arg14, %26) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %27 = "byre.alias"(%alloc_55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%23, %26, %27) {memory_effects = [1 : i32, 1 : i32, 2 : i32], scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %28 = "byre.alias"(%alloc_39) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%27, %28) {forward_transpose_type = "TRANSPOSE0213", memory_effects = [1 : i32, 2 : i32]} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %29 = "byre.alias"(%alloc_39) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %30 = "byre.alias"(%alloc_55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear(%29, %arg15, %arg16, %30) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %31 = "byre.alias"(%alloc_33) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %32 = "byre.alias"(%alloc_16) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %33 = "byre.alias"(%alloc_15) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %34 = "byre.alias"(%alloc_24) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%30, %arg17, %arg18, %17, %31, %32, %33, %34) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %35 = "byre.alias"(%alloc_48) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x512xf32>
    %36 = "byre.alias"(%alloc_49) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x512xf32>
    %37 = "byre.alias"(%alloc_1) {offset = 0 : i64} : (memref<0xi8>) -> memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%31, %arg19, %arg20, %35, %36, %37) {act_gelu = true, dropout_rate = 0.000000e+00 : f32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear(%35, %arg21, %arg22, %30) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %38 = "byre.alias"(%alloc_25) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %39 = "byre.alias"(%alloc_14) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %40 = "byre.alias"(%alloc_13) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %41 = "byre.alias"(%alloc_26) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%30, %arg23, %arg24, %31, %38, %39, %40, %41) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %42 = "byre.alias"(%alloc_27) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%38, %arg25, %arg26, %42) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %43 = "byre.alias"(%alloc_28) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%38, %arg27, %arg28, %43) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%42, %43, %22) {memory_effects = [1 : i32, 1 : i32, 2 : i32], scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %44 = "byre.alias"(%alloc_45) {offset = 0 : i64} : (memref<262144xi8>) -> memref<2x2x128x128xf32>
    %45 = "byre.alias"(%alloc_22) {offset = 0 : i64} : (memref<65536xi8>) -> memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%22, %alloc_58, %44, %24, %45) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %46 = "byre.alias"(%alloc_29) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%38, %arg29, %arg30, %46) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%44, %46, %27) {memory_effects = [1 : i32, 1 : i32, 2 : i32], scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %47 = "byre.alias"(%alloc_30) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%27, %47) {forward_transpose_type = "TRANSPOSE0213", memory_effects = [1 : i32, 2 : i32]} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %48 = "byre.alias"(%alloc_30) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear(%48, %arg31, %arg32, %30) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %49 = "byre.alias"(%alloc_31) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %50 = "byre.alias"(%alloc_12) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %51 = "byre.alias"(%alloc_11) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %52 = "byre.alias"(%alloc_32) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%30, %arg33, %arg34, %38, %49, %50, %51, %52) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %53 = "byre.alias"(%alloc_50) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x512xf32>
    %54 = "byre.alias"(%alloc_47) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x512xf32>
    %55 = "byre.alias"(%alloc_0) {offset = 0 : i64} : (memref<0xi8>) -> memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%49, %arg35, %arg36, %53, %54, %55) {act_gelu = true, dropout_rate = 0.000000e+00 : f32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear(%53, %arg37, %arg38, %30) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %56 = "byre.alias"(%alloc_41) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %57 = "byre.alias"(%alloc_10) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %58 = "byre.alias"(%alloc_9) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %59 = "byre.alias"(%alloc_42) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%30, %arg39, %arg40, %49, %56, %57, %58, %59) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %60 = "byre.alias"(%alloc_43) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %61 = "byre.alias"(%alloc_44) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %62 = "byre.alias"(%alloc) {offset = 0 : i64} : (memref<0xi8>) -> memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%56, %arg41, %arg42, %60, %61, %62) {act_gelu = true, dropout_rate = 0.000000e+00 : f32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>
    %63 = "byre.alias"(%alloc_40) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %64 = "byre.alias"(%alloc_8) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %65 = "byre.alias"(%alloc_7) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    byre.compute @ftv4.layernorm(%60, %arg43, %arg44, %63, %64, %65) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %66 = "byre.alias"(%alloc_40) {offset = 0 : i64} : (memref<131072xi8>) -> memref<256x128xf32>
    %67 = "byre.alias"(%alloc_55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    byre.compute @MatmulOpf32f32f32(%66, %arg4, %67) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>
    byre.compute @PTXOp(%67, %arg45, %arg46) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 3 : i32], kernel_name = "Unknown5", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256x30522xf32>, memref<30522xf32>, memref<2x128x30522xf32>
    %68 = "byre.alias"(%arg46) {offset = 0 : i64} : (memref<2x128x30522xf32>) -> memref<256x30522xf32>
    %69 = "byre.alias"(%alloc_53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256xf32>
    byre.compute @ReduceMaxOpf32f32(%68, %69) {dimensions = dense<1> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<256x30522xf32>, memref<256xf32>
    %70 = "byre.alias"(%alloc_54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    byre.compute @PTXOp(%69, %68, %67, %70) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown6", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceSumOpf32f32(%70, %69) {dimensions = dense<1> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<256x30522xf32>, memref<256xf32>
    %71 = "byre.alias"(%alloc_6) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    byre.compute @PTXOp(%69, %71) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown7", memory_effects = [1 : i32, 2 : i32]} : memref<256xf32>, memref<256xf32>
    %72 = "byre.alias"(%alloc_53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    %73 = "byre.alias"(%alloc_52) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    %74 = "byre.alias"(%alloc_51) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    byre.compute @PTXOp(%71, %67, %3, %2, %70, %72, %73, %74) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8", memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %75 = "byre.alias"(%alloc_55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<f32>
    byre.compute @ReduceSumOpf32f32(%72, %75) {dimensions = dense<[0, 1]> : tensor<2xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<256x30522xf32>, memref<f32>
    %76 = "byre.alias"(%alloc_53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<f32>
    byre.compute @ReduceSumOpf32f32(%70, %76) {dimensions = dense<[0, 1]> : tensor<2xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<256x30522xf32>, memref<f32>
    byre.compute @PTXOp(%75, %76, %arg47) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], kernel_name = "Unknown9", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<f32>, memref<f32>, memref<f32>
    byre.compute @PTXOp(%76, %75) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown10", memory_effects = [1 : i32, 2 : i32]} : memref<f32>, memref<f32>
    byre.compute @PTXOp(%75, %73, %70) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [0 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown11", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<f32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceSumOpf32f32(%70, %69) {dimensions = dense<1> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<256x30522xf32>, memref<256xf32>
    byre.compute @PTXOp(%69, %74, %70, %67) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown12", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %77 = "byre.alias"(%alloc_55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x30522xf32>
    %78 = "byre.alias"(%alloc_54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<30522x128xf32>
    byre.compute @MatmulOpf32f32f32(%66, %67, %78) {lhs_contracting_dimension = 0 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32>, memref<256x30522xf32>, memref<30522x128xf32>
    %79 = "byre.alias"(%alloc_54) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<256x128xf32>
    byre.compute @MatmulOpf32f32f32(%67, %arg4, %79) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>
    %80 = "byre.alias"(%alloc_54) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    %81 = "byre.alias"(%alloc_53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%80, %60, %arg43, %64, %65, %81, %arg87, %arg88) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%81, %56, %arg41, %61, %62, %80, %arg85, %arg86) {act_gelu = true, dropout_rate = 0.000000e+00 : f32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %82 = "byre.alias"(%alloc_52) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%80, %59, %arg39, %57, %58, %81, %arg83, %arg84, %82) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %83 = "byre.alias"(%alloc_54) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x128x512xf32>
    byre.compute @ftv4.linear_backward(%81, %53, %arg37, %83, %arg81, %arg82) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%83, %49, %arg35, %54, %55, %81, %arg79, %arg80) {act_gelu = true, dropout_rate = 0.000000e+00 : f32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    byre.compute @PTXOp(%82, %81, %80) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown14", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %84 = "byre.alias"(%alloc_50) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%80, %52, %arg33, %50, %51, %82, %arg77, %arg78, %84) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%82, %48, %arg31, %81, %arg75, %arg76) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %85 = "byre.alias"(%alloc_53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x2x64xf32>
    %86 = "byre.alias"(%alloc_54) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%85, %86) {forward_transpose_type = "TRANSPOSE0213", memory_effects = [1 : i32, 2 : i32]} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    %87 = "byre.alias"(%alloc_53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x128xf32>
    %88 = "byre.alias"(%alloc_52) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%86, %44, %46, %87, %88) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32], scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    %89 = "byre.alias"(%alloc_54) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax_backward(%87, %44, %45, %89) {dropout_rate = 0.000000e+00 : f32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %90 = "byre.alias"(%alloc_52) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    %91 = "byre.alias"(%alloc_51) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%89, %42, %43, %90, %91) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32], scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%90, %38, %arg25, %81, %arg69, %arg70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%88, %38, %arg29, %80, %arg73, %arg74) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %92 = "byre.alias"(%alloc_54) {offset = 15758336 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%91, %38, %arg27, %92, %arg71, %arg72) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @PTXOp(%84, %81, %80, %92, %82) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown15", memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %93 = "byre.alias"(%alloc_51) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%82, %41, %arg23, %39, %40, %81, %arg67, %arg68, %93) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%81, %35, %arg21, %83, %arg65, %arg66) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%83, %31, %arg19, %36, %37, %81, %arg63, %arg64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    byre.compute @PTXOp(%93, %81, %80) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown16", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%80, %34, %arg17, %32, %33, %81, %arg61, %arg62, %93) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%81, %29, %arg15, %80, %arg59, %arg60) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %94 = "byre.alias"(%alloc_54) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x128x2x64xf32>
    %95 = "byre.alias"(%alloc_53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%94, %95) {forward_transpose_type = "TRANSPOSE0213", memory_effects = [1 : i32, 2 : i32]} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%95, %23, %26, %89, %88) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32], scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.softmax_backward(%89, %23, %25, %87) {dropout_rate = 0.000000e+00 : f32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %96 = "byre.alias"(%alloc_54) {offset = 15758336 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%87, %20, %21, %96, %86) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32], scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%96, %17, %arg9, %81, %arg53, %arg54) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %97 = "byre.alias"(%alloc_53) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%88, %17, %arg13, %97, %arg57, %arg58) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %98 = "byre.alias"(%alloc_53) {offset = 262144 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%86, %17, %arg11, %98, %arg55, %arg56) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @PTXOp(%93, %81, %97, %98, %80) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown17", memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%80, %16, %arg7, %18, %19, %81, %arg51, %arg52) {memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %99 = "byre.alias"(%alloc_53) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<256x128xf32>
    byre.compute @PTXOp(%6, %81, %10, %79, %99) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [1 : i32, 3 : i32, 1 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown18", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>, memref<256x128xf32>, memref<256x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%78, %5, %79, %arg48) {dim = 0 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%alloc_57, %9, %99, %arg49) {dim = 0 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    %100 = "byre.alias"(%alloc_52) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<128x128xf32>
    byre.compute @ReduceSumOpf32f32(%81, %100) {dimensions = dense<0> : tensor<1xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<2x128x128xf32>, memref<128x128xf32>
    %101 = "byre.alias"(%alloc_54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<128x128xf32>
    byre.compute @PTXOp(%14, %100, %101) {BlockSize.x = 128 : i32, GridSize.x = 128 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown19", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%alloc_56, %13, %101, %arg50) {dim = 0 : i32, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    byre.compute @ReduceSumOpf32f32(%77, %arg89) {dimensions = dense<[0, 1]> : tensor<2xi64>, memory_effects = [1 : i32, 2 : i32]} : memref<2x128x30522xf32>, memref<30522xf32>
    return
  }
}

