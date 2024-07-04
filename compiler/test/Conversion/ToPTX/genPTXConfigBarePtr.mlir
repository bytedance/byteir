// RUN: byteir-opt -gen-ptx-config='use-bare-ptr-memref-call-conv=1' -convert-to-byre --allow-unregistered-dialect %s | FileCheck %s

module attributes {gpu.container_module}  {
  func.func private @Unknown0(%arg0: memref<1x128xi64>, %arg1: memref<128xi64>, %arg2: memref<128xi64>, %arg3: memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {byre_compute_name = "Unknown0", __byteir_elementwise_fusion__} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    %1 = memref.alloc() : memref<128xui32>
    %2 = memref.alloc() : memref<128xi64>
    %3 = memref.expand_shape %2 [[0, 1]] output_shape [128, 1] : memref<128xi64> into memref<128x1xi64>
    %4 = memref.alloc() : memref<128xi1>
    gpu.launch_func  @Unknown0_kernel::@Unknown0_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%0 : memref<128xi64>, %1 : memref<128xui32>, %arg1 : memref<128xi64>, %arg2 : memref<128xi64>, %2 : memref<128xi64>, %arg3 : memref<128xf64>, %4 : memref<128xi1>)
    return %1, %3, %4 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  gpu.module @Unknown0_kernel {
    gpu.func @Unknown0_kernel(%arg0: memref<128xi64>, %arg1: memref<128xui32>, %arg2: memref<128xi64>, %arg3: memref<128xi64>, %arg4: memref<128xi64>, %arg5: memref<128xf64>, %arg6: memref<128xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c128 : index
      scf.if %6 {
        %7 = memref.load %arg0[%5] : memref<128xi64>
        %8 = arith.trunci %7 : i64 to i32
        %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
        memref.store %9, %arg1[%5] : memref<128xui32>
        %10 = memref.load %arg2[%5] : memref<128xi64>
        %11 = memref.load %arg3[%5] : memref<128xi64>
        %12 = arith.addi %7, %11 : i64
        %13 = arith.cmpi slt, %7, %10 : i64
        %14 = arith.select %13, %12, %7 : i64
        memref.store %14, %arg4[%5] : memref<128xi64>
        %15 = memref.load %arg5[%5] : memref<128xf64>
        %16 = arith.sitofp %7 : i64 to f64
        %17 = arith.cmpf une, %16, %15 : f64
        memref.store %17, %arg6[%5] : memref<128xi1>
      }
      gpu.return
    }
  }
  func.func @main(%arg0: memref<1x128xi64> {__placeholder__byre.argname = "A"}) -> (memref<128xui32> {__placeholder__byre.argname = "B"}, memref<128x1xi64> {__placeholder__byre.argname = "C"}, memref<128xi1> {__placeholder__byre.argname = "D"}) attributes {__placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<128xi64>
    "lmhlo.constant"(%0) {value = dense<0> : tensor<128xi64>} : (memref<128xi64>) -> ()
    %1 = memref.alloc() : memref<128xf64>
    "lmhlo.constant"(%1) {value = dense<0.000000e+00> : tensor<128xf64>} : (memref<128xf64>) -> ()
    %2 = memref.alloc() : memref<128xi64>
    "lmhlo.constant"(%2) {value = dense<30522> : tensor<128xi64>} : (memref<128xi64>) -> ()
    %3:3 = call @Unknown0(%arg0, %0, %2, %1) : (memref<1x128xi64>, memref<128xi64>, memref<128xi64>, memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    return %3#0, %3#1, %3#2 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }

  // CHECK-NOT: func.func private @Unknown0
  // CHECK: byre.compute @PTXOp
  // CHEKC-SAME: {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, call_convention = "bare_ptr", kernel_name = "Unknown0_kernel"}
}

