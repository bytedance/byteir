// RUN: byteir-opt %s --to-llvm | FileCheck %s

// CHECK-LABEL: llvm.mlir.global
// CHECK: llvm.func

module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    memref.global "private" constant @__constant_100x1296xi32 : memref<100x1296xi32> = dense<1>
    memref.global "private" constant @__constant_100xi32 : memref<100xi32> = dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]>
    func.func @Unknown0(%arg0: memref<1x100x27x48x3xf32>, %arg1: memref<51200xi32>) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
      %c0_i32 = arith.constant 0 : i32
      %c5_i32 = arith.constant 5 : i32
      %c3_i32 = arith.constant 3 : i32
      %c6_i32 = arith.constant 6 : i32
      %c51200 = arith.constant 51200 : index
      %c1296 = arith.constant 1296 : index
      %c388800 = arith.constant 388800 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c129600 = arith.constant 129600 : index
      %0 = memref.get_global @__constant_100xi32 : memref<100xi32>
      %1 = memref.get_global @__constant_100x1296xi32 : memref<100x1296xi32>
      %alloc = memref.alloc() : memref<1x100x27x48x3xi32>
      %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3, 4]] : memref<1x100x27x48x3xf32> into memref<388800xf32>
      %collapse_shape_0 = memref.collapse_shape %alloc [[0, 1, 2, 3, 4]] : memref<1x100x27x48x3xi32> into memref<388800xi32>
      scf.for %arg2 = %c0 to %c388800 step %c1 {
        %2 = memref.load %collapse_shape[%arg2] : memref<388800xf32>
        %3 = arith.fptosi %2 : f32 to i32
        memref.store %3, %collapse_shape_0[%arg2] : memref<388800xi32>
      }
      %collapse_shape_1 = memref.collapse_shape %alloc [[0, 1], [2, 3], [4]] : memref<1x100x27x48x3xi32> into memref<100x1296x3xi32>
      %subview = memref.subview %collapse_shape_1[0, 0, 0] [100, 1296, 1] [1, 1, 1] : memref<100x1296x3xi32> to memref<100x1296x1xi32, strided<[3888, 3, 1]>>
      %subview_2 = memref.subview %collapse_shape_1[0, 0, 1] [100, 1296, 1] [1, 1, 1] : memref<100x1296x3xi32> to memref<100x1296x1xi32, strided<[3888, 3, 1], offset: 1>>
      %subview_3 = memref.subview %collapse_shape_1[0, 0, 2] [100, 1296, 1] [1, 1, 1] : memref<100x1296x3xi32> to memref<100x1296x1xi32, strided<[3888, 3, 1], offset: 2>>
      %alloc_4 = memref.alloc() : memref<100x1296x1xi32>
      scf.for %arg2 = %c0 to %c129600 step %c1 {
        %2 = arith.remsi %arg2, %c1296 : index
        %3 = arith.divsi %arg2, %c1296 : index
        %4 = memref.load %subview_3[%3, %2, %c0] : memref<100x1296x1xi32, strided<[3888, 3, 1], offset: 2>>
        %5 = memref.load %subview[%3, %2, %c0] : memref<100x1296x1xi32, strided<[3888, 3, 1]>>
        %6 = memref.load %subview_2[%3, %2, %c0] : memref<100x1296x1xi32, strided<[3888, 3, 1], offset: 1>>
        %7 = memref.load %0[%3] : memref<100xi32>
        %8 = arith.shrsi %6, %c5_i32 : i32
        %9 = arith.shli %8, %c3_i32 : i32
        %10 = arith.shrsi %5, %c5_i32 : i32
        %11 = arith.shli %10, %c6_i32 : i32
        %12 = arith.addi %11, %9 : i32
        %13 = arith.shrsi %4, %c5_i32 : i32
        %14 = arith.addi %13, %12 : i32
        %15 = arith.addi %14, %7 : i32
        memref.store %15, %alloc_4[%3, %2, %c0] : memref<100x1296x1xi32>
      }
      memref.dealloc %alloc : memref<1x100x27x48x3xi32>
      scf.for %arg2 = %c0 to %c51200 step %c1 {
        memref.store %c0_i32, %arg1[%arg2] : memref<51200xi32>
      }
      scf.for %arg2 = %c0 to %c129600 step %c1 {
        %2 = arith.remsi %arg2, %c1296 : index
        %3 = arith.divsi %arg2, %c1296 : index
        %4 = memref.load %alloc_4[%3, %2, %c0] : memref<100x1296x1xi32>
        %5 = arith.index_cast %4 : i32 to index
        %6 = memref.load %arg1[%5] : memref<51200xi32>
        %7 = memref.load %1[%3, %2] : memref<100x1296xi32>
        %8 = arith.addi %6, %7 : i32
        memref.store %8, %arg1[%5] : memref<51200xi32>
      }
      memref.dealloc %alloc_4 : memref<100x1296x1xi32>
      return
    }
  }
  func.func @main(%arg0: memref<1x100x27x48x3xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<51200xi32> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    byre.compute @LLVMJITOp(%arg0, %arg1) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<1x100x27x48x3xf32>, memref<51200xi32>
    return
  }
}