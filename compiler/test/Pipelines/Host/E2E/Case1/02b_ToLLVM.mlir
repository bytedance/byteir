// RUN: byteir-opt --to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.mlir.global
// CHECK: llvm.func
module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    memref.global "private" constant @__constant_100x1296xi32 : memref<100x1296xi32> = dense<1>
    memref.global "private" constant @__constant_100xi32 : memref<100xi32> = dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]>
    memref.global "private" constant @__constant_51200xi32 : memref<51200xi32> = dense<0>
    func.func @Unknown6(%arg0: memref<1x100x27x48x3xf32>, %arg1: memref<51200xi32>) attributes {__byre__kernel_name = "Unknown6", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
      %c6_i32 = arith.constant 6 : i32
      %c3_i32 = arith.constant 3 : i32
      %c5_i32 = arith.constant 5 : i32
      %c51200 = arith.constant 51200 : index
      %c1296 = arith.constant 1296 : index
      %c3 = arith.constant 3 : index
      %c48 = arith.constant 48 : index
      %c27 = arith.constant 27 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c-1 = arith.constant -1 : index
      %c100 = arith.constant 100 : index
      %c129600 = arith.constant 129600 : index
      %c388800 = arith.constant 388800 : index
      %0 = memref.get_global @__constant_51200xi32 : memref<51200xi32>
      %1 = memref.get_global @__constant_100xi32 : memref<100xi32>
      %2 = memref.get_global @__constant_100x1296xi32 : memref<100x1296xi32>
      %alloc = memref.alloc() : memref<1x100x27x48x3xi32>
      scf.for %arg2 = %c0 to %c388800 step %c1 {
        %3 = arith.remsi %arg2, %c3 : index
        %4 = arith.divsi %arg2, %c3 : index
        %5 = arith.remsi %4, %c48 : index
        %6 = arith.divsi %4, %c48 : index
        %7 = arith.remsi %6, %c27 : index
        %8 = arith.divsi %6, %c27 : index
        %9 = arith.remsi %8, %c100 : index
        %10 = arith.divsi %8, %c100 : index
        %11 = memref.load %arg0[%10, %9, %7, %5, %3] : memref<1x100x27x48x3xf32>
        %12 = arith.fptosi %11 : f32 to i32
        memref.store %12, %alloc[%10, %9, %7, %5, %3] : memref<1x100x27x48x3xi32>
      }
      %alloc_0 = memref.alloc() : memref<100x1296x1xi32>
      scf.for %arg2 = %c0 to %c129600 step %c1 {
        %3 = arith.remsi %arg2, %c1296 : index
        %4 = arith.divsi %arg2, %c1296 : index
        %5 = arith.cmpi slt, %4, %c0 : index
        %6 = arith.subi %c-1, %4 : index
        %7 = arith.select %5, %6, %4 : index
        %8 = arith.divsi %7, %c100 : index
        %9 = arith.subi %c-1, %8 : index
        %10 = arith.select %5, %9, %8 : index
        %11 = arith.remsi %4, %c100 : index
        %12 = arith.cmpi slt, %11, %c0 : index
        %13 = arith.addi %11, %c100 : index
        %14 = arith.select %12, %13, %11 : index
        %15 = arith.cmpi slt, %3, %c0 : index
        %16 = arith.subi %c-1, %3 : index
        %17 = arith.select %15, %16, %3 : index
        %18 = arith.divsi %17, %c48 : index
        %19 = arith.subi %c-1, %18 : index
        %20 = arith.select %15, %19, %18 : index
        %21 = arith.remsi %3, %c48 : index
        %22 = arith.cmpi slt, %21, %c0 : index
        %23 = arith.addi %21, %c48 : index
        %24 = arith.select %22, %23, %21 : index
        %25 = memref.load %alloc[%10, %14, %20, %24, %c2] : memref<1x100x27x48x3xi32>
        %26 = memref.load %alloc[%10, %14, %20, %24, %c0] : memref<1x100x27x48x3xi32>
        %27 = memref.load %alloc[%10, %14, %20, %24, %c1] : memref<1x100x27x48x3xi32>
        %28 = memref.load %1[%4] : memref<100xi32>
        %29 = arith.shrsi %27, %c5_i32 : i32
        %30 = arith.shli %29, %c3_i32 : i32
        %31 = arith.shrsi %26, %c5_i32 : i32
        %32 = arith.shli %31, %c6_i32 : i32
        %33 = arith.addi %32, %30 : i32
        %34 = arith.shrsi %25, %c5_i32 : i32
        %35 = arith.addi %34, %33 : i32
        %36 = arith.addi %35, %28 : i32
        memref.store %36, %alloc_0[%4, %3, %c0] : memref<100x1296x1xi32>
      }
      memref.dealloc %alloc : memref<1x100x27x48x3xi32>
      scf.for %arg2 = %c0 to %c51200 step %c1 {
        %3 = memref.load %0[%arg2] : memref<51200xi32>
        memref.store %3, %arg1[%arg2] : memref<51200xi32>
      }
      scf.for %arg2 = %c0 to %c129600 step %c1 {
        %3 = arith.remsi %arg2, %c1296 : index
        %4 = arith.divsi %arg2, %c1296 : index
        %5 = memref.load %alloc_0[%4, %3, %c0] : memref<100x1296x1xi32>
        %6 = arith.index_cast %5 : i32 to index
        %7 = memref.load %arg1[%6] : memref<51200xi32>
        %8 = memref.load %2[%4, %3] : memref<100x1296xi32>
        %9 = arith.addi %7, %8 : i32
        memref.store %9, %arg1[%6] : memref<51200xi32>
      }
      memref.dealloc %alloc_0 : memref<100x1296x1xi32>
      return
    }
  }
  func.func @main(%arg0: memref<1x100x27x48x3xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<51200xi32> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    byre.compute @LLVMJITOp(%arg0, %arg1) {kernel_name = "Unknown6", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<1x100x27x48x3xf32>, memref<51200xi32>
    return
  }
}
