// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: constant
// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.mlir.global private constant @__constant_100x1296xi32(dense<1> : tensor<100x1296xi32>) {addr_space = 0 : i32} : !llvm.array<100 x array<1296 x i32>>
  llvm.mlir.global private constant @__constant_100xi32(dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]> : tensor<100xi32>) {addr_space = 0 : i32} : !llvm.array<100 x i32>
  llvm.func @Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: !llvm.ptr, %arg14: !llvm.ptr, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(388800 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%2 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
    %4 = llvm.icmp "slt" %3, %1 : i64
    llvm.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = llvm.getelementptr %arg1[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.fptosi %6 : f32 to i32
    %8 = llvm.getelementptr %arg14[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %7, %8 : i32, !llvm.ptr
    %9 = llvm.add %3, %0 : i64
    llvm.br ^bb1(%9 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %7 = llvm.extractvalue %0[3, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %8 = llvm.extractvalue %0[3, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %9 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %10 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %11 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %12 = llvm.extractvalue %0[4, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %13 = llvm.extractvalue %0[4, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %14 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %19 = llvm.extractvalue %14[3, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %20 = llvm.extractvalue %14[3, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %21 = llvm.extractvalue %14[3, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %22 = llvm.extractvalue %14[3, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %23 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %24 = llvm.extractvalue %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %25 = llvm.extractvalue %14[4, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %26 = llvm.extractvalue %14[4, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %27 = llvm.extractvalue %14[4, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    llvm.call @Unknown0(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @Unknown1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64) attributes {__byre__kernel_name = "Unknown1", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3 : i64) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(1296 : index) : i64
    %4 = llvm.mlir.constant(3888 : index) : i64
    %5 = llvm.mlir.constant(100 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %8 = llvm.insertvalue %arg9, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.insertvalue %arg10, %8[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.insertvalue %arg11, %9[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.insertvalue %arg12, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.insertvalue %arg15, %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.insertvalue %arg13, %12[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.insertvalue %arg16, %13[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.insertvalue %arg14, %14[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.insertvalue %arg17, %15[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %18 = llvm.insertvalue %arg0, %17[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %arg1, %18[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %6, %19[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %5, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %4, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %3, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %2, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %1, %24[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %1, %25[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.intr.stacksave : !llvm.ptr
    %28 = llvm.alloca %1 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %26, %28 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %29 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %30 = llvm.insertvalue %0, %29[0] : !llvm.struct<(i64, ptr)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(i64, ptr)> 
    %32 = llvm.alloca %1 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %16, %32 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %33 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %34 = llvm.insertvalue %0, %33[0] : !llvm.struct<(i64, ptr)> 
    %35 = llvm.insertvalue %32, %34[1] : !llvm.struct<(i64, ptr)> 
    %36 = llvm.alloca %1 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %31, %36 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %37 = llvm.alloca %1 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %35, %37 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %38 = llvm.mlir.zero : !llvm.ptr
    %39 = llvm.getelementptr %38[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    llvm.call @memrefCopy(%40, %36, %37) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %27 : !llvm.ptr
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown1(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {__byre__kernel_name = "Unknown1", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.extractvalue %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.extractvalue %10[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.extractvalue %10[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.extractvalue %10[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @Unknown1(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @Unknown2(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64) attributes {__byre__kernel_name = "Unknown2", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3 : i64) : i64
    %1 = llvm.mlir.constant(3 : index) : i64
    %2 = llvm.mlir.constant(1296 : index) : i64
    %3 = llvm.mlir.constant(3888 : index) : i64
    %4 = llvm.mlir.constant(100 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %7 = llvm.insertvalue %arg9, %6[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.insertvalue %arg10, %7[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.insertvalue %arg11, %8[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.insertvalue %arg12, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.insertvalue %arg15, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.insertvalue %arg13, %11[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.insertvalue %arg16, %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.insertvalue %arg14, %13[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.insertvalue %arg17, %14[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %17 = llvm.insertvalue %arg0, %16[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.insertvalue %arg1, %17[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %5, %18[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %4, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %3, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %2, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %1, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %5, %23[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %5, %24[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.intr.stacksave : !llvm.ptr
    %27 = llvm.alloca %5 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %25, %27 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %28 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %29 = llvm.insertvalue %0, %28[0] : !llvm.struct<(i64, ptr)> 
    %30 = llvm.insertvalue %27, %29[1] : !llvm.struct<(i64, ptr)> 
    %31 = llvm.alloca %5 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %15, %31 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %33 = llvm.insertvalue %0, %32[0] : !llvm.struct<(i64, ptr)> 
    %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(i64, ptr)> 
    %35 = llvm.alloca %5 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %30, %35 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %36 = llvm.alloca %5 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %34, %36 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %37 = llvm.mlir.zero : !llvm.ptr
    %38 = llvm.getelementptr %37[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    llvm.call @memrefCopy(%39, %35, %36) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %26 : !llvm.ptr
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {__byre__kernel_name = "Unknown2", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.extractvalue %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.extractvalue %10[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.extractvalue %10[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.extractvalue %10[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @Unknown2(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @Unknown3(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64) attributes {__byre__kernel_name = "Unknown3", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3 : i64) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(1296 : index) : i64
    %4 = llvm.mlir.constant(3888 : index) : i64
    %5 = llvm.mlir.constant(100 : index) : i64
    %6 = llvm.mlir.constant(2 : index) : i64
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %8 = llvm.insertvalue %arg9, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.insertvalue %arg10, %8[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.insertvalue %arg11, %9[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.insertvalue %arg12, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.insertvalue %arg15, %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.insertvalue %arg13, %12[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.insertvalue %arg16, %13[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.insertvalue %arg14, %14[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.insertvalue %arg17, %15[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %18 = llvm.insertvalue %arg0, %17[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %arg1, %18[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %6, %19[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %5, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %4, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %3, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %2, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %1, %24[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %1, %25[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.intr.stacksave : !llvm.ptr
    %28 = llvm.alloca %1 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %26, %28 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %29 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %30 = llvm.insertvalue %0, %29[0] : !llvm.struct<(i64, ptr)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(i64, ptr)> 
    %32 = llvm.alloca %1 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %16, %32 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %33 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %34 = llvm.insertvalue %0, %33[0] : !llvm.struct<(i64, ptr)> 
    %35 = llvm.insertvalue %32, %34[1] : !llvm.struct<(i64, ptr)> 
    %36 = llvm.alloca %1 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %31, %36 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %37 = llvm.alloca %1 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %35, %37 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %38 = llvm.mlir.zero : !llvm.ptr
    %39 = llvm.getelementptr %38[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    llvm.call @memrefCopy(%40, %36, %37) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %27 : !llvm.ptr
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown3(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {__byre__kernel_name = "Unknown3", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.extractvalue %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.extractvalue %10[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.extractvalue %10[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.extractvalue %10[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @Unknown3(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @Unknown4(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr, %arg22: !llvm.ptr, %arg23: i64, %arg24: i64, %arg25: i64) attributes {__byre__kernel_name = "Unknown4", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.addressof @__constant_100x1296xi32 : !llvm.ptr
    %1 = llvm.mlir.addressof @__constant_100xi32 : !llvm.ptr
    %2 = llvm.mlir.constant(129600 : index) : i64
    %3 = llvm.mlir.constant(51200 : index) : i64
    %4 = llvm.mlir.constant(1296 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(3 : i32) : i32
    %8 = llvm.mlir.constant(6 : i32) : i32
    %9 = llvm.mlir.constant(5 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<100 x i32>
    %12 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<100 x array<1296 x i32>>
    %13 = llvm.mlir.zero : !llvm.ptr
    %14 = llvm.getelementptr %13[129600] : (!llvm.ptr) -> !llvm.ptr, i32
    %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
    %16 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr
    llvm.br ^bb1(%6 : i64)
  ^bb1(%17: i64):  // 2 preds: ^bb0, ^bb2
    %18 = llvm.icmp "slt" %17, %2 : i64
    llvm.cond_br %18, ^bb2, ^bb3(%6 : i64)
  ^bb2:  // pred: ^bb1
    %19 = llvm.srem %17, %4  : i64
    %20 = llvm.sdiv %17, %4  : i64
    %21 = llvm.mul %20, %4 : i64
    %22 = llvm.add %21, %19 : i64
    %23 = llvm.getelementptr %arg15[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %24 = llvm.load %23 : !llvm.ptr -> i32
    %25 = llvm.mul %20, %4 : i64
    %26 = llvm.add %25, %19 : i64
    %27 = llvm.getelementptr %arg1[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %28 = llvm.load %27 : !llvm.ptr -> i32
    %29 = llvm.mul %20, %4 : i64
    %30 = llvm.add %29, %19 : i64
    %31 = llvm.getelementptr %arg8[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %32 = llvm.load %31 : !llvm.ptr -> i32
    %33 = llvm.getelementptr %11[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %34 = llvm.load %33 : !llvm.ptr -> i32
    %35 = llvm.ashr %32, %9  : i32
    %36 = llvm.shl %35, %7 : i32
    %37 = llvm.ashr %28, %9  : i32
    %38 = llvm.shl %37, %8 : i32
    %39 = llvm.add %38, %36 : i32
    %40 = llvm.ashr %24, %9  : i32
    %41 = llvm.add %40, %39 : i32
    %42 = llvm.add %41, %34 : i32
    %43 = llvm.mul %20, %4 : i64
    %44 = llvm.add %43, %19 : i64
    %45 = llvm.getelementptr %16[%44] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %42, %45 : i32, !llvm.ptr
    %46 = llvm.add %17, %5 : i64
    llvm.br ^bb1(%46 : i64)
  ^bb3(%47: i64):  // 2 preds: ^bb1, ^bb4
    %48 = llvm.icmp "slt" %47, %3 : i64
    llvm.cond_br %48, ^bb4, ^bb5(%6 : i64)
  ^bb4:  // pred: ^bb3
    %49 = llvm.getelementptr %arg22[%47] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %10, %49 : i32, !llvm.ptr
    %50 = llvm.add %47, %5 : i64
    llvm.br ^bb3(%50 : i64)
  ^bb5(%51: i64):  // 2 preds: ^bb3, ^bb6
    %52 = llvm.icmp "slt" %51, %2 : i64
    llvm.cond_br %52, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %53 = llvm.srem %51, %4  : i64
    %54 = llvm.sdiv %51, %4  : i64
    %55 = llvm.mul %54, %4 : i64
    %56 = llvm.add %55, %53 : i64
    %57 = llvm.add %56, %6 : i64
    %58 = llvm.getelementptr %16[%57] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %59 = llvm.load %58 : !llvm.ptr -> i32
    %60 = llvm.sext %59 : i32 to i64
    %61 = llvm.getelementptr %arg22[%60] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %62 = llvm.load %61 : !llvm.ptr -> i32
    %63 = llvm.mul %54, %4 : i64
    %64 = llvm.add %63, %53 : i64
    %65 = llvm.getelementptr %12[%64] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %66 = llvm.load %65 : !llvm.ptr -> i32
    %67 = llvm.add %62, %66 : i32
    %68 = llvm.getelementptr %arg22[%60] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %67, %68 : i32, !llvm.ptr
    %69 = llvm.add %51, %5 : i64
    llvm.br ^bb5(%69 : i64)
  ^bb7:  // pred: ^bb5
    llvm.call @free(%16) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown4(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {__byre__kernel_name = "Unknown4", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.extractvalue %24[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.extractvalue %24[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.extractvalue %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.extractvalue %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @Unknown4(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23, %25, %26, %27, %28, %29) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
}