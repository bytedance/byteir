// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module, llvm.data_layout = ""} {
  llvm.func @Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(97 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(1103515245 : i32) : i32
    %4 = llvm.mlir.constant(12345 : i32) : i32
    %5 = llvm.mlir.constant(2.32830644E-10 : f32) : f32
    %6 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.br ^bb1(%1 : i64)
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
    %8 = llvm.icmp "slt" %7, %0 : i64
    llvm.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %9 = llvm.load %arg1 : !llvm.ptr -> i64
    %10 = llvm.load %arg4 : !llvm.ptr -> i64
    %11 = llvm.trunc %9 : i64 to i32
    %12 = llvm.trunc %10 : i64 to i32
    %13 = llvm.add %11, %12  : i32
    %14 = llvm.mul %13, %3  : i32
    %15 = llvm.add %14, %4  : i32
    %16 = llvm.trunc %7 : i64 to i32
    %17 = llvm.add %16, %15  : i32
    %18 = llvm.mul %17, %3  : i32
    %19 = llvm.add %18, %4  : i32
    %20 = llvm.uitofp %19 : i32 to f32
    %21 = llvm.fmul %20, %5  : f32
    %22 = llvm.fadd %21, %6  : f32
    %23 = llvm.mul %1, %0  : i64
    %24 = llvm.add %23, %7  : i64
    %25 = llvm.getelementptr %arg7[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %22, %25 : f32, !llvm.ptr
    %26 = llvm.add %7, %2  : i64
    llvm.br ^bb1(%26 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.extractvalue %4[2] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @Unknown0(%1, %2, %3, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
}