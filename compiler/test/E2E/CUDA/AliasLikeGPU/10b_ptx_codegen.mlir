// RUN: byteir-translate %s -gen-ptx -o-ptx device_output -dump-ptx | FileCheck %s

// CHECK-LABEL: .visible .entry Unknown0

module attributes {byre.container_module, gpu.container_module} {
  gpu.module @unified {
    llvm.func @Unknown0(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr {llvm.noalias}, %arg8: !llvm.ptr {llvm.noalias}, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr {llvm.noalias}, %arg15: !llvm.ptr {llvm.noalias}, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %2 = llvm.insertvalue %arg15, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %3 = llvm.insertvalue %arg16, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %4 = llvm.insertvalue %arg17, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %5 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %6 = llvm.insertvalue %arg8, %5[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %7 = llvm.insertvalue %arg9, %6[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %8 = llvm.insertvalue %arg10, %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %9 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %10 = llvm.insertvalue %arg1, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %11 = llvm.insertvalue %arg2, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %12 = llvm.insertvalue %arg3, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %13 = llvm.mlir.constant(0 : index) : i64
      %14 = llvm.mlir.constant(102400 : index) : i64
      %15 = nvvm.read.ptx.sreg.ctaid.x : i32
      %16 = llvm.sext %15 : i32 to i64
      %17 = nvvm.read.ptx.sreg.ntid.x : i32
      %18 = llvm.sext %17 : i32 to i64
      %19 = nvvm.read.ptx.sreg.tid.x : i32
      %20 = llvm.sext %19 : i32 to i64
      %21 = llvm.mul %18, %16 : i64
      %22 = llvm.add %20, %21 : i64
      %23 = nvvm.read.ptx.sreg.nctaid.x : i32
      %24 = llvm.sext %23 : i32 to i64
      %25 = llvm.mul %18, %24 : i64
      llvm.br ^bb1(%22 : i64)
    ^bb1(%26: i64):  // 2 preds: ^bb0, ^bb2
      %27 = llvm.icmp "slt" %26, %14 : i64
      llvm.cond_br %27, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %28 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
      %29 = llvm.insertvalue %26, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %30 = llvm.mlir.constant(1 : index) : i64
      %31 = llvm.insertvalue %30, %29[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %32 = llvm.mlir.constant(200 : index) : i64
      %33 = llvm.getelementptr %arg1[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %34 = llvm.mul %13, %32 : i64
      %35 = llvm.add %34, %13 : i64
      %36 = llvm.getelementptr %33[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %37 = llvm.load %36 : !llvm.ptr -> f32
      %38 = llvm.insertvalue %26, %6[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %39 = llvm.insertvalue %30, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %40 = llvm.getelementptr %arg8[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %41 = llvm.getelementptr %40[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %42 = llvm.load %41 : !llvm.ptr -> f32
      %43 = llvm.fadd %37, %42  : f32
      %44 = llvm.insertvalue %26, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %45 = llvm.insertvalue %30, %44[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %46 = llvm.getelementptr %arg15[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %47 = llvm.getelementptr %46[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %43, %47 : f32, !llvm.ptr
      %48 = llvm.add %26, %25 : i64
      llvm.br ^bb1(%48 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
}