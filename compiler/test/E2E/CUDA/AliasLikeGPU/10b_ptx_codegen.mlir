// RUN: byteir-translate %s -gen-ptx -o-ptx device_output -dump-ptx | FileCheck %s

// CHECK-LABEL: .visible .entry Unknown0

module attributes {byre.container_module, gpu.container_module} {
  gpu.module @unified {
    llvm.func @Unknown0(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr {llvm.noalias}, %arg8: !llvm.ptr {llvm.noalias}, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr {llvm.noalias}, %arg15: !llvm.ptr {llvm.noalias}, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %1 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %2 = llvm.insertvalue %arg15, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %3 = llvm.insertvalue %arg16, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %4 = llvm.insertvalue %arg17, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %5 = llvm.insertvalue %arg20, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %6 = llvm.insertvalue %arg18, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.insertvalue %arg7, %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %9 = llvm.insertvalue %arg8, %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %10 = llvm.insertvalue %arg9, %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %11 = llvm.insertvalue %arg10, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %12 = llvm.insertvalue %arg0, %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %13 = llvm.insertvalue %arg1, %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %14 = llvm.insertvalue %arg2, %13[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %15 = llvm.insertvalue %arg3, %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %16 = llvm.mlir.constant(0 : index) : i64
      %17 = llvm.mlir.constant(25600 : index) : i64
      %18 = nvvm.read.ptx.sreg.ctaid.x : i32
      %19 = llvm.sext %18 : i32 to i64
      %20 = nvvm.read.ptx.sreg.ntid.x : i32
      %21 = llvm.sext %20 : i32 to i64
      %22 = nvvm.read.ptx.sreg.tid.x : i32
      %23 = llvm.sext %22 : i32 to i64
      %24 = llvm.mul %21, %19 : i64
      %25 = llvm.add %23, %24 : i64
      %26 = nvvm.read.ptx.sreg.nctaid.x : i32
      %27 = llvm.sext %26 : i32 to i64
      %28 = llvm.mul %21, %27 : i64
      llvm.br ^bb1(%25 : i64)
    ^bb1(%29: i64):  // 2 preds: ^bb0, ^bb2
      %30 = llvm.icmp "slt" %29, %17 : i64
      llvm.cond_br %30, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
      %32 = llvm.insertvalue %29, %13[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %33 = llvm.mlir.constant(1 : index) : i64
      %34 = llvm.insertvalue %33, %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %35 = llvm.mlir.constant(200 : index) : i64
      %36 = llvm.getelementptr %arg1[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %37 = llvm.mul %16, %35 : i64
      %38 = llvm.add %37, %16 : i64
      %39 = llvm.getelementptr %36[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %40 = llvm.load %39 : !llvm.ptr -> f32
      %41 = llvm.insertvalue %29, %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %42 = llvm.insertvalue %33, %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %43 = llvm.getelementptr %arg8[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %44 = llvm.getelementptr %43[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %45 = llvm.load %44 : !llvm.ptr -> f32
      %46 = llvm.fadd %40, %45  : f32
      %47 = llvm.insertvalue %29, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %48 = llvm.insertvalue %33, %47[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %49 = llvm.insertvalue %35, %48[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %50 = llvm.insertvalue %33, %49[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %51 = llvm.mlir.constant(100 : index) : i64
      %52 = llvm.getelementptr %arg15[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %53 = llvm.mul %16, %51 : i64
      %54 = llvm.add %37, %53 : i64
      %55 = llvm.add %54, %16 : i64
      %56 = llvm.getelementptr %52[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %46, %56 : f32, !llvm.ptr
      %57 = llvm.add %29, %28 : i64
      llvm.br ^bb1(%57 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
}