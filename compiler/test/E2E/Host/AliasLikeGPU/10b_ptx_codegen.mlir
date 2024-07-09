// RUN: byteir-translate %s -gen-ptx -o-ptx device_output -dump-ptx | FileCheck %s

// CHECK-LABEL: .visible .entry Unknown0

module attributes {byre.container_module, gpu.container_module} {
  gpu.module @unified {
    llvm.func @Unknown0(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr {llvm.noalias}, %arg10: !llvm.ptr {llvm.noalias}, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr {llvm.noalias}, %arg19: !llvm.ptr {llvm.noalias}, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %1 = llvm.insertvalue %arg18, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %2 = llvm.insertvalue %arg19, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %3 = llvm.insertvalue %arg20, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %4 = llvm.insertvalue %arg21, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %5 = llvm.insertvalue %arg24, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %6 = llvm.insertvalue %arg22, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %7 = llvm.insertvalue %arg9, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %8 = llvm.insertvalue %arg10, %7[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %9 = llvm.insertvalue %arg11, %8[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %10 = llvm.insertvalue %arg12, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %11 = llvm.insertvalue %arg15, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %12 = llvm.insertvalue %arg13, %11[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %13 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %14 = llvm.insertvalue %arg1, %13[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %15 = llvm.insertvalue %arg2, %14[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %16 = llvm.insertvalue %arg3, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %17 = llvm.insertvalue %arg6, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %18 = llvm.insertvalue %arg4, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %19 = llvm.mlir.constant(0 : index) : i64
      %20 = llvm.mlir.constant(25600 : index) : i64
      %21 = nvvm.read.ptx.sreg.ctaid.x : i32
      %22 = llvm.sext %21 : i32 to i64
      %23 = nvvm.read.ptx.sreg.ntid.x : i32
      %24 = llvm.sext %23 : i32 to i64
      %25 = nvvm.read.ptx.sreg.tid.x : i32
      %26 = llvm.sext %25 : i32 to i64
      %27 = llvm.mul %24, %22 : i64
      %28 = llvm.add %26, %27 : i64
      %29 = nvvm.read.ptx.sreg.nctaid.x : i32
      %30 = llvm.sext %29 : i32 to i64
      %31 = llvm.mul %24, %30 : i64
      llvm.br ^bb1(%28 : i64)
    ^bb1(%32: i64):  // 2 preds: ^bb0, ^bb2
      %33 = llvm.icmp "slt" %32, %20 : i64
      llvm.cond_br %33, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
      %35 = llvm.insertvalue %32, %14[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %36 = llvm.mlir.constant(1 : index) : i64
      %37 = llvm.insertvalue %36, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %38 = llvm.mlir.constant(200 : index) : i64
      %39 = llvm.insertvalue %38, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %40 = llvm.insertvalue %36, %39[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %41 = llvm.mlir.constant(100 : index) : i64
      %42 = llvm.getelementptr %arg1[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %43 = llvm.mul %19, %38 : i64
      %44 = llvm.mul %19, %41 : i64
      %45 = llvm.add %43, %44 : i64
      %46 = llvm.add %45, %19 : i64
      %47 = llvm.getelementptr %42[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %48 = llvm.load %47 : !llvm.ptr -> f32
      %49 = llvm.insertvalue %32, %8[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %50 = llvm.insertvalue %36, %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %51 = llvm.insertvalue %38, %50[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %52 = llvm.insertvalue %36, %51[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %53 = llvm.getelementptr %arg10[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %54 = llvm.getelementptr %53[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %55 = llvm.load %54 : !llvm.ptr -> f32
      %56 = llvm.fadd %48, %55  : f32
      %57 = llvm.insertvalue %32, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %58 = llvm.insertvalue %36, %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %59 = llvm.insertvalue %38, %58[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %60 = llvm.insertvalue %36, %59[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
      %61 = llvm.getelementptr %arg19[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %62 = llvm.getelementptr %61[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %56, %62 : f32, !llvm.ptr
      %63 = llvm.add %32, %31 : i64
      llvm.br ^bb1(%63 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
}