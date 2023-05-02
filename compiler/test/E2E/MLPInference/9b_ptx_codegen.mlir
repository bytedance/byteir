// RUN: byteir-translate %s -gen-ptx -o-ptx device_output -dump-ptx | FileCheck %s

// CHECK-LABEL: .visible .entry Unknown

module attributes {byre.container_module, gpu.container_module, torch.debug_module_name = "GraphModule"} {
  gpu.module @unified {
    llvm.func @Unknown2(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %5 = llvm.insertvalue %arg6, %4[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %6 = llvm.insertvalue %arg7, %5[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %7 = llvm.insertvalue %arg8, %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %8 = llvm.insertvalue %arg12, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %9 = llvm.insertvalue %arg13, %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %10 = llvm.insertvalue %arg14, %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %11 = llvm.insertvalue %arg15, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %12 = llvm.mlir.constant(0 : index) : i64
      %13 = llvm.mlir.constant(20 : index) : i64
      %14 = llvm.mlir.constant(10 : index) : i64
      %15 = llvm.mlir.constant(-1 : index) : i64
      %16 = nvvm.read.ptx.sreg.ctaid.x : i32
      %17 = llvm.sext %16 : i32 to i64
      %18 = nvvm.read.ptx.sreg.ntid.x : i32
      %19 = llvm.sext %18 : i32 to i64
      %20 = nvvm.read.ptx.sreg.tid.x : i32
      %21 = llvm.sext %20 : i32 to i64
      %22 = llvm.mul %19, %17  : i64
      %23 = llvm.add %21, %22  : i64
      %24 = llvm.icmp "slt" %23, %13 : i64
      llvm.cond_br %24, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %25 = llvm.srem %23, %14  : i64
      %26 = llvm.icmp "slt" %25, %12 : i64
      %27 = llvm.add %25, %14  : i64
      %28 = llvm.select %26, %27, %25 : i1, i64
      %29 = llvm.icmp "slt" %23, %12 : i64
      %30 = llvm.sub %15, %23  : i64
      %31 = llvm.select %29, %30, %23 : i1, i64
      %32 = llvm.sdiv %31, %14  : i64
      %33 = llvm.sub %15, %32  : i64
      %34 = llvm.select %29, %33, %32 : i1, i64
      %35 = llvm.mul %34, %14  : i64
      %36 = llvm.add %35, %28  : i64
      %37 = llvm.getelementptr %arg6[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %38 = llvm.load %37 : !llvm.ptr -> f32
      %39 = llvm.getelementptr %arg1[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %40 = llvm.load %39 : !llvm.ptr -> f32
      %41 = llvm.fadd %38, %40  : f32
      %42 = llvm.getelementptr %arg13[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %41, %42 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
    llvm.func @Unknown1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %5 = llvm.insertvalue %arg6, %4[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %6 = llvm.insertvalue %arg7, %5[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %7 = llvm.insertvalue %arg8, %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %8 = llvm.insertvalue %arg12, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %9 = llvm.insertvalue %arg13, %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %10 = llvm.insertvalue %arg14, %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %11 = llvm.insertvalue %arg15, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %12 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %13 = llvm.mlir.constant(0 : index) : i64
      %14 = llvm.mlir.constant(40 : index) : i64
      %15 = llvm.mlir.constant(20 : index) : i64
      %16 = llvm.mlir.constant(-1 : index) : i64
      %17 = nvvm.read.ptx.sreg.ctaid.x : i32
      %18 = llvm.sext %17 : i32 to i64
      %19 = nvvm.read.ptx.sreg.ntid.x : i32
      %20 = llvm.sext %19 : i32 to i64
      %21 = nvvm.read.ptx.sreg.tid.x : i32
      %22 = llvm.sext %21 : i32 to i64
      %23 = llvm.mul %20, %18  : i64
      %24 = llvm.add %22, %23  : i64
      %25 = llvm.icmp "slt" %24, %14 : i64
      llvm.cond_br %25, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %26 = llvm.srem %24, %15  : i64
      %27 = llvm.icmp "slt" %26, %13 : i64
      %28 = llvm.add %26, %15  : i64
      %29 = llvm.select %27, %28, %26 : i1, i64
      %30 = llvm.icmp "slt" %24, %13 : i64
      %31 = llvm.sub %16, %24  : i64
      %32 = llvm.select %30, %31, %24 : i1, i64
      %33 = llvm.sdiv %32, %15  : i64
      %34 = llvm.sub %16, %33  : i64
      %35 = llvm.select %30, %34, %33 : i1, i64
      %36 = llvm.mul %35, %15  : i64
      %37 = llvm.add %36, %29  : i64
      %38 = llvm.getelementptr %arg6[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %39 = llvm.load %38 : !llvm.ptr -> f32
      %40 = llvm.getelementptr %arg1[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %41 = llvm.load %40 : !llvm.ptr -> f32
      %42 = llvm.fadd %39, %41  : f32
      %43 = llvm.intr.maxnum(%42, %12)  : (f32, f32) -> f32
      %44 = llvm.getelementptr %arg13[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %43, %44 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
    llvm.func @Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %5 = llvm.insertvalue %arg6, %4[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %6 = llvm.insertvalue %arg7, %5[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %7 = llvm.insertvalue %arg8, %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %8 = llvm.insertvalue %arg12, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %9 = llvm.insertvalue %arg13, %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %10 = llvm.insertvalue %arg14, %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %11 = llvm.insertvalue %arg15, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %12 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %13 = llvm.mlir.constant(0 : index) : i64
      %14 = llvm.mlir.constant(40 : index) : i64
      %15 = llvm.mlir.constant(20 : index) : i64
      %16 = llvm.mlir.constant(-1 : index) : i64
      %17 = nvvm.read.ptx.sreg.ctaid.x : i32
      %18 = llvm.sext %17 : i32 to i64
      %19 = nvvm.read.ptx.sreg.ntid.x : i32
      %20 = llvm.sext %19 : i32 to i64
      %21 = nvvm.read.ptx.sreg.tid.x : i32
      %22 = llvm.sext %21 : i32 to i64
      %23 = llvm.mul %20, %18  : i64
      %24 = llvm.add %22, %23  : i64
      %25 = llvm.icmp "slt" %24, %14 : i64
      llvm.cond_br %25, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %26 = llvm.srem %24, %15  : i64
      %27 = llvm.icmp "slt" %26, %13 : i64
      %28 = llvm.add %26, %15  : i64
      %29 = llvm.select %27, %28, %26 : i1, i64
      %30 = llvm.icmp "slt" %24, %13 : i64
      %31 = llvm.sub %16, %24  : i64
      %32 = llvm.select %30, %31, %24 : i1, i64
      %33 = llvm.sdiv %32, %15  : i64
      %34 = llvm.sub %16, %33  : i64
      %35 = llvm.select %30, %34, %33 : i1, i64
      %36 = llvm.mul %35, %15  : i64
      %37 = llvm.add %36, %29  : i64
      %38 = llvm.getelementptr %arg6[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %39 = llvm.load %38 : !llvm.ptr -> f32
      %40 = llvm.getelementptr %arg1[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %41 = llvm.load %40 : !llvm.ptr -> f32
      %42 = llvm.fadd %39, %41  : f32
      %43 = llvm.intr.maxnum(%42, %12)  : (f32, f32) -> f32
      %44 = llvm.getelementptr %arg13[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %43, %44 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
}