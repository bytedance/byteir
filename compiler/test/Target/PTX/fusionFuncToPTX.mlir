// RUN: byteir-translate -gen-ptx -dump-ptx %s | FileCheck %s

module attributes {gpu.container_module}  {
  func.func @fusion_broadcast(%arg0: memref<6x12x96xf32>, %arg1: memref<6x12x96x96xf32>) -> memref<6x12x96x96xf32> {
    %0 = memref.alloc() : memref<6x12x96x96xf32>
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %1 = arith.subi %c6, %c0 : index
    %c12 = arith.constant 12 : index
    %2 = arith.subi %c12, %c0 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @fusion_broadcast_kernel::@fusion_broadcast_kernel blocks in (%1, %c1, %c1) threads in (%2, %c1, %c1) args(%arg1 : memref<6x12x96x96xf32>, %arg0 : memref<6x12x96xf32>, %0 : memref<6x12x96x96xf32>)
    return %0 : memref<6x12x96x96xf32>
  }
  gpu.module @fusion_broadcast_kernel {
    llvm.func @__nv_expf(f32) -> f32
    llvm.func @fusion_broadcast_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %8 = llvm.insertvalue %arg5, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %9 = llvm.insertvalue %arg9, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %10 = llvm.insertvalue %arg6, %9[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %11 = llvm.insertvalue %arg10, %10[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %13 = llvm.insertvalue %arg11, %12[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %14 = llvm.insertvalue %arg12, %13[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %15 = llvm.insertvalue %arg13, %14[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %16 = llvm.insertvalue %arg14, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %17 = llvm.insertvalue %arg17, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %18 = llvm.insertvalue %arg15, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %19 = llvm.insertvalue %arg18, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %20 = llvm.insertvalue %arg16, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %21 = llvm.insertvalue %arg19, %20[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %22 = llvm.insertvalue %arg20, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %23 = llvm.insertvalue %arg21, %22[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %24 = llvm.insertvalue %arg22, %23[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %25 = llvm.insertvalue %arg23, %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %26 = llvm.insertvalue %arg27, %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %27 = llvm.insertvalue %arg24, %26[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %28 = llvm.insertvalue %arg28, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %29 = llvm.insertvalue %arg25, %28[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %30 = llvm.insertvalue %arg29, %29[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %31 = llvm.insertvalue %arg26, %30[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %32 = llvm.insertvalue %arg30, %31[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %33 = llvm.mlir.constant(0 : index) : i64
      %34 = llvm.mlir.constant(96 : index) : i64
      %35 = llvm.mlir.constant(1 : index) : i64
      %36 = nvvm.read.ptx.sreg.ctaid.x : i32
      %37 = llvm.sext %36 : i32 to i64
      %38 = nvvm.read.ptx.sreg.tid.x : i32
      %39 = llvm.sext %38 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      llvm.br ^bb2(%33 : i64)
    ^bb2(%40: i64):  // 2 preds: ^bb1, ^bb6
      %41 = llvm.icmp "slt" %40, %34 : i64
      llvm.cond_br %41, ^bb3, ^bb7
    ^bb3:  // pred: ^bb2
      llvm.br ^bb4(%33 : i64)
    ^bb4(%42: i64):  // 2 preds: ^bb3, ^bb5
      %43 = llvm.icmp "slt" %42, %34 : i64
      llvm.cond_br %43, ^bb5, ^bb6
    ^bb5:  // pred: ^bb4
      %44 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %45 = llvm.mlir.constant(110592 : index) : i64
      %46 = llvm.mul %37, %45  : i64
      %47 = llvm.mlir.constant(9216 : index) : i64
      %48 = llvm.mul %39, %47  : i64
      %49 = llvm.add %46, %48  : i64
      %50 = llvm.mul %40, %34  : i64
      %51 = llvm.add %49, %50  : i64
      %52 = llvm.add %51, %42  : i64
      %53 = llvm.getelementptr %44[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %54 = llvm.load %53 : !llvm.ptr -> f32
      %55 = llvm.extractvalue %21[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %56 = llvm.mlir.constant(1152 : index) : i64
      %57 = llvm.mul %37, %56  : i64
      %58 = llvm.mul %39, %34  : i64
      %59 = llvm.add %57, %58  : i64
      %60 = llvm.add %59, %40  : i64
      %61 = llvm.getelementptr %55[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %62 = llvm.load %61 : !llvm.ptr -> f32
      %63 = llvm.fsub %54, %62  : f32
      %64 = llvm.call @__nv_expf(%63) : (f32) -> f32
      %65 = llvm.extractvalue %32[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %66 = llvm.getelementptr %65[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %64, %66 : f32, !llvm.ptr
      %67 = llvm.add %42, %35  : i64
      llvm.br ^bb4(%67 : i64)
    ^bb6:  // pred: ^bb4
      %68 = llvm.add %40, %35  : i64
      llvm.br ^bb2(%68 : i64)
    ^bb7:  // pred: ^bb2
      llvm.return
    }
  }
}

// CHECK-LABEL: .visible .entry fusion_broadcast_kernel
// CHECK: ex2.approx.ftz.f32