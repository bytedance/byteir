; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @Unknown0(ptr %0, ptr %1, i64 %2, ptr %3, ptr %4, i64 %5, ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12) {
  br label %14

14:                                               ; preds = %17, %13
  %15 = phi i64 [ %34, %17 ], [ 0, %13 ]
  %16 = icmp slt i64 %15, 97
  br i1 %16, label %17, label %35

17:                                               ; preds = %14
  %18 = load i64, ptr %1, align 4
  %19 = load i64, ptr %4, align 4
  %20 = trunc i64 %18 to i32
  %21 = trunc i64 %19 to i32
  %22 = add i32 %20, %21
  %23 = mul i32 %22, 1103515245
  %24 = add i32 %23, 12345
  %25 = trunc i64 %15 to i32
  %26 = add i32 %25, %24
  %27 = mul i32 %26, 1103515245
  %28 = add i32 %27, 12345
  %29 = uitofp i32 %28 to float
  %30 = fmul float %29, 0x3DF0000000000000
  %31 = fadd float %30, 0.000000e+00
  %32 = add i64 0, %15
  %33 = getelementptr float, ptr %7, i64 %32
  store float %31, ptr %33, align 4
  %34 = add i64 %15, 1
  br label %14

35:                                               ; preds = %14
  ret void
}

define void @_mlir_ciface_Unknown0(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64 }, ptr %0, align 8
  %5 = extractvalue { ptr, ptr, i64 } %4, 0
  %6 = extractvalue { ptr, ptr, i64 } %4, 1
  %7 = extractvalue { ptr, ptr, i64 } %4, 2
  %8 = load { ptr, ptr, i64 }, ptr %1, align 8
  %9 = extractvalue { ptr, ptr, i64 } %8, 0
  %10 = extractvalue { ptr, ptr, i64 } %8, 1
  %11 = extractvalue { ptr, ptr, i64 } %8, 2
  %12 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 0
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 1
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 2
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 1
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 0
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 1
  call void @Unknown0(ptr %5, ptr %6, i64 %7, ptr %9, ptr %10, i64 %11, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
