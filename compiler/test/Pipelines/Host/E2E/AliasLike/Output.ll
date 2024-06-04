; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @Unknown0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26) {
  br label %28

28:                                               ; preds = %31, %27
  %29 = phi i64 [ %38, %31 ], [ 0, %27 ]
  %30 = icmp slt i64 %29, 25600
  br i1 %30, label %31, label %39

31:                                               ; preds = %28
  %32 = getelementptr float, ptr %1, i64 %29
  %33 = load float, ptr %32, align 4
  %34 = getelementptr float, ptr %10, i64 %29
  %35 = load float, ptr %34, align 4
  %36 = fadd float %33, %35
  %37 = getelementptr float, ptr %19, i64 %29
  store float %36, ptr %37, align 4
  %38 = add i64 %29, 1
  br label %28

39:                                               ; preds = %28
  ret void
}

define void @_mlir_ciface_Unknown0(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  %5 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 1
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 2
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 1
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 2
  %14 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %15 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 0
  %16 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 1
  %17 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 2
  %18 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 3, 0
  %19 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 3, 1
  %20 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 3, 2
  %21 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 4, 0
  %22 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 4, 1
  %23 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 4, 2
  %24 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %2, align 8
  %25 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 0
  %26 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 1
  %27 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 2
  %28 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 3, 0
  %29 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 3, 1
  %30 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 3, 2
  %31 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 4, 0
  %32 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 4, 1
  %33 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, 4, 2
  call void @Unknown0(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, ptr %25, ptr %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
