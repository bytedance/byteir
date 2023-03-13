; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @Unknown0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21) {
  br label %23

23:                                               ; preds = %26, %22
  %24 = phi i64 [ %31, %26 ], [ 0, %22 ]
  %25 = icmp slt i64 %24, 150528
  br i1 %25, label %26, label %32

26:                                               ; preds = %23
  %27 = getelementptr float, ptr %1, i64 %24
  %28 = load float, ptr %27, align 4
  %29 = fptrunc float %28 to half
  %30 = getelementptr half, ptr %12, i64 %24
  store half %29, ptr %30, align 2
  %31 = add i64 %24, 1
  br label %23

32:                                               ; preds = %23
  ret void
}

define void @_mlir_ciface_Unknown0(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %0, align 8
  %4 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 2
  %10 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 3
  %11 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 1
  %13 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 2
  %14 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 3
  %15 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %1, align 8
  %16 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 0
  %17 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 1
  %18 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 2
  %19 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 3, 0
  %20 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 3, 1
  %21 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 3, 2
  %22 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 3, 3
  %23 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 4, 0
  %24 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 4, 1
  %25 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 4, 2
  %26 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 4, 3
  call void @Unknown0(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
