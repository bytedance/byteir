; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

; Function Attrs: memory(none)
declare float @tanhf(float) #0

define void @Unknown0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
  br label %11

11:                                               ; preds = %14, %10
  %12 = phi i64 [ %19, %14 ], [ 0, %10 ]
  %13 = icmp slt i64 %12, 32
  br i1 %13, label %14, label %20

14:                                               ; preds = %11
  %15 = getelementptr float, ptr %1, i64 %12
  %16 = load float, ptr %15, align 4
  %17 = call float @tanhf(float %16)
  %18 = getelementptr float, ptr %6, i64 %12
  store float %17, ptr %18, align 4
  %19 = add i64 %12, 1
  br label %11

20:                                               ; preds = %11
  ret void
}

define void @_mlir_ciface_Unknown0(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %4 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 4, 0
  %9 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, 1
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, 2
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, 3, 0
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, 4, 0
  call void @Unknown0(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14)
  ret void
}

attributes #0 = { memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
