; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @print()

declare void @llvm.memset.p0i32.i32(i32*, i8, i32, i1)

define void @add(i32* %0, i32* %1, i32* %2, i32 %3) {
  call void @llvm.memset.p0i32.i32(i32* %0, i8 0, i32 %3, i1 false)
  %5 = sext i32 %3 to i64
  br label %6

6:                                                ; preds = %9, %4
  %7 = phi i64 [ %16, %9 ], [ 0, %4 ]
  %8 = icmp slt i64 %7, %5
  br i1 %8, label %9, label %17

9:                                                ; preds = %6
  %10 = getelementptr i32, i32* %0, i64 %7
  %11 = load i32, i32* %10, align 4
  %12 = getelementptr i32, i32* %1, i64 %7
  %13 = load i32, i32* %12, align 4
  %14 = add i32 %11, %13
  %15 = getelementptr i32, i32* %2, i64 %7
  store i32 %14, i32* %15, align 4
  %16 = add i64 %7, 1
  br label %6

17:                                               ; preds = %6
  call void @print()
  ret void
}
