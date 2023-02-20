; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_1x128xi32 = private constant [1 x [128 x i32]] [[128 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127]]

declare ptr @malloc(i64)

declare void @free(ptr)

define void @Unknown1(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, ptr %22, ptr %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, ptr %29, ptr %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35) {
  br label %37

37:                                               ; preds = %40, %36
  %38 = phi i64 [ %57, %40 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 128
  br i1 %39, label %40, label %58

40:                                               ; preds = %37
  %41 = icmp slt i64 %38, 0
  %42 = add i64 %38, 128
  %43 = select i1 %41, i64 %42, i64 %38
  %44 = add i64 0, %43
  %45 = getelementptr i32, ptr @__constant_1x128xi32, i64 %44
  %46 = load i32, ptr %45, align 4
  %47 = load i64, ptr %11, align 4
  %48 = load i64, ptr %1, align 4
  %49 = load i64, ptr %6, align 4
  %50 = add i64 %48, %49
  %51 = add i64 %47, %50
  %52 = trunc i64 %51 to i32
  %53 = icmp slt i32 %46, %52
  %54 = zext i1 %53 to i32
  %55 = add i64 0, %43
  %56 = getelementptr i32, ptr %23, i64 %55
  store i32 %54, ptr %56, align 4
  %57 = add i64 %38, 1
  br label %37

58:                                               ; preds = %61, %37
  %59 = phi i64 [ %74, %61 ], [ 0, %37 ]
  %60 = icmp slt i64 %59, 128
  br i1 %60, label %61, label %75

61:                                               ; preds = %58
  %62 = icmp slt i64 %59, 0
  %63 = add i64 %59, 128
  %64 = select i1 %62, i64 %63, i64 %59
  %65 = add i64 0, %64
  %66 = getelementptr i32, ptr %23, i64 %65
  %67 = load i32, ptr %66, align 4
  %68 = add i64 0, %64
  %69 = getelementptr i32, ptr %16, i64 %68
  %70 = load i32, ptr %69, align 4
  %71 = mul i32 %67, %70
  %72 = add i64 0, %64
  %73 = getelementptr i32, ptr %30, i64 %72
  store i32 %71, ptr %73, align 4
  %74 = add i64 %59, 1
  br label %58

75:                                               ; preds = %58
  ret void
}

define void @_mlir_ciface_Unknown1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5) {
  %7 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 0
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 2
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 3, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 4, 0
  %13 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 2
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 3, 0
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 4, 0
  %19 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 0
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 1
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 2
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 3, 0
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 4, 0
  %25 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %3, align 8
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 0
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 1
  %28 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 2
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3, 0
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3, 1
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 4, 0
  %32 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 4, 1
  %33 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %4, align 8
  %34 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 0
  %35 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 1
  %36 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 2
  %37 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 3, 0
  %38 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 3, 1
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 4, 0
  %40 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 4, 1
  %41 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %5, align 8
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 0
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %44 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 2
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 3, 0
  %46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 3, 1
  %47 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 4, 0
  %48 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 4, 1
  call void @Unknown1(ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, ptr %34, ptr %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40, ptr %42, ptr %43, i64 %44, i64 %45, i64 %46, i64 %47, i64 %48)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
