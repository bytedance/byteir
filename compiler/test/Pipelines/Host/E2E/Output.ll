; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_1x128xi32 = private constant [1 x [128 x i32]] [[128 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127]]

declare ptr @malloc(i64)

declare void @free(ptr)

define { { ptr, ptr, i64, [2 x i64], [2 x i64] }, { ptr, ptr, i64, [2 x i64], [2 x i64] } } @Unknown1(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21) {
  %23 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 128) to i64))
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %23, 0
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, ptr %23, 1
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 0, 2
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 1, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 128, 3, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 128, 4, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 1, 4, 1
  %31 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 128) to i64))
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %31, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, ptr %31, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 0, 2
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 1, 3, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 128, 3, 1
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 128, 4, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 1, 4, 1
  br label %39

39:                                               ; preds = %42, %22
  %40 = phi i64 [ %59, %42 ], [ 0, %22 ]
  %41 = icmp slt i64 %40, 128
  br i1 %41, label %42, label %60

42:                                               ; preds = %39
  %43 = icmp slt i64 %40, 0
  %44 = add i64 %40, 128
  %45 = select i1 %43, i64 %44, i64 %40
  %46 = add i64 0, %45
  %47 = getelementptr i32, ptr @__constant_1x128xi32, i64 %46
  %48 = load i32, ptr %47, align 4
  %49 = load i64, ptr %11, align 4
  %50 = load i64, ptr %1, align 4
  %51 = load i64, ptr %6, align 4
  %52 = add i64 %50, %51
  %53 = add i64 %49, %52
  %54 = trunc i64 %53 to i32
  %55 = icmp slt i32 %48, %54
  %56 = zext i1 %55 to i32
  %57 = add i64 0, %45
  %58 = getelementptr i32, ptr %31, i64 %57
  store i32 %56, ptr %58, align 4
  %59 = add i64 %40, 1
  br label %39

60:                                               ; preds = %63, %39
  %61 = phi i64 [ %76, %63 ], [ 0, %39 ]
  %62 = icmp slt i64 %61, 128
  br i1 %62, label %63, label %77

63:                                               ; preds = %60
  %64 = icmp slt i64 %61, 0
  %65 = add i64 %61, 128
  %66 = select i1 %64, i64 %65, i64 %61
  %67 = add i64 0, %66
  %68 = getelementptr i32, ptr %31, i64 %67
  %69 = load i32, ptr %68, align 4
  %70 = add i64 0, %66
  %71 = getelementptr i32, ptr %16, i64 %70
  %72 = load i32, ptr %71, align 4
  %73 = mul i32 %69, %72
  %74 = add i64 0, %66
  %75 = getelementptr i32, ptr %23, i64 %74
  store i32 %73, ptr %75, align 4
  %76 = add i64 %61, 1
  br label %60

77:                                               ; preds = %60
  %78 = insertvalue { { ptr, ptr, i64, [2 x i64], [2 x i64] }, { ptr, ptr, i64, [2 x i64], [2 x i64] } } undef, { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 0
  %79 = insertvalue { { ptr, ptr, i64, [2 x i64], [2 x i64] }, { ptr, ptr, i64, [2 x i64], [2 x i64] } } %78, { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  ret { { ptr, ptr, i64, [2 x i64], [2 x i64] }, { ptr, ptr, i64, [2 x i64], [2 x i64] } } %79
}

define void @_mlir_ciface_Unknown1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
  %6 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 2
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 3, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 4, 0
  %12 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 0
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 2
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 4, 0
  %18 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 0
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 1
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 2
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 3, 0
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 4, 0
  %24 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %4, align 8
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 0
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 2
  %28 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 3, 0
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 3, 1
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 4, 0
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 4, 1
  %32 = call { { ptr, ptr, i64, [2 x i64], [2 x i64] }, { ptr, ptr, i64, [2 x i64], [2 x i64] } } @Unknown1(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23, ptr %25, ptr %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31)
  store { { ptr, ptr, i64, [2 x i64], [2 x i64] }, { ptr, ptr, i64, [2 x i64], [2 x i64] } } %32, ptr %0, align 8
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
