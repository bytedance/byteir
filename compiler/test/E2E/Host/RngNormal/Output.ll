; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @Unknown0(ptr %0, ptr %1, i64 %2, ptr %3, ptr %4, i64 %5, ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12) {
  br label %14

14:                                               ; preds = %17, %13
  %15 = phi i64 [ %178, %17 ], [ 0, %13 ]
  %16 = icmp slt i64 %15, 97
  br i1 %16, label %17, label %179

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
  %29 = zext i32 %28 to i64
  %30 = mul i64 %29, 3528531795
  %31 = trunc i64 %30 to i32
  %32 = lshr i64 %30, 32
  %33 = trunc i64 %32 to i32
  %34 = xor i32 %33, %21
  %35 = add i32 %20, -1640531527
  %36 = add i32 %21, -1150833019
  %37 = zext i32 %20 to i64
  %38 = zext i32 %34 to i64
  %39 = mul i64 %37, 3528531795
  %40 = trunc i64 %39 to i32
  %41 = lshr i64 %39, 32
  %42 = trunc i64 %41 to i32
  %43 = mul i64 %38, 3449720151
  %44 = trunc i64 %43 to i32
  %45 = lshr i64 %43, 32
  %46 = trunc i64 %45 to i32
  %47 = xor i32 %46, %35
  %48 = xor i32 %42, %31
  %49 = xor i32 %48, %36
  %50 = add i32 %20, 1013904242
  %51 = add i32 %21, 1993301258
  %52 = zext i32 %47 to i64
  %53 = zext i32 %49 to i64
  %54 = mul i64 %52, 3528531795
  %55 = trunc i64 %54 to i32
  %56 = lshr i64 %54, 32
  %57 = trunc i64 %56 to i32
  %58 = mul i64 %53, 3449720151
  %59 = trunc i64 %58 to i32
  %60 = lshr i64 %58, 32
  %61 = trunc i64 %60 to i32
  %62 = xor i32 %61, %44
  %63 = xor i32 %62, %50
  %64 = xor i32 %57, %40
  %65 = xor i32 %64, %51
  %66 = add i32 %20, -626627285
  %67 = add i32 %21, 842468239
  %68 = zext i32 %63 to i64
  %69 = zext i32 %65 to i64
  %70 = mul i64 %68, 3528531795
  %71 = trunc i64 %70 to i32
  %72 = lshr i64 %70, 32
  %73 = trunc i64 %72 to i32
  %74 = mul i64 %69, 3449720151
  %75 = trunc i64 %74 to i32
  %76 = lshr i64 %74, 32
  %77 = trunc i64 %76 to i32
  %78 = xor i32 %77, %59
  %79 = xor i32 %78, %66
  %80 = xor i32 %73, %55
  %81 = xor i32 %80, %67
  %82 = add i32 %20, 2027808484
  %83 = add i32 %21, -308364780
  %84 = zext i32 %79 to i64
  %85 = zext i32 %81 to i64
  %86 = mul i64 %84, 3528531795
  %87 = trunc i64 %86 to i32
  %88 = lshr i64 %86, 32
  %89 = trunc i64 %88 to i32
  %90 = mul i64 %85, 3449720151
  %91 = trunc i64 %90 to i32
  %92 = lshr i64 %90, 32
  %93 = trunc i64 %92 to i32
  %94 = xor i32 %93, %75
  %95 = xor i32 %94, %82
  %96 = xor i32 %89, %71
  %97 = xor i32 %96, %83
  %98 = add i32 %20, 387276957
  %99 = add i32 %21, -1459197799
  %100 = zext i32 %95 to i64
  %101 = zext i32 %97 to i64
  %102 = mul i64 %100, 3528531795
  %103 = trunc i64 %102 to i32
  %104 = lshr i64 %102, 32
  %105 = trunc i64 %104 to i32
  %106 = mul i64 %101, 3449720151
  %107 = trunc i64 %106 to i32
  %108 = lshr i64 %106, 32
  %109 = trunc i64 %108 to i32
  %110 = xor i32 %109, %91
  %111 = xor i32 %110, %98
  %112 = xor i32 %105, %87
  %113 = xor i32 %112, %99
  %114 = add i32 %20, -1253254570
  %115 = add i32 %21, 1684936478
  %116 = zext i32 %111 to i64
  %117 = zext i32 %113 to i64
  %118 = mul i64 %116, 3528531795
  %119 = trunc i64 %118 to i32
  %120 = lshr i64 %118, 32
  %121 = trunc i64 %120 to i32
  %122 = mul i64 %117, 3449720151
  %123 = trunc i64 %122 to i32
  %124 = lshr i64 %122, 32
  %125 = trunc i64 %124 to i32
  %126 = xor i32 %125, %107
  %127 = xor i32 %126, %114
  %128 = xor i32 %121, %103
  %129 = xor i32 %128, %115
  %130 = add i32 %20, 1401181199
  %131 = add i32 %21, 534103459
  %132 = zext i32 %127 to i64
  %133 = zext i32 %129 to i64
  %134 = mul i64 %132, 3528531795
  %135 = trunc i64 %134 to i32
  %136 = lshr i64 %134, 32
  %137 = trunc i64 %136 to i32
  %138 = mul i64 %133, 3449720151
  %139 = lshr i64 %138, 32
  %140 = trunc i64 %139 to i32
  %141 = xor i32 %140, %123
  %142 = xor i32 %141, %130
  %143 = xor i32 %137, %119
  %144 = xor i32 %143, %131
  %145 = add i32 %21, -616729560
  %146 = zext i32 %142 to i64
  %147 = zext i32 %144 to i64
  %148 = mul i64 %146, 3528531795
  %149 = lshr i64 %148, 32
  %150 = trunc i64 %149 to i32
  %151 = mul i64 %147, 3449720151
  %152 = trunc i64 %151 to i32
  %153 = xor i32 %150, %135
  %154 = xor i32 %153, %145
  %155 = add i32 %20, -1879881855
  %156 = zext i32 %154 to i64
  %157 = mul i64 %156, 3449720151
  %158 = trunc i64 %157 to i32
  %159 = lshr i64 %157, 32
  %160 = trunc i64 %159 to i32
  %161 = xor i32 %160, %152
  %162 = xor i32 %161, %155
  %163 = uitofp i32 %162 to float
  %164 = fmul float %163, 0x3DF0000000000000
  %165 = fadd float %164, 0x3DE0000000000000
  %166 = uitofp i32 %158 to float
  %167 = fmul float %166, 0x3DF0000000000000
  %168 = fadd float %167, 0x3DE0000000000000
  %169 = call float @llvm.log.f32(float %165)
  %170 = fmul float %169, -2.000000e+00
  %171 = call float @llvm.sqrt.f32(float %170)
  %172 = fmul float %168, 0x401921FB40000000
  %173 = call float @llvm.cos.f32(float %172)
  %174 = fmul float %171, %173
  %175 = fadd float %174, 0.000000e+00
  %176 = add i64 0, %15
  %177 = getelementptr float, ptr %7, i64 %176
  store float %175, ptr %177, align 4
  %178 = add i64 %15, 1
  br label %14

179:                                              ; preds = %14
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

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.log.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.cos.f32(float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
