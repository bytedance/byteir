; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @Unknown0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21) {
  br label %23

23:                                               ; preds = %26, %22
  %24 = phi i64 [ %179, %26 ], [ 0, %22 ]
  %25 = icmp slt i64 %24, 2048
  br i1 %25, label %26, label %180

26:                                               ; preds = %23
  %27 = srem i64 %24, 4
  %28 = sdiv i64 %24, 4
  %29 = srem i64 %28, 8
  %30 = sdiv i64 %28, 8
  %31 = mul i64 %27, 8
  %32 = mul i64 %29, 8
  %33 = mul i64 %31, 4096
  %34 = add i64 0, %33
  %35 = mul i64 %30, 64
  %36 = add i64 %34, %35
  %37 = add i64 %36, %32
  %38 = getelementptr float, ptr %1, i64 %37
  %39 = load <8 x float>, ptr %38, align 4
  %40 = add i64 %31, 1
  %41 = mul i64 %40, 4096
  %42 = add i64 0, %41
  %43 = mul i64 %30, 64
  %44 = add i64 %42, %43
  %45 = add i64 %44, %32
  %46 = getelementptr float, ptr %1, i64 %45
  %47 = load <8 x float>, ptr %46, align 4
  %48 = add i64 %31, 2
  %49 = mul i64 %48, 4096
  %50 = add i64 0, %49
  %51 = mul i64 %30, 64
  %52 = add i64 %50, %51
  %53 = add i64 %52, %32
  %54 = getelementptr float, ptr %1, i64 %53
  %55 = load <8 x float>, ptr %54, align 4
  %56 = add i64 %31, 3
  %57 = mul i64 %56, 4096
  %58 = add i64 0, %57
  %59 = mul i64 %30, 64
  %60 = add i64 %58, %59
  %61 = add i64 %60, %32
  %62 = getelementptr float, ptr %1, i64 %61
  %63 = load <8 x float>, ptr %62, align 4
  %64 = add i64 %31, 4
  %65 = mul i64 %64, 4096
  %66 = add i64 0, %65
  %67 = mul i64 %30, 64
  %68 = add i64 %66, %67
  %69 = add i64 %68, %32
  %70 = getelementptr float, ptr %1, i64 %69
  %71 = load <8 x float>, ptr %70, align 4
  %72 = add i64 %31, 5
  %73 = mul i64 %72, 4096
  %74 = add i64 0, %73
  %75 = mul i64 %30, 64
  %76 = add i64 %74, %75
  %77 = add i64 %76, %32
  %78 = getelementptr float, ptr %1, i64 %77
  %79 = load <8 x float>, ptr %78, align 4
  %80 = add i64 %31, 6
  %81 = mul i64 %80, 4096
  %82 = add i64 0, %81
  %83 = mul i64 %30, 64
  %84 = add i64 %82, %83
  %85 = add i64 %84, %32
  %86 = getelementptr float, ptr %1, i64 %85
  %87 = load <8 x float>, ptr %86, align 4
  %88 = add i64 %31, 7
  %89 = mul i64 %88, 4096
  %90 = add i64 0, %89
  %91 = mul i64 %30, 64
  %92 = add i64 %90, %91
  %93 = add i64 %92, %32
  %94 = getelementptr float, ptr %1, i64 %93
  %95 = load <8 x float>, ptr %94, align 4
  %96 = shufflevector <8 x float> %39, <8 x float> %47, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  %97 = shufflevector <8 x float> %39, <8 x float> %47, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  %98 = shufflevector <8 x float> %55, <8 x float> %63, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  %99 = shufflevector <8 x float> %55, <8 x float> %63, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  %100 = shufflevector <8 x float> %71, <8 x float> %79, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  %101 = shufflevector <8 x float> %71, <8 x float> %79, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  %102 = shufflevector <8 x float> %87, <8 x float> %95, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  %103 = shufflevector <8 x float> %87, <8 x float> %95, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  %104 = shufflevector <8 x float> %96, <8 x float> %98, <8 x i32> <i32 2, i32 3, i32 8, i32 9, i32 6, i32 7, i32 12, i32 13>
  %105 = shufflevector <8 x float> %97, <8 x float> %99, <8 x i32> <i32 2, i32 3, i32 8, i32 9, i32 6, i32 7, i32 12, i32 13>
  %106 = shufflevector <8 x float> %100, <8 x float> %102, <8 x i32> <i32 2, i32 3, i32 8, i32 9, i32 6, i32 7, i32 12, i32 13>
  %107 = shufflevector <8 x float> %101, <8 x float> %103, <8 x i32> <i32 2, i32 3, i32 8, i32 9, i32 6, i32 7, i32 12, i32 13>
  %108 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0xcc", "=x,x,x"(<8 x float> %96, <8 x float> %104)
  %109 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0x33", "=x,x,x"(<8 x float> %98, <8 x float> %104)
  %110 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0xcc", "=x,x,x"(<8 x float> %97, <8 x float> %105)
  %111 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0x33", "=x,x,x"(<8 x float> %99, <8 x float> %105)
  %112 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0xcc", "=x,x,x"(<8 x float> %100, <8 x float> %106)
  %113 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0x33", "=x,x,x"(<8 x float> %102, <8 x float> %106)
  %114 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0xcc", "=x,x,x"(<8 x float> %101, <8 x float> %107)
  %115 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0x33", "=x,x,x"(<8 x float> %103, <8 x float> %107)
  %116 = shufflevector <8 x float> %108, <8 x float> %112, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %117 = shufflevector <8 x float> %109, <8 x float> %113, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %118 = shufflevector <8 x float> %110, <8 x float> %114, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %119 = shufflevector <8 x float> %111, <8 x float> %115, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %120 = shufflevector <8 x float> %108, <8 x float> %112, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %121 = shufflevector <8 x float> %109, <8 x float> %113, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %122 = shufflevector <8 x float> %110, <8 x float> %114, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %123 = shufflevector <8 x float> %111, <8 x float> %115, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %124 = mul i64 %30, 2048
  %125 = add i64 0, %124
  %126 = mul i64 %32, 32
  %127 = add i64 %125, %126
  %128 = add i64 %127, %31
  %129 = getelementptr float, ptr %12, i64 %128
  store <8 x float> %116, ptr %129, align 4
  %130 = add i64 %32, 1
  %131 = mul i64 %30, 2048
  %132 = add i64 0, %131
  %133 = mul i64 %130, 32
  %134 = add i64 %132, %133
  %135 = add i64 %134, %31
  %136 = getelementptr float, ptr %12, i64 %135
  store <8 x float> %117, ptr %136, align 4
  %137 = add i64 %32, 2
  %138 = mul i64 %30, 2048
  %139 = add i64 0, %138
  %140 = mul i64 %137, 32
  %141 = add i64 %139, %140
  %142 = add i64 %141, %31
  %143 = getelementptr float, ptr %12, i64 %142
  store <8 x float> %118, ptr %143, align 4
  %144 = add i64 %32, 3
  %145 = mul i64 %30, 2048
  %146 = add i64 0, %145
  %147 = mul i64 %144, 32
  %148 = add i64 %146, %147
  %149 = add i64 %148, %31
  %150 = getelementptr float, ptr %12, i64 %149
  store <8 x float> %119, ptr %150, align 4
  %151 = add i64 %32, 4
  %152 = mul i64 %30, 2048
  %153 = add i64 0, %152
  %154 = mul i64 %151, 32
  %155 = add i64 %153, %154
  %156 = add i64 %155, %31
  %157 = getelementptr float, ptr %12, i64 %156
  store <8 x float> %120, ptr %157, align 4
  %158 = add i64 %32, 5
  %159 = mul i64 %30, 2048
  %160 = add i64 0, %159
  %161 = mul i64 %158, 32
  %162 = add i64 %160, %161
  %163 = add i64 %162, %31
  %164 = getelementptr float, ptr %12, i64 %163
  store <8 x float> %121, ptr %164, align 4
  %165 = add i64 %32, 6
  %166 = mul i64 %30, 2048
  %167 = add i64 0, %166
  %168 = mul i64 %165, 32
  %169 = add i64 %167, %168
  %170 = add i64 %169, %31
  %171 = getelementptr float, ptr %12, i64 %170
  store <8 x float> %122, ptr %171, align 4
  %172 = add i64 %32, 7
  %173 = mul i64 %30, 2048
  %174 = add i64 0, %173
  %175 = mul i64 %172, 32
  %176 = add i64 %174, %175
  %177 = add i64 %176, %31
  %178 = getelementptr float, ptr %12, i64 %177
  store <8 x float> %123, ptr %178, align 4
  %179 = add i64 %24, 1
  br label %23

180:                                              ; preds = %23
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
