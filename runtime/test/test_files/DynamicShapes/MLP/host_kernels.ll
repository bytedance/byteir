; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define { i64, i64, i64, i64 } @shapeComputaionFunc_0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = mul i64 %3, 20
  %9 = add i64 %8, -1
  %10 = sdiv i64 %9, 1024
  %11 = add i64 %10, 1
  %12 = sub i64 0, %8
  %13 = sdiv i64 %12, 1024
  %14 = sub i64 0, %13
  %15 = icmp sgt i64 %8, 0
  %16 = select i1 %15, i64 %11, i64 %14
  %17 = call i64 @llvm.smax.i64(i64 %16, i64 1)
  %18 = insertvalue { i64, i64, i64, i64 } { i64 1, i64 undef, i64 undef, i64 undef }, i64 %3, 1
  %19 = insertvalue { i64, i64, i64, i64 } %18, i64 %17, 2
  %20 = insertvalue { i64, i64, i64, i64 } %19, i64 256, 3
  ret { i64, i64, i64, i64 } %20
}

define void @_mlir_ciface_shapeComputaionFunc_0(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 1
  %11 = call { i64, i64, i64, i64 } @shapeComputaionFunc_0(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10)
  store { i64, i64, i64, i64 } %11, ptr %0, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}