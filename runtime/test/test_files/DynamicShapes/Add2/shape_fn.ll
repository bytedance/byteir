; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define { i64, i64 } @shape_fn(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8) !dbg !3 {
  %10 = insertvalue { i64, i64 } undef, i64 %3, 0, !dbg !7
  %11 = insertvalue { i64, i64 } %10, i64 %4, 1, !dbg !9
  ret { i64, i64 } %11, !dbg !10
}

define void @_mlir_ciface_shape_fn(ptr %0, ptr %1) !dbg !11 {
  %3 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8, !dbg !12
  %4 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 0, !dbg !14
  %5 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 1, !dbg !15
  %6 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 2, !dbg !16
  %7 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 3, 0, !dbg !17
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 3, 1, !dbg !18
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 3, 2, !dbg !19
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 4, 0, !dbg !20
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 4, 1, !dbg !21
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 4, 2, !dbg !22
  %13 = call { i64, i64 } @shape_fn(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12), !dbg !23
  store { i64, i64 } %13, ptr %0, align 4, !dbg !24
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "shape_fn", linkageName: "shape_fn", scope: null, file: !4, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "<stdin>", directory: "/root/workspace/byteir")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, column: 10, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 5, column: 10, scope: !8)
!10 = !DILocation(line: 6, column: 5, scope: !8)
!11 = distinct !DISubprogram(name: "_mlir_ciface_shape_fn", linkageName: "_mlir_ciface_shape_fn", scope: null, file: !4, line: 8, type: !5, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!12 = !DILocation(line: 9, column: 10, scope: !13)
!13 = !DILexicalBlockFile(scope: !11, file: !4, discriminator: 0)
!14 = !DILocation(line: 10, column: 10, scope: !13)
!15 = !DILocation(line: 11, column: 10, scope: !13)
!16 = !DILocation(line: 12, column: 10, scope: !13)
!17 = !DILocation(line: 13, column: 10, scope: !13)
!18 = !DILocation(line: 14, column: 10, scope: !13)
!19 = !DILocation(line: 15, column: 10, scope: !13)
!20 = !DILocation(line: 16, column: 10, scope: !13)
!21 = !DILocation(line: 17, column: 10, scope: !13)
!22 = !DILocation(line: 18, column: 10, scope: !13)
!23 = !DILocation(line: 19, column: 11, scope: !13)
!24 = !DILocation(line: 20, column: 5, scope: !13)
!25 = !DILocation(line: 21, column: 5, scope: !13)
