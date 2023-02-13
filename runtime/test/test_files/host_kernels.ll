; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_1x128xi32 = private constant [1 x [128 x i32]] [[128 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127]]

declare ptr @malloc(i64)

declare void @free(ptr)

define void @Unknown1(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25) !dbg !3 {
  br label %27, !dbg !7

27:                                               ; preds = %30, %26
  %28 = phi i64 [ %50, %30 ], [ 0, %26 ]
  %29 = icmp slt i64 %28, 128, !dbg !9
  br i1 %29, label %30, label %51, !dbg !10

30:                                               ; preds = %27
  %31 = icmp slt i64 %28, 0, !dbg !11
  %32 = add i64 %28, 128, !dbg !12
  %33 = select i1 %31, i64 %32, i64 %28, !dbg !13
  %34 = sub i64 -1, %28, !dbg !14
  %35 = select i1 %31, i64 %34, i64 %28, !dbg !15
  %36 = sdiv i64 %35, 128, !dbg !16
  %37 = sub i64 -1, %36, !dbg !17
  %38 = select i1 %31, i64 %37, i64 %36, !dbg !18
  %39 = mul i64 %38, 128, !dbg !19
  %40 = add i64 %39, %33, !dbg !20
  %41 = getelementptr i32, ptr @__constant_1x128xi32, i64 %40, !dbg !21
  %42 = load i32, ptr %41, align 4, !dbg !22
  %43 = getelementptr i32, ptr %1, i64 %38, !dbg !23
  %44 = load i32, ptr %43, align 4, !dbg !24
  %45 = icmp slt i32 %42, %44, !dbg !25
  %46 = zext i1 %45 to i32, !dbg !26
  %47 = mul i64 %38, 128, !dbg !27
  %48 = add i64 %47, %33, !dbg !28
  %49 = getelementptr i32, ptr %13, i64 %48, !dbg !29
  store i32 %46, ptr %49, align 4, !dbg !30
  %50 = add i64 %28, 1, !dbg !31
  br label %27, !dbg !32

51:                                               ; preds = %54, %27
  %52 = phi i64 [ %75, %54 ], [ 0, %27 ]
  %53 = icmp slt i64 %52, 128, !dbg !33
  br i1 %53, label %54, label %76, !dbg !34

54:                                               ; preds = %51
  %55 = icmp slt i64 %52, 0, !dbg !35
  %56 = add i64 %52, 128, !dbg !36
  %57 = select i1 %55, i64 %56, i64 %52, !dbg !37
  %58 = sub i64 -1, %52, !dbg !38
  %59 = select i1 %55, i64 %58, i64 %52, !dbg !39
  %60 = sdiv i64 %59, 128, !dbg !40
  %61 = sub i64 -1, %60, !dbg !41
  %62 = select i1 %55, i64 %61, i64 %60, !dbg !42
  %63 = mul i64 %62, 128, !dbg !43
  %64 = add i64 %63, %57, !dbg !44
  %65 = getelementptr i32, ptr %13, i64 %64, !dbg !45
  %66 = load i32, ptr %65, align 4, !dbg !46
  %67 = mul i64 %62, 128, !dbg !47
  %68 = add i64 %67, %57, !dbg !48
  %69 = getelementptr i32, ptr %6, i64 %68, !dbg !49
  %70 = load i32, ptr %69, align 4, !dbg !50
  %71 = mul i32 %66, %70, !dbg !51
  %72 = mul i64 %62, 128, !dbg !52
  %73 = add i64 %72, %57, !dbg !53
  %74 = getelementptr i32, ptr %20, i64 %73, !dbg !54
  store i32 %71, ptr %74, align 4, !dbg !55
  %75 = add i64 %52, 1, !dbg !56
  br label %51, !dbg !57

76:                                               ; preds = %51
  ret void, !dbg !58
}

define void @_mlir_ciface_Unknown1(ptr %0, ptr %1, ptr %2, ptr %3) !dbg !59 {
  %5 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8, !dbg !60
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 0, !dbg !62
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1, !dbg !63
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 2, !dbg !64
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 3, 0, !dbg !65
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 4, 0, !dbg !66
  %11 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8, !dbg !67
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 0, !dbg !68
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 1, !dbg !69
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 2, !dbg !70
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 3, 0, !dbg !71
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 3, 1, !dbg !72
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 4, 0, !dbg !73
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 4, 1, !dbg !74
  %19 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8, !dbg !75
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 0, !dbg !76
  %21 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 1, !dbg !77
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 2, !dbg !78
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 3, 0, !dbg !79
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 3, 1, !dbg !80
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 4, 0, !dbg !81
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 4, 1, !dbg !82
  %27 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %3, align 8, !dbg !83
  %28 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 0, !dbg !84
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 1, !dbg !85
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 2, !dbg !86
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 3, 0, !dbg !87
  %32 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 3, 1, !dbg !88
  %33 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 4, 0, !dbg !89
  %34 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 4, 1, !dbg !90
  call void @Unknown1(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34), !dbg !91
  ret void, !dbg !92
}

define void @Unknown0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19) !dbg !93 {
  %21 = load i64, ptr %11, align 4, !dbg !94
  %22 = load i64, ptr %1, align 4, !dbg !96
  %23 = load i64, ptr %6, align 4, !dbg !97
  %24 = add i64 %22, %23, !dbg !98
  %25 = add i64 %21, %24, !dbg !99
  %26 = trunc i64 %25 to i32, !dbg !100
  store i32 %26, ptr %16, align 4, !dbg !101
  ret void, !dbg !102
}

define void @_mlir_ciface_Unknown0(ptr %0, ptr %1, ptr %2, ptr %3) !dbg !103 {
  %5 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8, !dbg !104
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 0, !dbg !106
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1, !dbg !107
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 2, !dbg !108
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 3, 0, !dbg !109
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 4, 0, !dbg !110
  %11 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8, !dbg !111
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 0, !dbg !112
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1, !dbg !113
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 2, !dbg !114
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3, 0, !dbg !115
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 4, 0, !dbg !116
  %17 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8, !dbg !117
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 0, !dbg !118
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 1, !dbg !119
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 2, !dbg !120
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 3, 0, !dbg !121
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 4, 0, !dbg !122
  %23 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8, !dbg !123
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0, !dbg !124
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1, !dbg !125
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2, !dbg !126
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0, !dbg !127
  %28 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0, !dbg !128
  call void @Unknown0(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, ptr %24, ptr %25, i64 %26, i64 %27, i64 %28), !dbg !129
  ret void, !dbg !130
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Unknown1", linkageName: "Unknown1", scope: null, file: !4, line: 7, type: !5, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "../test/Pipelines/Host/ToLLVMIR.mlir", directory: "/root/workspace/byteir/build")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 14, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 16, column: 10, scope: !8)
!10 = !DILocation(line: 17, column: 5, scope: !8)
!11 = !DILocation(line: 19, column: 10, scope: !8)
!12 = !DILocation(line: 20, column: 10, scope: !8)
!13 = !DILocation(line: 21, column: 11, scope: !8)
!14 = !DILocation(line: 22, column: 11, scope: !8)
!15 = !DILocation(line: 23, column: 11, scope: !8)
!16 = !DILocation(line: 24, column: 11, scope: !8)
!17 = !DILocation(line: 25, column: 11, scope: !8)
!18 = !DILocation(line: 26, column: 11, scope: !8)
!19 = !DILocation(line: 27, column: 11, scope: !8)
!20 = !DILocation(line: 28, column: 11, scope: !8)
!21 = !DILocation(line: 29, column: 11, scope: !8)
!22 = !DILocation(line: 30, column: 11, scope: !8)
!23 = !DILocation(line: 31, column: 11, scope: !8)
!24 = !DILocation(line: 32, column: 11, scope: !8)
!25 = !DILocation(line: 33, column: 11, scope: !8)
!26 = !DILocation(line: 34, column: 11, scope: !8)
!27 = !DILocation(line: 35, column: 11, scope: !8)
!28 = !DILocation(line: 36, column: 11, scope: !8)
!29 = !DILocation(line: 37, column: 11, scope: !8)
!30 = !DILocation(line: 38, column: 5, scope: !8)
!31 = !DILocation(line: 39, column: 11, scope: !8)
!32 = !DILocation(line: 40, column: 5, scope: !8)
!33 = !DILocation(line: 42, column: 11, scope: !8)
!34 = !DILocation(line: 43, column: 5, scope: !8)
!35 = !DILocation(line: 45, column: 11, scope: !8)
!36 = !DILocation(line: 46, column: 11, scope: !8)
!37 = !DILocation(line: 47, column: 11, scope: !8)
!38 = !DILocation(line: 48, column: 11, scope: !8)
!39 = !DILocation(line: 49, column: 11, scope: !8)
!40 = !DILocation(line: 50, column: 11, scope: !8)
!41 = !DILocation(line: 51, column: 11, scope: !8)
!42 = !DILocation(line: 52, column: 11, scope: !8)
!43 = !DILocation(line: 53, column: 11, scope: !8)
!44 = !DILocation(line: 54, column: 11, scope: !8)
!45 = !DILocation(line: 55, column: 11, scope: !8)
!46 = !DILocation(line: 56, column: 11, scope: !8)
!47 = !DILocation(line: 57, column: 11, scope: !8)
!48 = !DILocation(line: 58, column: 11, scope: !8)
!49 = !DILocation(line: 59, column: 11, scope: !8)
!50 = !DILocation(line: 60, column: 11, scope: !8)
!51 = !DILocation(line: 61, column: 11, scope: !8)
!52 = !DILocation(line: 62, column: 11, scope: !8)
!53 = !DILocation(line: 63, column: 11, scope: !8)
!54 = !DILocation(line: 64, column: 11, scope: !8)
!55 = !DILocation(line: 65, column: 5, scope: !8)
!56 = !DILocation(line: 66, column: 11, scope: !8)
!57 = !DILocation(line: 67, column: 5, scope: !8)
!58 = !DILocation(line: 69, column: 5, scope: !8)
!59 = distinct !DISubprogram(name: "_mlir_ciface_Unknown1", linkageName: "_mlir_ciface_Unknown1", scope: null, file: !4, line: 71, type: !5, scopeLine: 71, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!60 = !DILocation(line: 72, column: 10, scope: !61)
!61 = !DILexicalBlockFile(scope: !59, file: !4, discriminator: 0)
!62 = !DILocation(line: 73, column: 10, scope: !61)
!63 = !DILocation(line: 74, column: 10, scope: !61)
!64 = !DILocation(line: 75, column: 10, scope: !61)
!65 = !DILocation(line: 76, column: 10, scope: !61)
!66 = !DILocation(line: 77, column: 10, scope: !61)
!67 = !DILocation(line: 78, column: 10, scope: !61)
!68 = !DILocation(line: 79, column: 10, scope: !61)
!69 = !DILocation(line: 80, column: 10, scope: !61)
!70 = !DILocation(line: 81, column: 10, scope: !61)
!71 = !DILocation(line: 82, column: 11, scope: !61)
!72 = !DILocation(line: 83, column: 11, scope: !61)
!73 = !DILocation(line: 84, column: 11, scope: !61)
!74 = !DILocation(line: 85, column: 11, scope: !61)
!75 = !DILocation(line: 86, column: 11, scope: !61)
!76 = !DILocation(line: 87, column: 11, scope: !61)
!77 = !DILocation(line: 88, column: 11, scope: !61)
!78 = !DILocation(line: 89, column: 11, scope: !61)
!79 = !DILocation(line: 90, column: 11, scope: !61)
!80 = !DILocation(line: 91, column: 11, scope: !61)
!81 = !DILocation(line: 92, column: 11, scope: !61)
!82 = !DILocation(line: 93, column: 11, scope: !61)
!83 = !DILocation(line: 94, column: 11, scope: !61)
!84 = !DILocation(line: 95, column: 11, scope: !61)
!85 = !DILocation(line: 96, column: 11, scope: !61)
!86 = !DILocation(line: 97, column: 11, scope: !61)
!87 = !DILocation(line: 98, column: 11, scope: !61)
!88 = !DILocation(line: 99, column: 11, scope: !61)
!89 = !DILocation(line: 100, column: 11, scope: !61)
!90 = !DILocation(line: 101, column: 11, scope: !61)
!91 = !DILocation(line: 102, column: 5, scope: !61)
!92 = !DILocation(line: 103, column: 5, scope: !61)
!93 = distinct !DISubprogram(name: "Unknown0", linkageName: "Unknown0", scope: null, file: !4, line: 105, type: !5, scopeLine: 105, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!94 = !DILocation(line: 106, column: 10, scope: !95)
!95 = !DILexicalBlockFile(scope: !93, file: !4, discriminator: 0)
!96 = !DILocation(line: 107, column: 10, scope: !95)
!97 = !DILocation(line: 108, column: 10, scope: !95)
!98 = !DILocation(line: 109, column: 10, scope: !95)
!99 = !DILocation(line: 110, column: 10, scope: !95)
!100 = !DILocation(line: 111, column: 10, scope: !95)
!101 = !DILocation(line: 112, column: 5, scope: !95)
!102 = !DILocation(line: 113, column: 5, scope: !95)
!103 = distinct !DISubprogram(name: "_mlir_ciface_Unknown0", linkageName: "_mlir_ciface_Unknown0", scope: null, file: !4, line: 115, type: !5, scopeLine: 115, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!104 = !DILocation(line: 116, column: 10, scope: !105)
!105 = !DILexicalBlockFile(scope: !103, file: !4, discriminator: 0)
!106 = !DILocation(line: 117, column: 10, scope: !105)
!107 = !DILocation(line: 118, column: 10, scope: !105)
!108 = !DILocation(line: 119, column: 10, scope: !105)
!109 = !DILocation(line: 120, column: 10, scope: !105)
!110 = !DILocation(line: 121, column: 10, scope: !105)
!111 = !DILocation(line: 122, column: 10, scope: !105)
!112 = !DILocation(line: 123, column: 10, scope: !105)
!113 = !DILocation(line: 124, column: 10, scope: !105)
!114 = !DILocation(line: 125, column: 10, scope: !105)
!115 = !DILocation(line: 126, column: 11, scope: !105)
!116 = !DILocation(line: 127, column: 11, scope: !105)
!117 = !DILocation(line: 128, column: 11, scope: !105)
!118 = !DILocation(line: 129, column: 11, scope: !105)
!119 = !DILocation(line: 130, column: 11, scope: !105)
!120 = !DILocation(line: 131, column: 11, scope: !105)
!121 = !DILocation(line: 132, column: 11, scope: !105)
!122 = !DILocation(line: 133, column: 11, scope: !105)
!123 = !DILocation(line: 134, column: 11, scope: !105)
!124 = !DILocation(line: 135, column: 11, scope: !105)
!125 = !DILocation(line: 136, column: 11, scope: !105)
!126 = !DILocation(line: 137, column: 11, scope: !105)
!127 = !DILocation(line: 138, column: 11, scope: !105)
!128 = !DILocation(line: 139, column: 11, scope: !105)
!129 = !DILocation(line: 140, column: 5, scope: !105)
!130 = !DILocation(line: 141, column: 5, scope: !105)
