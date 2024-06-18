// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module} {
  llvm.func @Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(32 : index) : i64
    %1 = llvm.mlir.constant(4096 : index) : i64
    %2 = llvm.mlir.constant(131072 : index) : i64
    %3 = llvm.mlir.constant(2048 : index) : i64
    %4 = llvm.mlir.constant(8 : index) : i64
    %5 = llvm.mlir.constant(64 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(2 : index) : i64
    %9 = llvm.mlir.constant(3 : index) : i64
    %10 = llvm.mlir.constant(4 : index) : i64
    %11 = llvm.mlir.constant(5 : index) : i64
    %12 = llvm.mlir.constant(6 : index) : i64
    %13 = llvm.mlir.constant(7 : index) : i64
    llvm.br ^bb1(%6 : i64)
  ^bb1(%14: i64):  // 2 preds: ^bb0, ^bb2
    %15 = llvm.icmp "slt" %14, %3 : i64
    llvm.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.srem %14, %4  : i64
    %17 = llvm.sdiv %14, %4  : i64
    %18 = llvm.srem %17, %5  : i64
    %19 = llvm.sdiv %17, %5  : i64
    %20 = llvm.mul %16, %4  : i64
    %21 = llvm.mul %19, %4  : i64
    %22 = llvm.mul %6, %2  : i64
    %23 = llvm.mul %21, %1  : i64
    %24 = llvm.add %22, %23  : i64
    %25 = llvm.mul %18, %5  : i64
    %26 = llvm.add %24, %25  : i64
    %27 = llvm.add %26, %20  : i64
    %28 = llvm.getelementptr %arg1[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
    %30 = llvm.add %21, %7  : i64
    %31 = llvm.mul %6, %2  : i64
    %32 = llvm.mul %30, %1  : i64
    %33 = llvm.add %31, %32  : i64
    %34 = llvm.mul %18, %5  : i64
    %35 = llvm.add %33, %34  : i64
    %36 = llvm.add %35, %20  : i64
    %37 = llvm.getelementptr %arg1[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
    %39 = llvm.add %21, %8  : i64
    %40 = llvm.mul %6, %2  : i64
    %41 = llvm.mul %39, %1  : i64
    %42 = llvm.add %40, %41  : i64
    %43 = llvm.mul %18, %5  : i64
    %44 = llvm.add %42, %43  : i64
    %45 = llvm.add %44, %20  : i64
    %46 = llvm.getelementptr %arg1[%45] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %47 = llvm.load %46 {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
    %48 = llvm.add %21, %9  : i64
    %49 = llvm.mul %6, %2  : i64
    %50 = llvm.mul %48, %1  : i64
    %51 = llvm.add %49, %50  : i64
    %52 = llvm.mul %18, %5  : i64
    %53 = llvm.add %51, %52  : i64
    %54 = llvm.add %53, %20  : i64
    %55 = llvm.getelementptr %arg1[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.load %55 {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
    %57 = llvm.add %21, %10  : i64
    %58 = llvm.mul %6, %2  : i64
    %59 = llvm.mul %57, %1  : i64
    %60 = llvm.add %58, %59  : i64
    %61 = llvm.mul %18, %5  : i64
    %62 = llvm.add %60, %61  : i64
    %63 = llvm.add %62, %20  : i64
    %64 = llvm.getelementptr %arg1[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %65 = llvm.load %64 {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
    %66 = llvm.add %21, %11  : i64
    %67 = llvm.mul %6, %2  : i64
    %68 = llvm.mul %66, %1  : i64
    %69 = llvm.add %67, %68  : i64
    %70 = llvm.mul %18, %5  : i64
    %71 = llvm.add %69, %70  : i64
    %72 = llvm.add %71, %20  : i64
    %73 = llvm.getelementptr %arg1[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %74 = llvm.load %73 {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
    %75 = llvm.add %21, %12  : i64
    %76 = llvm.mul %6, %2  : i64
    %77 = llvm.mul %75, %1  : i64
    %78 = llvm.add %76, %77  : i64
    %79 = llvm.mul %18, %5  : i64
    %80 = llvm.add %78, %79  : i64
    %81 = llvm.add %80, %20  : i64
    %82 = llvm.getelementptr %arg1[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %83 = llvm.load %82 {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
    %84 = llvm.add %21, %13  : i64
    %85 = llvm.mul %6, %2  : i64
    %86 = llvm.mul %84, %1  : i64
    %87 = llvm.add %85, %86  : i64
    %88 = llvm.mul %18, %5  : i64
    %89 = llvm.add %87, %88  : i64
    %90 = llvm.add %89, %20  : i64
    %91 = llvm.getelementptr %arg1[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.load %91 {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
    %93 = llvm.shufflevector %29, %38 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32> 
    %94 = llvm.shufflevector %29, %38 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32> 
    %95 = llvm.shufflevector %47, %56 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32> 
    %96 = llvm.shufflevector %47, %56 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32> 
    %97 = llvm.shufflevector %65, %74 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32> 
    %98 = llvm.shufflevector %65, %74 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32> 
    %99 = llvm.shufflevector %83, %92 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32> 
    %100 = llvm.shufflevector %83, %92 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32> 
    %101 = llvm.shufflevector %93, %95 [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32> 
    %102 = llvm.shufflevector %94, %96 [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32> 
    %103 = llvm.shufflevector %97, %99 [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32> 
    %104 = llvm.shufflevector %98, %100 [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32> 
    %105 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %93, %101 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %106 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %95, %101 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %107 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %94, %102 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %108 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %96, %102 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %109 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %97, %103 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %110 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %99, %103 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %111 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %98, %104 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %112 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %100, %104 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %113 = llvm.shufflevector %105, %109 [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32> 
    %114 = llvm.shufflevector %106, %110 [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32> 
    %115 = llvm.shufflevector %107, %111 [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32> 
    %116 = llvm.shufflevector %108, %112 [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32> 
    %117 = llvm.shufflevector %105, %109 [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32> 
    %118 = llvm.shufflevector %106, %110 [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32> 
    %119 = llvm.shufflevector %107, %111 [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32> 
    %120 = llvm.shufflevector %108, %112 [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32> 
    %121 = llvm.mul %6, %2  : i64
    %122 = llvm.mul %18, %3  : i64
    %123 = llvm.add %121, %122  : i64
    %124 = llvm.mul %20, %0  : i64
    %125 = llvm.add %123, %124  : i64
    %126 = llvm.add %125, %21  : i64
    %127 = llvm.getelementptr %arg12[%126] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %113, %127 {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
    %128 = llvm.add %20, %7  : i64
    %129 = llvm.mul %6, %2  : i64
    %130 = llvm.mul %18, %3  : i64
    %131 = llvm.add %129, %130  : i64
    %132 = llvm.mul %128, %0  : i64
    %133 = llvm.add %131, %132  : i64
    %134 = llvm.add %133, %21  : i64
    %135 = llvm.getelementptr %arg12[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %114, %135 {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
    %136 = llvm.add %20, %8  : i64
    %137 = llvm.mul %6, %2  : i64
    %138 = llvm.mul %18, %3  : i64
    %139 = llvm.add %137, %138  : i64
    %140 = llvm.mul %136, %0  : i64
    %141 = llvm.add %139, %140  : i64
    %142 = llvm.add %141, %21  : i64
    %143 = llvm.getelementptr %arg12[%142] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %115, %143 {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
    %144 = llvm.add %20, %9  : i64
    %145 = llvm.mul %6, %2  : i64
    %146 = llvm.mul %18, %3  : i64
    %147 = llvm.add %145, %146  : i64
    %148 = llvm.mul %144, %0  : i64
    %149 = llvm.add %147, %148  : i64
    %150 = llvm.add %149, %21  : i64
    %151 = llvm.getelementptr %arg12[%150] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %116, %151 {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
    %152 = llvm.add %20, %10  : i64
    %153 = llvm.mul %6, %2  : i64
    %154 = llvm.mul %18, %3  : i64
    %155 = llvm.add %153, %154  : i64
    %156 = llvm.mul %152, %0  : i64
    %157 = llvm.add %155, %156  : i64
    %158 = llvm.add %157, %21  : i64
    %159 = llvm.getelementptr %arg12[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %117, %159 {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
    %160 = llvm.add %20, %11  : i64
    %161 = llvm.mul %6, %2  : i64
    %162 = llvm.mul %18, %3  : i64
    %163 = llvm.add %161, %162  : i64
    %164 = llvm.mul %160, %0  : i64
    %165 = llvm.add %163, %164  : i64
    %166 = llvm.add %165, %21  : i64
    %167 = llvm.getelementptr %arg12[%166] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %118, %167 {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
    %168 = llvm.add %20, %12  : i64
    %169 = llvm.mul %6, %2  : i64
    %170 = llvm.mul %18, %3  : i64
    %171 = llvm.add %169, %170  : i64
    %172 = llvm.mul %168, %0  : i64
    %173 = llvm.add %171, %172  : i64
    %174 = llvm.add %173, %21  : i64
    %175 = llvm.getelementptr %arg12[%174] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %119, %175 {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
    %176 = llvm.add %20, %13  : i64
    %177 = llvm.mul %6, %2  : i64
    %178 = llvm.mul %18, %3  : i64
    %179 = llvm.add %177, %178  : i64
    %180 = llvm.mul %176, %0  : i64
    %181 = llvm.add %179, %180  : i64
    %182 = llvm.add %181, %21  : i64
    %183 = llvm.getelementptr %arg12[%182] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %120, %183 {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
    %184 = llvm.add %14, %7  : i64
    llvm.br ^bb1(%184 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.extractvalue %0[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.extractvalue %0[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %13 = llvm.extractvalue %12[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.extractvalue %12[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.extractvalue %12[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.extractvalue %12[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.extractvalue %12[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.extractvalue %12[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @Unknown0(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
}