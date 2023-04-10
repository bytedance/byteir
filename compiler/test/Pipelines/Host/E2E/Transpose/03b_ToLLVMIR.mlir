// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module, llvm.data_layout = ""} {
  llvm.func @Unknown0(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr<f32>, %arg12: !llvm.ptr<f32>, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(32 : index) : i64
    %1 = llvm.mlir.constant(4096 : index) : i64
    %2 = llvm.mlir.constant(131072 : index) : i64
    %3 = llvm.mlir.constant(2048 : index) : i64
    %4 = llvm.mlir.constant(8 : index) : i64
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.constant(2 : index) : i64
    %9 = llvm.mlir.constant(3 : index) : i64
    %10 = llvm.mlir.constant(4 : index) : i64
    %11 = llvm.mlir.constant(5 : index) : i64
    %12 = llvm.mlir.constant(6 : index) : i64
    %13 = llvm.mlir.constant(7 : index) : i64
    llvm.br ^bb1(%5 : i64)
  ^bb1(%14: i64):  // 2 preds: ^bb0, ^bb2
    %15 = llvm.icmp "slt" %14, %3 : i64
    llvm.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.srem %14, %4  : i64
    %17 = llvm.sdiv %14, %4  : i64
    %18 = llvm.srem %17, %7  : i64
    %19 = llvm.sdiv %17, %7  : i64
    %20 = llvm.mul %16, %4  : i64
    %21 = llvm.mul %19, %4  : i64
    %22 = llvm.mul %5, %2  : i64
    %23 = llvm.mul %21, %1  : i64
    %24 = llvm.add %22, %23  : i64
    %25 = llvm.mul %18, %7  : i64
    %26 = llvm.add %24, %25  : i64
    %27 = llvm.add %26, %20  : i64
    %28 = llvm.getelementptr %arg1[%27] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %29 = llvm.bitcast %28 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    %30 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %31 = llvm.add %21, %6  : i64
    %32 = llvm.mul %5, %2  : i64
    %33 = llvm.mul %31, %1  : i64
    %34 = llvm.add %32, %33  : i64
    %35 = llvm.mul %18, %7  : i64
    %36 = llvm.add %34, %35  : i64
    %37 = llvm.add %36, %20  : i64
    %38 = llvm.getelementptr %arg1[%37] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %39 = llvm.bitcast %38 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    %40 = llvm.load %39 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %41 = llvm.add %21, %8  : i64
    %42 = llvm.mul %5, %2  : i64
    %43 = llvm.mul %41, %1  : i64
    %44 = llvm.add %42, %43  : i64
    %45 = llvm.mul %18, %7  : i64
    %46 = llvm.add %44, %45  : i64
    %47 = llvm.add %46, %20  : i64
    %48 = llvm.getelementptr %arg1[%47] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %49 = llvm.bitcast %48 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    %50 = llvm.load %49 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %51 = llvm.add %21, %9  : i64
    %52 = llvm.mul %5, %2  : i64
    %53 = llvm.mul %51, %1  : i64
    %54 = llvm.add %52, %53  : i64
    %55 = llvm.mul %18, %7  : i64
    %56 = llvm.add %54, %55  : i64
    %57 = llvm.add %56, %20  : i64
    %58 = llvm.getelementptr %arg1[%57] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %59 = llvm.bitcast %58 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    %60 = llvm.load %59 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %61 = llvm.add %21, %10  : i64
    %62 = llvm.mul %5, %2  : i64
    %63 = llvm.mul %61, %1  : i64
    %64 = llvm.add %62, %63  : i64
    %65 = llvm.mul %18, %7  : i64
    %66 = llvm.add %64, %65  : i64
    %67 = llvm.add %66, %20  : i64
    %68 = llvm.getelementptr %arg1[%67] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %69 = llvm.bitcast %68 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    %70 = llvm.load %69 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %71 = llvm.add %21, %11  : i64
    %72 = llvm.mul %5, %2  : i64
    %73 = llvm.mul %71, %1  : i64
    %74 = llvm.add %72, %73  : i64
    %75 = llvm.mul %18, %7  : i64
    %76 = llvm.add %74, %75  : i64
    %77 = llvm.add %76, %20  : i64
    %78 = llvm.getelementptr %arg1[%77] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %79 = llvm.bitcast %78 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    %80 = llvm.load %79 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %81 = llvm.add %21, %12  : i64
    %82 = llvm.mul %5, %2  : i64
    %83 = llvm.mul %81, %1  : i64
    %84 = llvm.add %82, %83  : i64
    %85 = llvm.mul %18, %7  : i64
    %86 = llvm.add %84, %85  : i64
    %87 = llvm.add %86, %20  : i64
    %88 = llvm.getelementptr %arg1[%87] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %89 = llvm.bitcast %88 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    %90 = llvm.load %89 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %91 = llvm.add %21, %13  : i64
    %92 = llvm.mul %5, %2  : i64
    %93 = llvm.mul %91, %1  : i64
    %94 = llvm.add %92, %93  : i64
    %95 = llvm.mul %18, %7  : i64
    %96 = llvm.add %94, %95  : i64
    %97 = llvm.add %96, %20  : i64
    %98 = llvm.getelementptr %arg1[%97] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %99 = llvm.bitcast %98 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    %100 = llvm.load %99 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %101 = llvm.shufflevector %30, %40 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32> 
    %102 = llvm.shufflevector %30, %40 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32> 
    %103 = llvm.shufflevector %50, %60 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32> 
    %104 = llvm.shufflevector %50, %60 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32> 
    %105 = llvm.shufflevector %70, %80 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32> 
    %106 = llvm.shufflevector %70, %80 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32> 
    %107 = llvm.shufflevector %90, %100 [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32> 
    %108 = llvm.shufflevector %90, %100 [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32> 
    %109 = llvm.shufflevector %101, %103 [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32> 
    %110 = llvm.shufflevector %102, %104 [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32> 
    %111 = llvm.shufflevector %105, %107 [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32> 
    %112 = llvm.shufflevector %106, %108 [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32> 
    %113 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %101, %109 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %114 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %103, %109 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %115 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %102, %110 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %116 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %104, %110 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %117 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %105, %111 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %118 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %107, %111 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %119 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %106, %112 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %120 = llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %108, %112 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %121 = llvm.shufflevector %113, %117 [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32> 
    %122 = llvm.shufflevector %114, %118 [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32> 
    %123 = llvm.shufflevector %115, %119 [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32> 
    %124 = llvm.shufflevector %116, %120 [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32> 
    %125 = llvm.shufflevector %113, %117 [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32> 
    %126 = llvm.shufflevector %114, %118 [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32> 
    %127 = llvm.shufflevector %115, %119 [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32> 
    %128 = llvm.shufflevector %116, %120 [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32> 
    %129 = llvm.mul %5, %2  : i64
    %130 = llvm.mul %18, %3  : i64
    %131 = llvm.add %129, %130  : i64
    %132 = llvm.mul %20, %0  : i64
    %133 = llvm.add %131, %132  : i64
    %134 = llvm.add %133, %21  : i64
    %135 = llvm.getelementptr %arg12[%134] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %136 = llvm.bitcast %135 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.store %121, %136 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %137 = llvm.add %20, %6  : i64
    %138 = llvm.mul %5, %2  : i64
    %139 = llvm.mul %18, %3  : i64
    %140 = llvm.add %138, %139  : i64
    %141 = llvm.mul %137, %0  : i64
    %142 = llvm.add %140, %141  : i64
    %143 = llvm.add %142, %21  : i64
    %144 = llvm.getelementptr %arg12[%143] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %145 = llvm.bitcast %144 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.store %122, %145 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %146 = llvm.add %20, %8  : i64
    %147 = llvm.mul %5, %2  : i64
    %148 = llvm.mul %18, %3  : i64
    %149 = llvm.add %147, %148  : i64
    %150 = llvm.mul %146, %0  : i64
    %151 = llvm.add %149, %150  : i64
    %152 = llvm.add %151, %21  : i64
    %153 = llvm.getelementptr %arg12[%152] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %154 = llvm.bitcast %153 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.store %123, %154 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %155 = llvm.add %20, %9  : i64
    %156 = llvm.mul %5, %2  : i64
    %157 = llvm.mul %18, %3  : i64
    %158 = llvm.add %156, %157  : i64
    %159 = llvm.mul %155, %0  : i64
    %160 = llvm.add %158, %159  : i64
    %161 = llvm.add %160, %21  : i64
    %162 = llvm.getelementptr %arg12[%161] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %163 = llvm.bitcast %162 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.store %124, %163 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %164 = llvm.add %20, %10  : i64
    %165 = llvm.mul %5, %2  : i64
    %166 = llvm.mul %18, %3  : i64
    %167 = llvm.add %165, %166  : i64
    %168 = llvm.mul %164, %0  : i64
    %169 = llvm.add %167, %168  : i64
    %170 = llvm.add %169, %21  : i64
    %171 = llvm.getelementptr %arg12[%170] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %172 = llvm.bitcast %171 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.store %125, %172 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %173 = llvm.add %20, %11  : i64
    %174 = llvm.mul %5, %2  : i64
    %175 = llvm.mul %18, %3  : i64
    %176 = llvm.add %174, %175  : i64
    %177 = llvm.mul %173, %0  : i64
    %178 = llvm.add %176, %177  : i64
    %179 = llvm.add %178, %21  : i64
    %180 = llvm.getelementptr %arg12[%179] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %181 = llvm.bitcast %180 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.store %126, %181 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %182 = llvm.add %20, %12  : i64
    %183 = llvm.mul %5, %2  : i64
    %184 = llvm.mul %18, %3  : i64
    %185 = llvm.add %183, %184  : i64
    %186 = llvm.mul %182, %0  : i64
    %187 = llvm.add %185, %186  : i64
    %188 = llvm.add %187, %21  : i64
    %189 = llvm.getelementptr %arg12[%188] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %190 = llvm.bitcast %189 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.store %127, %190 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %191 = llvm.add %20, %13  : i64
    %192 = llvm.mul %5, %2  : i64
    %193 = llvm.mul %18, %3  : i64
    %194 = llvm.add %192, %193  : i64
    %195 = llvm.mul %191, %0  : i64
    %196 = llvm.add %194, %195  : i64
    %197 = llvm.add %196, %21  : i64
    %198 = llvm.getelementptr %arg12[%197] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %199 = llvm.bitcast %198 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.store %128, %199 {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>
    %200 = llvm.add %14, %6  : i64
    llvm.br ^bb1(%200 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown0(%arg0: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.extractvalue %0[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.extractvalue %0[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %13 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.extractvalue %12[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.extractvalue %12[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.extractvalue %12[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.extractvalue %12[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.extractvalue %12[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @Unknown0(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
}