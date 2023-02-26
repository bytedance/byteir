// RUN: byteir-translate %s -mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: constant
// CHECK-LABEL: define void @_mlir_ciface_Unknown
module attributes {byre.container_module, llvm.data_layout = ""} {
  llvm.func @free(!llvm.ptr<i8>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private constant @__constant_100x1296xi32(dense<1> : tensor<100x1296xi32>) {addr_space = 0 : i32} : !llvm.array<100 x array<1296 x i32>>
  llvm.mlir.global private constant @__constant_100xi32(dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]> : tensor<100xi32>) {addr_space = 0 : i32} : !llvm.array<100 x i32>
  llvm.mlir.global private constant @__constant_51200xi32(dense<0> : tensor<51200xi32>) {addr_space = 0 : i32} : !llvm.array<51200 x i32>
  llvm.func @Unknown6(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: !llvm.ptr<i32>, %arg14: !llvm.ptr<i32>, %arg15: i64, %arg16: i64, %arg17: i64) attributes {__byre__kernel_name = "Unknown6", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3888 : index) : i64
    %1 = llvm.mlir.constant(144 : index) : i64
    %2 = llvm.mlir.constant(388800 : index) : i64
    %3 = llvm.mlir.constant(129600 : index) : i64
    %4 = llvm.mlir.constant(100 : index) : i64
    %5 = llvm.mlir.constant(-1 : index) : i64
    %6 = llvm.mlir.constant(2 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(27 : index) : i64
    %10 = llvm.mlir.constant(48 : index) : i64
    %11 = llvm.mlir.constant(3 : index) : i64
    %12 = llvm.mlir.constant(1296 : index) : i64
    %13 = llvm.mlir.constant(51200 : index) : i64
    %14 = llvm.mlir.constant(5 : i32) : i32
    %15 = llvm.mlir.constant(3 : i32) : i32
    %16 = llvm.mlir.constant(6 : i32) : i32
    %17 = llvm.mlir.addressof @__constant_51200xi32 : !llvm.ptr<array<51200 x i32>>
    %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr<array<51200 x i32>>) -> !llvm.ptr<i32>
    %19 = llvm.mlir.addressof @__constant_100xi32 : !llvm.ptr<array<100 x i32>>
    %20 = llvm.getelementptr %19[0, 0] : (!llvm.ptr<array<100 x i32>>) -> !llvm.ptr<i32>
    %21 = llvm.mlir.addressof @__constant_100x1296xi32 : !llvm.ptr<array<100 x array<1296 x i32>>>
    %22 = llvm.getelementptr %21[0, 0, 0] : (!llvm.ptr<array<100 x array<1296 x i32>>>) -> !llvm.ptr<i32>
    %23 = llvm.mlir.null : !llvm.ptr<i32>
    %24 = llvm.getelementptr %23[388800] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
    %25 = llvm.ptrtoint %24 : !llvm.ptr<i32> to i64
    %26 = llvm.call @malloc(%25) : (i64) -> !llvm.ptr<i8>
    %27 = llvm.bitcast %26 : !llvm.ptr<i8> to !llvm.ptr<i32>
    llvm.br ^bb1(%7 : i64)
  ^bb1(%28: i64):  // 2 preds: ^bb0, ^bb2
    %29 = llvm.icmp "slt" %28, %2 : i64
    llvm.cond_br %29, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %30 = llvm.srem %28, %11  : i64
    %31 = llvm.sdiv %28, %11  : i64
    %32 = llvm.srem %31, %10  : i64
    %33 = llvm.sdiv %31, %10  : i64
    %34 = llvm.srem %33, %9  : i64
    %35 = llvm.sdiv %33, %9  : i64
    %36 = llvm.srem %35, %4  : i64
    %37 = llvm.sdiv %35, %4  : i64
    %38 = llvm.mul %37, %2  : i64
    %39 = llvm.mul %36, %0  : i64
    %40 = llvm.add %38, %39  : i64
    %41 = llvm.mul %34, %1  : i64
    %42 = llvm.add %40, %41  : i64
    %43 = llvm.mul %32, %11  : i64
    %44 = llvm.add %42, %43  : i64
    %45 = llvm.add %44, %30  : i64
    %46 = llvm.getelementptr %arg1[%45] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %47 = llvm.load %46 : !llvm.ptr<f32>
    %48 = llvm.fptosi %47 : f32 to i32
    %49 = llvm.mul %37, %2  : i64
    %50 = llvm.mul %36, %0  : i64
    %51 = llvm.add %49, %50  : i64
    %52 = llvm.mul %34, %1  : i64
    %53 = llvm.add %51, %52  : i64
    %54 = llvm.mul %32, %11  : i64
    %55 = llvm.add %53, %54  : i64
    %56 = llvm.add %55, %30  : i64
    %57 = llvm.getelementptr %27[%56] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %48, %57 : !llvm.ptr<i32>
    %58 = llvm.add %28, %8  : i64
    llvm.br ^bb1(%58 : i64)
  ^bb3:  // pred: ^bb1
    %59 = llvm.mlir.null : !llvm.ptr<i32>
    %60 = llvm.getelementptr %59[129600] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
    %61 = llvm.ptrtoint %60 : !llvm.ptr<i32> to i64
    %62 = llvm.call @malloc(%61) : (i64) -> !llvm.ptr<i8>
    %63 = llvm.bitcast %62 : !llvm.ptr<i8> to !llvm.ptr<i32>
    llvm.br ^bb4(%7 : i64)
  ^bb4(%64: i64):  // 2 preds: ^bb3, ^bb5
    %65 = llvm.icmp "slt" %64, %3 : i64
    llvm.cond_br %65, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %66 = llvm.srem %64, %12  : i64
    %67 = llvm.sdiv %64, %12  : i64
    %68 = llvm.icmp "slt" %67, %7 : i64
    %69 = llvm.sub %5, %67  : i64
    %70 = llvm.select %68, %69, %67 : i1, i64
    %71 = llvm.sdiv %70, %4  : i64
    %72 = llvm.sub %5, %71  : i64
    %73 = llvm.select %68, %72, %71 : i1, i64
    %74 = llvm.srem %67, %4  : i64
    %75 = llvm.icmp "slt" %74, %7 : i64
    %76 = llvm.add %74, %4  : i64
    %77 = llvm.select %75, %76, %74 : i1, i64
    %78 = llvm.icmp "slt" %66, %7 : i64
    %79 = llvm.sub %5, %66  : i64
    %80 = llvm.select %78, %79, %66 : i1, i64
    %81 = llvm.sdiv %80, %10  : i64
    %82 = llvm.sub %5, %81  : i64
    %83 = llvm.select %78, %82, %81 : i1, i64
    %84 = llvm.srem %66, %10  : i64
    %85 = llvm.icmp "slt" %84, %7 : i64
    %86 = llvm.add %84, %10  : i64
    %87 = llvm.select %85, %86, %84 : i1, i64
    %88 = llvm.mul %73, %2  : i64
    %89 = llvm.mul %77, %0  : i64
    %90 = llvm.add %88, %89  : i64
    %91 = llvm.mul %83, %1  : i64
    %92 = llvm.add %90, %91  : i64
    %93 = llvm.mul %87, %11  : i64
    %94 = llvm.add %92, %93  : i64
    %95 = llvm.add %94, %6  : i64
    %96 = llvm.getelementptr %27[%95] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %97 = llvm.load %96 : !llvm.ptr<i32>
    %98 = llvm.mul %73, %2  : i64
    %99 = llvm.mul %77, %0  : i64
    %100 = llvm.add %98, %99  : i64
    %101 = llvm.mul %83, %1  : i64
    %102 = llvm.add %100, %101  : i64
    %103 = llvm.mul %87, %11  : i64
    %104 = llvm.add %102, %103  : i64
    %105 = llvm.add %104, %7  : i64
    %106 = llvm.getelementptr %27[%105] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %107 = llvm.load %106 : !llvm.ptr<i32>
    %108 = llvm.mul %73, %2  : i64
    %109 = llvm.mul %77, %0  : i64
    %110 = llvm.add %108, %109  : i64
    %111 = llvm.mul %83, %1  : i64
    %112 = llvm.add %110, %111  : i64
    %113 = llvm.mul %87, %11  : i64
    %114 = llvm.add %112, %113  : i64
    %115 = llvm.add %114, %8  : i64
    %116 = llvm.getelementptr %27[%115] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %117 = llvm.load %116 : !llvm.ptr<i32>
    %118 = llvm.getelementptr %20[%67] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %119 = llvm.load %118 : !llvm.ptr<i32>
    %120 = llvm.ashr %117, %14  : i32
    %121 = llvm.shl %120, %15  : i32
    %122 = llvm.ashr %107, %14  : i32
    %123 = llvm.shl %122, %16  : i32
    %124 = llvm.add %123, %121  : i32
    %125 = llvm.ashr %97, %14  : i32
    %126 = llvm.add %125, %124  : i32
    %127 = llvm.add %126, %119  : i32
    %128 = llvm.mul %67, %12  : i64
    %129 = llvm.add %128, %66  : i64
    %130 = llvm.add %129, %7  : i64
    %131 = llvm.getelementptr %63[%130] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %127, %131 : !llvm.ptr<i32>
    %132 = llvm.add %64, %8  : i64
    llvm.br ^bb4(%132 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @free(%26) : (!llvm.ptr<i8>) -> ()
    llvm.br ^bb7(%7 : i64)
  ^bb7(%133: i64):  // 2 preds: ^bb6, ^bb8
    %134 = llvm.icmp "slt" %133, %13 : i64
    llvm.cond_br %134, ^bb8, ^bb9(%7 : i64)
  ^bb8:  // pred: ^bb7
    %135 = llvm.getelementptr %18[%133] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %136 = llvm.load %135 : !llvm.ptr<i32>
    %137 = llvm.getelementptr %arg14[%133] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %136, %137 : !llvm.ptr<i32>
    %138 = llvm.add %133, %8  : i64
    llvm.br ^bb7(%138 : i64)
  ^bb9(%139: i64):  // 2 preds: ^bb7, ^bb10
    %140 = llvm.icmp "slt" %139, %3 : i64
    llvm.cond_br %140, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %141 = llvm.srem %139, %12  : i64
    %142 = llvm.sdiv %139, %12  : i64
    %143 = llvm.mul %142, %12  : i64
    %144 = llvm.add %143, %141  : i64
    %145 = llvm.add %144, %7  : i64
    %146 = llvm.getelementptr %63[%145] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %147 = llvm.load %146 : !llvm.ptr<i32>
    %148 = llvm.sext %147 : i32 to i64
    %149 = llvm.getelementptr %arg14[%148] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %150 = llvm.load %149 : !llvm.ptr<i32>
    %151 = llvm.mul %142, %12  : i64
    %152 = llvm.add %151, %141  : i64
    %153 = llvm.getelementptr %22[%152] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %154 = llvm.load %153 : !llvm.ptr<i32>
    %155 = llvm.add %150, %154  : i32
    %156 = llvm.getelementptr %arg14[%148] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %155, %156 : !llvm.ptr<i32>
    %157 = llvm.add %139, %8  : i64
    llvm.br ^bb9(%157 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @free(%62) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown6(%arg0: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {__byre__kernel_name = "Unknown6", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %7 = llvm.extractvalue %0[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %8 = llvm.extractvalue %0[3, 4] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %9 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %10 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %11 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %12 = llvm.extractvalue %0[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %13 = llvm.extractvalue %0[4, 4] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)> 
    %14 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @Unknown6(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i64, i64, i64) -> ()
    llvm.return
  }
}

