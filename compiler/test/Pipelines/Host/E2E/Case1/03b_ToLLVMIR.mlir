// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: constant
// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module, llvm.data_layout = ""} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_100xi32(dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]> : tensor<100xi32>) {addr_space = 0 : i32} : !llvm.array<100 x i32>
  llvm.mlir.global private constant @__constant_100x1296xi32(dense<1> : tensor<100x1296xi32>) {addr_space = 0 : i32} : !llvm.array<100 x array<1296 x i32>>
  llvm.func @Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: !llvm.ptr, %arg14: !llvm.ptr, %arg15: i64, %arg16: i64, %arg17: i64) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3888 : index) : i64
    %1 = llvm.mlir.constant(3 : index) : i64
    %2 = llvm.mlir.constant(129600 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(388800 : index) : i64
    %6 = llvm.mlir.constant(1296 : index) : i64
    %7 = llvm.mlir.constant(51200 : index) : i64
    %8 = llvm.mlir.constant(6 : i32) : i32
    %9 = llvm.mlir.constant(3 : i32) : i32
    %10 = llvm.mlir.constant(5 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.addressof @__constant_100x1296xi32 : !llvm.ptr
    %13 = llvm.getelementptr %12[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<100 x array<1296 x i32>>
    %14 = llvm.mlir.addressof @__constant_100xi32 : !llvm.ptr
    %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<100 x i32>
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.getelementptr %16[388800] : (!llvm.ptr) -> !llvm.ptr, i32
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.call @malloc(%18) : (i64) -> !llvm.ptr
    llvm.br ^bb1(%3 : i64)
  ^bb1(%20: i64):  // 2 preds: ^bb0, ^bb2
    %21 = llvm.icmp "slt" %20, %5 : i64
    llvm.cond_br %21, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %22 = llvm.getelementptr %arg1[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %23 = llvm.load %22 : !llvm.ptr -> f32
    %24 = llvm.fptosi %23 : f32 to i32
    %25 = llvm.getelementptr %19[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %24, %25 : i32, !llvm.ptr
    %26 = llvm.add %20, %4  : i64
    llvm.br ^bb1(%26 : i64)
  ^bb3:  // pred: ^bb1
    %27 = llvm.mlir.zero : !llvm.ptr
    %28 = llvm.getelementptr %27[129600] : (!llvm.ptr) -> !llvm.ptr, i32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    llvm.br ^bb4(%3 : i64)
  ^bb4(%31: i64):  // 2 preds: ^bb3, ^bb5
    %32 = llvm.icmp "slt" %31, %2 : i64
    llvm.cond_br %32, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %33 = llvm.srem %31, %6  : i64
    %34 = llvm.sdiv %31, %6  : i64
    %35 = llvm.getelementptr %19[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %36 = llvm.mul %34, %0  : i64
    %37 = llvm.mul %33, %1  : i64
    %38 = llvm.add %36, %37  : i64
    %39 = llvm.add %38, %3  : i64
    %40 = llvm.getelementptr %35[%39] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %41 = llvm.load %40 : !llvm.ptr -> i32
    %42 = llvm.mul %34, %0  : i64
    %43 = llvm.mul %33, %1  : i64
    %44 = llvm.add %42, %43  : i64
    %45 = llvm.add %44, %3  : i64
    %46 = llvm.getelementptr %19[%45] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %47 = llvm.load %46 : !llvm.ptr -> i32
    %48 = llvm.getelementptr %19[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %49 = llvm.mul %34, %0  : i64
    %50 = llvm.mul %33, %1  : i64
    %51 = llvm.add %49, %50  : i64
    %52 = llvm.add %51, %3  : i64
    %53 = llvm.getelementptr %48[%52] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %54 = llvm.load %53 : !llvm.ptr -> i32
    %55 = llvm.getelementptr %15[%34] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %56 = llvm.load %55 : !llvm.ptr -> i32
    %57 = llvm.ashr %54, %10  : i32
    %58 = llvm.shl %57, %9  : i32
    %59 = llvm.ashr %47, %10  : i32
    %60 = llvm.shl %59, %8  : i32
    %61 = llvm.add %60, %58  : i32
    %62 = llvm.ashr %41, %10  : i32
    %63 = llvm.add %62, %61  : i32
    %64 = llvm.add %63, %56  : i32
    %65 = llvm.mul %34, %6  : i64
    %66 = llvm.add %65, %33  : i64
    %67 = llvm.add %66, %3  : i64
    %68 = llvm.getelementptr %30[%67] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %64, %68 : i32, !llvm.ptr
    %69 = llvm.add %31, %4  : i64
    llvm.br ^bb4(%69 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @free(%19) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%3 : i64)
  ^bb7(%70: i64):  // 2 preds: ^bb6, ^bb8
    %71 = llvm.icmp "slt" %70, %7 : i64
    llvm.cond_br %71, ^bb8, ^bb9(%3 : i64)
  ^bb8:  // pred: ^bb7
    %72 = llvm.getelementptr %arg14[%70] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %11, %72 : i32, !llvm.ptr
    %73 = llvm.add %70, %4  : i64
    llvm.br ^bb7(%73 : i64)
  ^bb9(%74: i64):  // 2 preds: ^bb7, ^bb10
    %75 = llvm.icmp "slt" %74, %2 : i64
    llvm.cond_br %75, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %76 = llvm.srem %74, %6  : i64
    %77 = llvm.sdiv %74, %6  : i64
    %78 = llvm.mul %77, %6  : i64
    %79 = llvm.add %78, %76  : i64
    %80 = llvm.add %79, %3  : i64
    %81 = llvm.getelementptr %30[%80] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %82 = llvm.load %81 : !llvm.ptr -> i32
    %83 = llvm.sext %82 : i32 to i64
    %84 = llvm.getelementptr %arg14[%83] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %85 = llvm.load %84 : !llvm.ptr -> i32
    %86 = llvm.mul %77, %6  : i64
    %87 = llvm.add %86, %76  : i64
    %88 = llvm.getelementptr %13[%87] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %89 = llvm.load %88 : !llvm.ptr -> i32
    %90 = llvm.add %85, %89  : i32
    %91 = llvm.getelementptr %arg14[%83] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %90, %91 : i32, !llvm.ptr
    %92 = llvm.add %74, %4  : i64
    llvm.br ^bb9(%92 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @free(%30) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %7 = llvm.extractvalue %0[3, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %8 = llvm.extractvalue %0[3, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %9 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %10 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %11 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %12 = llvm.extractvalue %0[4, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %13 = llvm.extractvalue %0[4, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %14 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @Unknown0(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
}