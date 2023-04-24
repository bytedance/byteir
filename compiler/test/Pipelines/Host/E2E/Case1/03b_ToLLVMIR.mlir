// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: constant
// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module, llvm.data_layout = ""} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_100xi32(dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]> : tensor<100xi32>) {addr_space = 0 : i32} : !llvm.array<100 x i32>
  llvm.mlir.global private constant @__constant_100x1296xi32(dense<1> : tensor<100x1296xi32>) {addr_space = 0 : i32} : !llvm.array<100 x array<1296 x i32>>
  llvm.func @Unknown6(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: !llvm.ptr, %arg14: !llvm.ptr, %arg15: i64, %arg16: i64, %arg17: i64) attributes {__byre__kernel_name = "Unknown6", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(3888 : index) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(129600 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(388800 : index) : i64
    %7 = llvm.mlir.constant(1296 : index) : i64
    %8 = llvm.mlir.constant(51200 : index) : i64
    %9 = llvm.mlir.constant(6 : i32) : i32
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.mlir.constant(5 : i32) : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.mlir.addressof @__constant_100x1296xi32 : !llvm.ptr
    %14 = llvm.getelementptr %13[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<100 x array<1296 x i32>>
    %15 = llvm.mlir.addressof @__constant_100xi32 : !llvm.ptr
    %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<100 x i32>
    %17 = llvm.mlir.null : !llvm.ptr
    %18 = llvm.getelementptr %17[388800] : (!llvm.ptr) -> !llvm.ptr, i32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.call @malloc(%19) : (i64) -> !llvm.ptr
    llvm.br ^bb1(%4 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb2
    %22 = llvm.icmp "slt" %21, %6 : i64
    llvm.cond_br %22, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %23 = llvm.getelementptr %arg1[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %24 = llvm.load %23 : !llvm.ptr -> f32
    %25 = llvm.fptosi %24 : f32 to i32
    %26 = llvm.getelementptr %20[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %25, %26 : i32, !llvm.ptr
    %27 = llvm.add %21, %5  : i64
    llvm.br ^bb1(%27 : i64)
  ^bb3:  // pred: ^bb1
    %28 = llvm.mlir.null : !llvm.ptr
    %29 = llvm.getelementptr %28[129600] : (!llvm.ptr) -> !llvm.ptr, i32
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
    llvm.br ^bb4(%4 : i64)
  ^bb4(%32: i64):  // 2 preds: ^bb3, ^bb5
    %33 = llvm.icmp "slt" %32, %3 : i64
    llvm.cond_br %33, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %34 = llvm.srem %32, %7  : i64
    %35 = llvm.sdiv %32, %7  : i64
    %36 = llvm.mul %35, %1  : i64
    %37 = llvm.add %36, %0  : i64
    %38 = llvm.mul %34, %2  : i64
    %39 = llvm.add %37, %38  : i64
    %40 = llvm.add %39, %4  : i64
    %41 = llvm.getelementptr %20[%40] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %42 = llvm.load %41 : !llvm.ptr -> i32
    %43 = llvm.mul %35, %1  : i64
    %44 = llvm.mul %34, %2  : i64
    %45 = llvm.add %43, %44  : i64
    %46 = llvm.add %45, %4  : i64
    %47 = llvm.getelementptr %20[%46] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %48 = llvm.load %47 : !llvm.ptr -> i32
    %49 = llvm.mul %35, %1  : i64
    %50 = llvm.add %49, %5  : i64
    %51 = llvm.mul %34, %2  : i64
    %52 = llvm.add %50, %51  : i64
    %53 = llvm.add %52, %4  : i64
    %54 = llvm.getelementptr %20[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %55 = llvm.load %54 : !llvm.ptr -> i32
    %56 = llvm.getelementptr %16[%35] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %57 = llvm.load %56 : !llvm.ptr -> i32
    %58 = llvm.ashr %55, %11  : i32
    %59 = llvm.shl %58, %10  : i32
    %60 = llvm.ashr %48, %11  : i32
    %61 = llvm.shl %60, %9  : i32
    %62 = llvm.add %61, %59  : i32
    %63 = llvm.ashr %42, %11  : i32
    %64 = llvm.add %63, %62  : i32
    %65 = llvm.add %64, %57  : i32
    %66 = llvm.mul %35, %7  : i64
    %67 = llvm.add %66, %34  : i64
    %68 = llvm.add %67, %4  : i64
    %69 = llvm.getelementptr %31[%68] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %65, %69 : i32, !llvm.ptr
    %70 = llvm.add %32, %5  : i64
    llvm.br ^bb4(%70 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @free(%20) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%4 : i64)
  ^bb7(%71: i64):  // 2 preds: ^bb6, ^bb8
    %72 = llvm.icmp "slt" %71, %8 : i64
    llvm.cond_br %72, ^bb8, ^bb9(%4 : i64)
  ^bb8:  // pred: ^bb7
    %73 = llvm.getelementptr %arg14[%71] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %12, %73 : i32, !llvm.ptr
    %74 = llvm.add %71, %5  : i64
    llvm.br ^bb7(%74 : i64)
  ^bb9(%75: i64):  // 2 preds: ^bb7, ^bb10
    %76 = llvm.icmp "slt" %75, %3 : i64
    llvm.cond_br %76, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %77 = llvm.srem %75, %7  : i64
    %78 = llvm.sdiv %75, %7  : i64
    %79 = llvm.mul %78, %7  : i64
    %80 = llvm.add %79, %77  : i64
    %81 = llvm.add %80, %4  : i64
    %82 = llvm.getelementptr %31[%81] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %83 = llvm.load %82 : !llvm.ptr -> i32
    %84 = llvm.sext %83 : i32 to i64
    %85 = llvm.getelementptr %arg14[%84] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %86 = llvm.load %85 : !llvm.ptr -> i32
    %87 = llvm.mul %78, %7  : i64
    %88 = llvm.add %87, %77  : i64
    %89 = llvm.getelementptr %14[%88] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %90 = llvm.load %89 : !llvm.ptr -> i32
    %91 = llvm.add %86, %90  : i32
    %92 = llvm.getelementptr %arg14[%84] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %91, %92 : i32, !llvm.ptr
    %93 = llvm.add %75, %5  : i64
    llvm.br ^bb9(%93 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @free(%31) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown6(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {__byre__kernel_name = "Unknown6", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
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
    llvm.call @Unknown6(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
}