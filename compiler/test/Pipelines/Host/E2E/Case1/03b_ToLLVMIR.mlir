// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: constant
// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module, llvm.data_layout = ""} {
  llvm.func @free(!llvm.ptr<i8>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private constant @__constant_100x1296xi32(dense<1> : tensor<100x1296xi32>) {addr_space = 0 : i32} : !llvm.array<100 x array<1296 x i32>>
  llvm.mlir.global private constant @__constant_100xi32(dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]> : tensor<100xi32>) {addr_space = 0 : i32} : !llvm.array<100 x i32>
  llvm.func @Unknown6(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: !llvm.ptr<i32>, %arg14: !llvm.ptr<i32>, %arg15: i64, %arg16: i64, %arg17: i64) attributes {__byre__kernel_name = "Unknown6", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(3888 : index) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(129600 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(388800 : index) : i64
    %7 = llvm.mlir.constant(1296 : index) : i64
    %8 = llvm.mlir.constant(51200 : index) : i64
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(5 : i32) : i32
    %11 = llvm.mlir.constant(3 : i32) : i32
    %12 = llvm.mlir.constant(6 : i32) : i32
    %13 = llvm.mlir.addressof @__constant_100xi32 : !llvm.ptr<array<100 x i32>>
    %14 = llvm.getelementptr %13[0, 0] : (!llvm.ptr<array<100 x i32>>) -> !llvm.ptr<i32>
    %15 = llvm.mlir.addressof @__constant_100x1296xi32 : !llvm.ptr<array<100 x array<1296 x i32>>>
    %16 = llvm.getelementptr %15[0, 0, 0] : (!llvm.ptr<array<100 x array<1296 x i32>>>) -> !llvm.ptr<i32>
    %17 = llvm.mlir.null : !llvm.ptr<i32>
    %18 = llvm.getelementptr %17[388800] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
    %19 = llvm.ptrtoint %18 : !llvm.ptr<i32> to i64
    %20 = llvm.call @malloc(%19) : (i64) -> !llvm.ptr<i8>
    %21 = llvm.bitcast %20 : !llvm.ptr<i8> to !llvm.ptr<i32>
    llvm.br ^bb1(%4 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb2
    %23 = llvm.icmp "slt" %22, %6 : i64
    llvm.cond_br %23, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %24 = llvm.getelementptr %arg1[%22] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %25 = llvm.load %24 : !llvm.ptr<f32>
    %26 = llvm.fptosi %25 : f32 to i32
    %27 = llvm.getelementptr %21[%22] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %26, %27 : !llvm.ptr<i32>
    %28 = llvm.add %22, %5  : i64
    llvm.br ^bb1(%28 : i64)
  ^bb3:  // pred: ^bb1
    %29 = llvm.mlir.null : !llvm.ptr<i32>
    %30 = llvm.getelementptr %29[129600] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
    %31 = llvm.ptrtoint %30 : !llvm.ptr<i32> to i64
    %32 = llvm.call @malloc(%31) : (i64) -> !llvm.ptr<i8>
    %33 = llvm.bitcast %32 : !llvm.ptr<i8> to !llvm.ptr<i32>
    llvm.br ^bb4(%4 : i64)
  ^bb4(%34: i64):  // 2 preds: ^bb3, ^bb5
    %35 = llvm.icmp "slt" %34, %3 : i64
    llvm.cond_br %35, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %36 = llvm.srem %34, %7  : i64
    %37 = llvm.sdiv %34, %7  : i64
    %38 = llvm.mul %37, %1  : i64
    %39 = llvm.add %38, %0  : i64
    %40 = llvm.mul %36, %2  : i64
    %41 = llvm.add %39, %40  : i64
    %42 = llvm.add %41, %4  : i64
    %43 = llvm.getelementptr %21[%42] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %44 = llvm.load %43 : !llvm.ptr<i32>
    %45 = llvm.mul %37, %1  : i64
    %46 = llvm.mul %36, %2  : i64
    %47 = llvm.add %45, %46  : i64
    %48 = llvm.add %47, %4  : i64
    %49 = llvm.getelementptr %21[%48] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %50 = llvm.load %49 : !llvm.ptr<i32>
    %51 = llvm.mul %37, %1  : i64
    %52 = llvm.add %51, %5  : i64
    %53 = llvm.mul %36, %2  : i64
    %54 = llvm.add %52, %53  : i64
    %55 = llvm.add %54, %4  : i64
    %56 = llvm.getelementptr %21[%55] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %57 = llvm.load %56 : !llvm.ptr<i32>
    %58 = llvm.getelementptr %14[%37] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %59 = llvm.load %58 : !llvm.ptr<i32>
    %60 = llvm.ashr %57, %10  : i32
    %61 = llvm.shl %60, %11  : i32
    %62 = llvm.ashr %50, %10  : i32
    %63 = llvm.shl %62, %12  : i32
    %64 = llvm.add %63, %61  : i32
    %65 = llvm.ashr %44, %10  : i32
    %66 = llvm.add %65, %64  : i32
    %67 = llvm.add %66, %59  : i32
    %68 = llvm.mul %37, %7  : i64
    %69 = llvm.add %68, %36  : i64
    %70 = llvm.add %69, %4  : i64
    %71 = llvm.getelementptr %33[%70] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %67, %71 : !llvm.ptr<i32>
    %72 = llvm.add %34, %5  : i64
    llvm.br ^bb4(%72 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @free(%20) : (!llvm.ptr<i8>) -> ()
    llvm.br ^bb7(%4 : i64)
  ^bb7(%73: i64):  // 2 preds: ^bb6, ^bb8
    %74 = llvm.icmp "slt" %73, %8 : i64
    llvm.cond_br %74, ^bb8, ^bb9(%4 : i64)
  ^bb8:  // pred: ^bb7
    %75 = llvm.getelementptr %arg14[%73] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %9, %75 : !llvm.ptr<i32>
    %76 = llvm.add %73, %5  : i64
    llvm.br ^bb7(%76 : i64)
  ^bb9(%77: i64):  // 2 preds: ^bb7, ^bb10
    %78 = llvm.icmp "slt" %77, %3 : i64
    llvm.cond_br %78, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %79 = llvm.srem %77, %7  : i64
    %80 = llvm.sdiv %77, %7  : i64
    %81 = llvm.mul %80, %7  : i64
    %82 = llvm.add %81, %79  : i64
    %83 = llvm.add %82, %4  : i64
    %84 = llvm.getelementptr %33[%83] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %85 = llvm.load %84 : !llvm.ptr<i32>
    %86 = llvm.sext %85 : i32 to i64
    %87 = llvm.getelementptr %arg14[%86] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %88 = llvm.load %87 : !llvm.ptr<i32>
    %89 = llvm.mul %80, %7  : i64
    %90 = llvm.add %89, %79  : i64
    %91 = llvm.getelementptr %16[%90] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %92 = llvm.load %91 : !llvm.ptr<i32>
    %93 = llvm.add %88, %92  : i32
    %94 = llvm.getelementptr %arg14[%86] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %93, %94 : !llvm.ptr<i32>
    %95 = llvm.add %77, %5  : i64
    llvm.br ^bb9(%95 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @free(%32) : (!llvm.ptr<i8>) -> ()
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