// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: constant
// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module, llvm.data_layout = ""} {
  llvm.func @free(!llvm.ptr<i8>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private constant @__constant_100x1296xi32(dense<1> : tensor<100x1296xi32>) {addr_space = 0 : i32} : !llvm.array<100 x array<1296 x i32>>
  llvm.mlir.global private constant @__constant_100xi32(dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]> : tensor<100xi32>) {addr_space = 0 : i32} : !llvm.array<100 x i32>
  llvm.mlir.global private constant @__constant_51200xi32(dense<0> : tensor<51200xi32>) {addr_space = 0 : i32} : !llvm.array<51200 x i32>
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
    %9 = llvm.mlir.constant(5 : i32) : i32
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.mlir.constant(6 : i32) : i32
    %12 = llvm.mlir.addressof @__constant_51200xi32 : !llvm.ptr<array<51200 x i32>>
    %13 = llvm.getelementptr %12[0, 0] : (!llvm.ptr<array<51200 x i32>>) -> !llvm.ptr<i32>
    %14 = llvm.mlir.addressof @__constant_100xi32 : !llvm.ptr<array<100 x i32>>
    %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr<array<100 x i32>>) -> !llvm.ptr<i32>
    %16 = llvm.mlir.addressof @__constant_100x1296xi32 : !llvm.ptr<array<100 x array<1296 x i32>>>
    %17 = llvm.getelementptr %16[0, 0, 0] : (!llvm.ptr<array<100 x array<1296 x i32>>>) -> !llvm.ptr<i32>
    %18 = llvm.mlir.null : !llvm.ptr<i32>
    %19 = llvm.getelementptr %18[388800] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
    %20 = llvm.ptrtoint %19 : !llvm.ptr<i32> to i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr<i8>
    %22 = llvm.bitcast %21 : !llvm.ptr<i8> to !llvm.ptr<i32>
    llvm.br ^bb1(%4 : i64)
  ^bb1(%23: i64):  // 2 preds: ^bb0, ^bb2
    %24 = llvm.icmp "slt" %23, %6 : i64
    llvm.cond_br %24, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %25 = llvm.getelementptr %arg1[%23] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %26 = llvm.load %25 : !llvm.ptr<f32>
    %27 = llvm.fptosi %26 : f32 to i32
    %28 = llvm.getelementptr %22[%23] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %27, %28 : !llvm.ptr<i32>
    %29 = llvm.add %23, %5  : i64
    llvm.br ^bb1(%29 : i64)
  ^bb3:  // pred: ^bb1
    %30 = llvm.mlir.null : !llvm.ptr<i32>
    %31 = llvm.getelementptr %30[129600] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
    %32 = llvm.ptrtoint %31 : !llvm.ptr<i32> to i64
    %33 = llvm.call @malloc(%32) : (i64) -> !llvm.ptr<i8>
    %34 = llvm.bitcast %33 : !llvm.ptr<i8> to !llvm.ptr<i32>
    llvm.br ^bb4(%4 : i64)
  ^bb4(%35: i64):  // 2 preds: ^bb3, ^bb5
    %36 = llvm.icmp "slt" %35, %3 : i64
    llvm.cond_br %36, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %37 = llvm.srem %35, %7  : i64
    %38 = llvm.sdiv %35, %7  : i64
    %39 = llvm.mul %38, %1  : i64
    %40 = llvm.add %39, %0  : i64
    %41 = llvm.mul %37, %2  : i64
    %42 = llvm.add %40, %41  : i64
    %43 = llvm.add %42, %4  : i64
    %44 = llvm.getelementptr %22[%43] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %45 = llvm.load %44 : !llvm.ptr<i32>
    %46 = llvm.mul %38, %1  : i64
    %47 = llvm.mul %37, %2  : i64
    %48 = llvm.add %46, %47  : i64
    %49 = llvm.add %48, %4  : i64
    %50 = llvm.getelementptr %22[%49] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %51 = llvm.load %50 : !llvm.ptr<i32>
    %52 = llvm.mul %38, %1  : i64
    %53 = llvm.add %52, %5  : i64
    %54 = llvm.mul %37, %2  : i64
    %55 = llvm.add %53, %54  : i64
    %56 = llvm.add %55, %4  : i64
    %57 = llvm.getelementptr %22[%56] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %58 = llvm.load %57 : !llvm.ptr<i32>
    %59 = llvm.getelementptr %15[%38] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %60 = llvm.load %59 : !llvm.ptr<i32>
    %61 = llvm.ashr %58, %9  : i32
    %62 = llvm.shl %61, %10  : i32
    %63 = llvm.ashr %51, %9  : i32
    %64 = llvm.shl %63, %11  : i32
    %65 = llvm.add %64, %62  : i32
    %66 = llvm.ashr %45, %9  : i32
    %67 = llvm.add %66, %65  : i32
    %68 = llvm.add %67, %60  : i32
    %69 = llvm.mul %38, %7  : i64
    %70 = llvm.add %69, %37  : i64
    %71 = llvm.add %70, %4  : i64
    %72 = llvm.getelementptr %34[%71] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %68, %72 : !llvm.ptr<i32>
    %73 = llvm.add %35, %5  : i64
    llvm.br ^bb4(%73 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @free(%21) : (!llvm.ptr<i8>) -> ()
    llvm.br ^bb7(%4 : i64)
  ^bb7(%74: i64):  // 2 preds: ^bb6, ^bb8
    %75 = llvm.icmp "slt" %74, %8 : i64
    llvm.cond_br %75, ^bb8, ^bb9(%4 : i64)
  ^bb8:  // pred: ^bb7
    %76 = llvm.getelementptr %13[%74] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %77 = llvm.load %76 : !llvm.ptr<i32>
    %78 = llvm.getelementptr %arg14[%74] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %77, %78 : !llvm.ptr<i32>
    %79 = llvm.add %74, %5  : i64
    llvm.br ^bb7(%79 : i64)
  ^bb9(%80: i64):  // 2 preds: ^bb7, ^bb10
    %81 = llvm.icmp "slt" %80, %3 : i64
    llvm.cond_br %81, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %82 = llvm.srem %80, %7  : i64
    %83 = llvm.sdiv %80, %7  : i64
    %84 = llvm.mul %83, %7  : i64
    %85 = llvm.add %84, %82  : i64
    %86 = llvm.add %85, %4  : i64
    %87 = llvm.getelementptr %34[%86] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %88 = llvm.load %87 : !llvm.ptr<i32>
    %89 = llvm.sext %88 : i32 to i64
    %90 = llvm.getelementptr %arg14[%89] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %91 = llvm.load %90 : !llvm.ptr<i32>
    %92 = llvm.mul %83, %7  : i64
    %93 = llvm.add %92, %82  : i64
    %94 = llvm.getelementptr %17[%93] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %95 = llvm.load %94 : !llvm.ptr<i32>
    %96 = llvm.add %91, %95  : i32
    %97 = llvm.getelementptr %arg14[%89] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %96, %97 : !llvm.ptr<i32>
    %98 = llvm.add %80, %5  : i64
    llvm.br ^bb9(%98 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @free(%33) : (!llvm.ptr<i8>) -> ()
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