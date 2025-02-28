// RUN: byteir-opt %s -byre-host | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    func.func @Unknown0(%arg0: memref<i64>, %arg1: memref<i64>, %arg2: memref<1x97xf32>) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 6.28318548 : f32
      %c-1879881855_i32 = arith.constant -1879881855 : i32
      %c-616729560_i32 = arith.constant -616729560 : i32
      %c534103459_i32 = arith.constant 534103459 : i32
      %c1401181199_i32 = arith.constant 1401181199 : i32
      %c1684936478_i32 = arith.constant 1684936478 : i32
      %c-1253254570_i32 = arith.constant -1253254570 : i32
      %c-1459197799_i32 = arith.constant -1459197799 : i32
      %c387276957_i32 = arith.constant 387276957 : i32
      %c-308364780_i32 = arith.constant -308364780 : i32
      %c2027808484_i32 = arith.constant 2027808484 : i32
      %c842468239_i32 = arith.constant 842468239 : i32
      %c-626627285_i32 = arith.constant -626627285 : i32
      %c1993301258_i32 = arith.constant 1993301258 : i32
      %c1013904242_i32 = arith.constant 1013904242 : i32
      %cst_1 = arith.constant -2.000000e+00 : f32
      %cst_2 = arith.constant 2.3283064365386963E-10 : f64
      %c-1150833019_i32 = arith.constant -1150833019 : i32
      %c-1640531527_i32 = arith.constant -1640531527 : i32
      %c32_i64 = arith.constant 32 : i64
      %c3449720151_i64 = arith.constant 3449720151 : i64
      %c3528531795_i64 = arith.constant 3528531795 : i64
      %c12345_i32 = arith.constant 12345 : i32
      %c1103515245_i32 = arith.constant 1103515245 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c97 = arith.constant 97 : index
      scf.for %arg3 = %c0 to %c97 step %c1 {
        %0 = memref.load %arg0[] : memref<i64>
        %1 = memref.load %arg1[] : memref<i64>
        %2 = arith.trunci %0 : i64 to i32
        %3 = arith.trunci %1 : i64 to i32
        %4 = arith.addi %2, %3 : i32
        %5 = arith.muli %4, %c1103515245_i32 : i32
        %6 = arith.addi %5, %c12345_i32 : i32
        %7 = arith.index_cast %arg3 : index to i32
        %8 = arith.addi %7, %6 : i32
        %9 = arith.muli %8, %c1103515245_i32 : i32
        %10 = arith.addi %9, %c12345_i32 : i32
        %11 = arith.extui %10 : i32 to i64
        %12 = arith.muli %11, %c3528531795_i64 : i64
        %13 = arith.trunci %12 : i64 to i32
        %14 = arith.shrui %12, %c32_i64 : i64
        %15 = arith.trunci %14 : i64 to i32
        %16 = arith.xori %15, %3 : i32
        %17 = arith.addi %2, %c-1640531527_i32 : i32
        %18 = arith.addi %3, %c-1150833019_i32 : i32
        %19 = arith.extui %2 : i32 to i64
        %20 = arith.extui %16 : i32 to i64
        %21 = arith.muli %19, %c3528531795_i64 : i64
        %22 = arith.trunci %21 : i64 to i32
        %23 = arith.shrui %21, %c32_i64 : i64
        %24 = arith.trunci %23 : i64 to i32
        %25 = arith.muli %20, %c3449720151_i64 : i64
        %26 = arith.trunci %25 : i64 to i32
        %27 = arith.shrui %25, %c32_i64 : i64
        %28 = arith.trunci %27 : i64 to i32
        %29 = arith.xori %28, %17 : i32
        %30 = arith.xori %24, %13 : i32
        %31 = arith.xori %30, %18 : i32
        %32 = arith.addi %2, %c1013904242_i32 : i32
        %33 = arith.addi %3, %c1993301258_i32 : i32
        %34 = arith.extui %29 : i32 to i64
        %35 = arith.extui %31 : i32 to i64
        %36 = arith.muli %34, %c3528531795_i64 : i64
        %37 = arith.trunci %36 : i64 to i32
        %38 = arith.shrui %36, %c32_i64 : i64
        %39 = arith.trunci %38 : i64 to i32
        %40 = arith.muli %35, %c3449720151_i64 : i64
        %41 = arith.trunci %40 : i64 to i32
        %42 = arith.shrui %40, %c32_i64 : i64
        %43 = arith.trunci %42 : i64 to i32
        %44 = arith.xori %43, %26 : i32
        %45 = arith.xori %44, %32 : i32
        %46 = arith.xori %39, %22 : i32
        %47 = arith.xori %46, %33 : i32
        %48 = arith.addi %2, %c-626627285_i32 : i32
        %49 = arith.addi %3, %c842468239_i32 : i32
        %50 = arith.extui %45 : i32 to i64
        %51 = arith.extui %47 : i32 to i64
        %52 = arith.muli %50, %c3528531795_i64 : i64
        %53 = arith.trunci %52 : i64 to i32
        %54 = arith.shrui %52, %c32_i64 : i64
        %55 = arith.trunci %54 : i64 to i32
        %56 = arith.muli %51, %c3449720151_i64 : i64
        %57 = arith.trunci %56 : i64 to i32
        %58 = arith.shrui %56, %c32_i64 : i64
        %59 = arith.trunci %58 : i64 to i32
        %60 = arith.xori %59, %41 : i32
        %61 = arith.xori %60, %48 : i32
        %62 = arith.xori %55, %37 : i32
        %63 = arith.xori %62, %49 : i32
        %64 = arith.addi %2, %c2027808484_i32 : i32
        %65 = arith.addi %3, %c-308364780_i32 : i32
        %66 = arith.extui %61 : i32 to i64
        %67 = arith.extui %63 : i32 to i64
        %68 = arith.muli %66, %c3528531795_i64 : i64
        %69 = arith.trunci %68 : i64 to i32
        %70 = arith.shrui %68, %c32_i64 : i64
        %71 = arith.trunci %70 : i64 to i32
        %72 = arith.muli %67, %c3449720151_i64 : i64
        %73 = arith.trunci %72 : i64 to i32
        %74 = arith.shrui %72, %c32_i64 : i64
        %75 = arith.trunci %74 : i64 to i32
        %76 = arith.xori %75, %57 : i32
        %77 = arith.xori %76, %64 : i32
        %78 = arith.xori %71, %53 : i32
        %79 = arith.xori %78, %65 : i32
        %80 = arith.addi %2, %c387276957_i32 : i32
        %81 = arith.addi %3, %c-1459197799_i32 : i32
        %82 = arith.extui %77 : i32 to i64
        %83 = arith.extui %79 : i32 to i64
        %84 = arith.muli %82, %c3528531795_i64 : i64
        %85 = arith.trunci %84 : i64 to i32
        %86 = arith.shrui %84, %c32_i64 : i64
        %87 = arith.trunci %86 : i64 to i32
        %88 = arith.muli %83, %c3449720151_i64 : i64
        %89 = arith.trunci %88 : i64 to i32
        %90 = arith.shrui %88, %c32_i64 : i64
        %91 = arith.trunci %90 : i64 to i32
        %92 = arith.xori %91, %73 : i32
        %93 = arith.xori %92, %80 : i32
        %94 = arith.xori %87, %69 : i32
        %95 = arith.xori %94, %81 : i32
        %96 = arith.addi %2, %c-1253254570_i32 : i32
        %97 = arith.addi %3, %c1684936478_i32 : i32
        %98 = arith.extui %93 : i32 to i64
        %99 = arith.extui %95 : i32 to i64
        %100 = arith.muli %98, %c3528531795_i64 : i64
        %101 = arith.trunci %100 : i64 to i32
        %102 = arith.shrui %100, %c32_i64 : i64
        %103 = arith.trunci %102 : i64 to i32
        %104 = arith.muli %99, %c3449720151_i64 : i64
        %105 = arith.trunci %104 : i64 to i32
        %106 = arith.shrui %104, %c32_i64 : i64
        %107 = arith.trunci %106 : i64 to i32
        %108 = arith.xori %107, %89 : i32
        %109 = arith.xori %108, %96 : i32
        %110 = arith.xori %103, %85 : i32
        %111 = arith.xori %110, %97 : i32
        %112 = arith.addi %2, %c1401181199_i32 : i32
        %113 = arith.addi %3, %c534103459_i32 : i32
        %114 = arith.extui %109 : i32 to i64
        %115 = arith.extui %111 : i32 to i64
        %116 = arith.muli %114, %c3528531795_i64 : i64
        %117 = arith.trunci %116 : i64 to i32
        %118 = arith.shrui %116, %c32_i64 : i64
        %119 = arith.trunci %118 : i64 to i32
        %120 = arith.muli %115, %c3449720151_i64 : i64
        %121 = arith.shrui %120, %c32_i64 : i64
        %122 = arith.trunci %121 : i64 to i32
        %123 = arith.xori %122, %105 : i32
        %124 = arith.xori %123, %112 : i32
        %125 = arith.xori %119, %101 : i32
        %126 = arith.xori %125, %113 : i32
        %127 = arith.addi %3, %c-616729560_i32 : i32
        %128 = arith.extui %124 : i32 to i64
        %129 = arith.extui %126 : i32 to i64
        %130 = arith.muli %128, %c3528531795_i64 : i64
        %131 = arith.shrui %130, %c32_i64 : i64
        %132 = arith.trunci %131 : i64 to i32
        %133 = arith.muli %129, %c3449720151_i64 : i64
        %134 = arith.trunci %133 : i64 to i32
        %135 = arith.xori %132, %117 : i32
        %136 = arith.xori %135, %127 : i32
        %137 = arith.addi %2, %c-1879881855_i32 : i32
        %138 = arith.extui %136 : i32 to i64
        %139 = arith.muli %138, %c3449720151_i64 : i64
        %140 = arith.trunci %139 : i64 to i32
        %141 = arith.shrui %139, %c32_i64 : i64
        %142 = arith.trunci %141 : i64 to i32
        %143 = arith.xori %142, %134 : i32
        %144 = arith.xori %143, %137 : i32
        %145 = arith.extui %144 : i32 to i64
        %146 = arith.uitofp %145 : i64 to f64
        %147 = arith.mulf %146, %cst_2 : f64
        %148 = arith.truncf %147 : f64 to f32
        %149 = arith.extui %140 : i32 to i64
        %150 = arith.uitofp %149 : i64 to f64
        %151 = arith.mulf %150, %cst_2 : f64
        %152 = arith.truncf %151 : f64 to f32
        %153 = math.log %148 : f32
        %154 = arith.mulf %153, %cst_1 : f32
        %155 = math.sqrt %154 : f32
        %156 = arith.mulf %152, %cst_0 : f32
        %157 = math.cos %156 : f32
        %158 = arith.mulf %155, %157 : f32
        %159 = arith.addf %158, %cst : f32
        memref.store %159, %arg2[%c0, %arg3] : memref<1x97xf32>
      }
      return
    }
  }
  func.func @main(%arg0: memref<1x97xf32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %alloc = memref.alloc() : memref<256xi8, "cpu">
    %0 = "byre.alias"(%alloc) <{offset = 0 : i64}> : (memref<256xi8, "cpu">) -> memref<i64, "cpu">
    byre.compute @GetSeed(%0) {device = "cpu", memory_effects = [2 : i32]} : memref<i64, "cpu">
    %1 = "byre.alias"(%alloc) <{offset = 128 : i64}> : (memref<256xi8, "cpu">) -> memref<i64, "cpu">
    byre.compute @NextOffset(%1) {device = "cpu", memory_effects = [2 : i32]} : memref<i64, "cpu">
    byre.compute @LLVMJITOp(%0, %1, %arg0) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<i64, "cpu">, memref<i64, "cpu">, memref<1x97xf32, "cpu">
    return
  }
}