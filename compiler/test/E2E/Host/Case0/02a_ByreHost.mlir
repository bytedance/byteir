// RUN: byteir-opt %s -byre-host="device-file-name=your_file target=cpu" | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    memref.global "private" constant @__constant_1x128xi32 : memref<1x128xi32> = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000">
    func.func @Unknown0(%arg0: memref<1xi64>, %arg1: memref<1xi64>, %arg2: memref<1xi64>, %arg3: memref<1x128xi32>, %arg4: memref<1x128xi32>, %arg5: memref<1x128xi32>) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %0 = memref.get_global @__constant_1x128xi32 : memref<1x128xi32>
      scf.for %arg6 = %c0 to %c128 step %c1 {
        %1 = memref.load %0[%c0, %arg6] : memref<1x128xi32>
        %2 = memref.load %arg2[%c0] : memref<1xi64>
        %3 = memref.load %arg0[%c0] : memref<1xi64>
        %4 = memref.load %arg1[%c0] : memref<1xi64>
        %5 = arith.addi %3, %4 : i64
        %6 = arith.addi %2, %5 : i64
        %7 = arith.trunci %6 : i64 to i32
        %8 = arith.cmpi slt, %1, %7 : i32
        %9 = arith.extui %8 : i1 to i32
        memref.store %9, %arg4[%c0, %arg6] : memref<1x128xi32>
      }
      %collapse_shape = memref.collapse_shape %arg4 [[0, 1]] : memref<1x128xi32> into memref<128xi32>
      %collapse_shape_0 = memref.collapse_shape %arg3 [[0, 1]] : memref<1x128xi32> into memref<128xi32>
      %collapse_shape_1 = memref.collapse_shape %arg5 [[0, 1]] : memref<1x128xi32> into memref<128xi32>
      scf.for %arg6 = %c0 to %c128 step %c1 {
        %1 = memref.load %collapse_shape[%arg6] : memref<128xi32>
        %2 = memref.load %collapse_shape_0[%arg6] : memref<128xi32>
        %3 = arith.muli %1, %2 : i32
        memref.store %3, %collapse_shape_1[%arg6] : memref<128xi32>
      }
      return
    }
  }
  func.func @main(%arg0: memref<1xi64, "cpu"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<1xi64, "cpu"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1xi64, "cpu"> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x128xi32, "cpu"> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<1x128xi32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg5: memref<1x128xi32, "cpu"> {byre.argname = "Output1", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    byre.compute @LLVMJITOp(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<1xi64, "cpu">, memref<1xi64, "cpu">, memref<1xi64, "cpu">, memref<1x128xi32, "cpu">, memref<1x128xi32, "cpu">, memref<1x128xi32, "cpu">
    return
  }
}