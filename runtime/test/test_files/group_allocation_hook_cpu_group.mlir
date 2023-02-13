module attributes {byre.container_module} {
    func.func @main(%arg0: memref<32xf32, "cpu_group"> {byre.argname = "Input0", byre.argtype = 1 : i32},
               %arg1: memref<32xf32, "cpu_group"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
        %0 = memref.alloc() : memref<32xf32, "cpu_group">
        byre.compute @CheckGroupAllocationHook(%arg0, %0, %arg1) {base = 0xdeadbeef: i64} : memref<32xf32, "cpu_group">, memref<32xf32, "cpu_group">, memref<32xf32, "cpu_group">
        return
    }
}
