// RUN: byteir-opt %s -o %t
// RUN: byteir-opt --load-byre %s.bc -o %t0 && diff %t %t0
// RUN: byteir-opt --dump-byre="file-name=%t1.bc version=1.0.0" %s &>/dev/null && diff %t1.bc %s.bc
// RUN: byteir-opt --load-byre %t1.bc -o %t1 && diff %t %t1
// RUN: byteir-opt --load-byre --dump-byre="file-name=%t2.bc version=1.0.0" %s.bc &>/dev/null && diff %t2.bc %s.bc
// RUN: byteir-opt --load-byre %t2.bc -o %t2 && diff %t %t2

module attributes {byre.container_module} {
  func.func @test_compute(%arg0: memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B", byre.arg_alias_index = 0 : i64}) attributes {byre.entry_point, byteir.entry_point, tf.original_input_names} {
    %alloc = memref.alloc() : memref<4xf32>
    byre.compute @some_kernel(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
    return
  }
}
