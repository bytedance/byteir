// RUN: byteir-opt %s -memory-planning --canonicalize --cse | FileCheck %s
// RUN: byteir-opt %s -memory-planning="alignment=64" --canonicalize --cse | byteir-stat --alloc-cnt | FileCheck %s --check-prefix CHECK-STAT
// RUN: byteir-opt %s -memory-planning="alloca" --canonicalize --cse | FileCheck %s --check-prefix CHECK-ALLOCA

func.func @test_basic_reuse(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>) -> memref<256xf32> attributes {__placeholder__byre.entry_point} {
  %0 = memref.alloc() : memref<256xf32>
  %1 = memref.alloc() : memref<256xf32>
  %2 = memref.alloc() : memref<256xf32>
  %3 = memref.alloc() : memref<256xf32>
  %4 = memref.alloc() : memref<256xf32>
  "lmhlo.add"(%arg0, %arg1, %0) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %0, %1) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %1, %2) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %2, %3) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %3, %4) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  return %4 : memref<256xf32>
}

// CHECK-LABEL: func.func @test_basic_reuse
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   memref.alloc
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   return

// CHECK-STAT-LABEL: test_basic_reuse
//  CHECK-STAT:   total_static_allocated_memory = 2048

func.func @test_align_to_64(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>) {
  %0 = memref.alloc() : memref<4xf32>
  %1 = memref.alloc() : memref<4xf32>
  %2 = memref.alloc() : memref<4xf32>
  %3 = memref.alloc() : memref<4xf32>
  "lmhlo.add"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  "lmhlo.add"(%arg1, %0, %1) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  "lmhlo.add"(%arg1, %1, %2) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  "lmhlo.add"(%arg1, %2, %3) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  "lmhlo.add"(%arg1, %3, %arg2) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  return 
}

// CHECK-LABEL: func.func @test_align_to_64
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   memref.alloc
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   return

// CHECK-STAT-LABEL: test_align_to_64
//  CHECK-STAT:   total_static_allocated_memory = 128

func.func @test_reuse_sub_chunk(%arg0 : memref<512xf32>, %arg1 : memref<128xf32>, %arg2 : memref<512xf32>, %arg3 : memref<128xf32>) {
  %0 = memref.alloc() : memref<512xf32>
  %1 = memref.alloc() : memref<128xf32>
  "lmhlo.add"(%arg0, %arg0, %0) : (memref<512xf32>, memref<512xf32>,  memref<512xf32>) -> ()
  "lmhlo.add"(%arg0, %0, %arg2) : (memref<512xf32>, memref<512xf32>,  memref<512xf32>) -> ()
  "lmhlo.add"(%arg1, %arg1, %1) : (memref<128xf32>, memref<128xf32>,  memref<128xf32>) -> ()
  "lmhlo.add"(%arg1, %1, %arg3) : (memref<128xf32>, memref<128xf32>,  memref<128xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @test_reuse_sub_chunk
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   memref.alloc
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   return

// CHECK-STAT-LABEL: test_reuse_sub_chunk
//  CHECK-STAT:   total_static_allocated_memory = 2048

func.func @test_reuse_sub_chunk_2(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %arg2 : memref<128xf32>, %arg3 : memref<128xf32>) {
  %0 = memref.alloc() : memref<512xf32>
  %1 = memref.alloc() : memref<128xf32>
  %2 = memref.alloc() : memref<128xf32>
  "lmhlo.add"(%arg0, %arg0, %0) : (memref<512xf32>, memref<512xf32>,  memref<512xf32>) -> ()
  "lmhlo.add"(%arg0, %0, %arg1) : (memref<512xf32>, memref<512xf32>,  memref<512xf32>) -> ()
  "lmhlo.add"(%arg2, %arg2, %1) : (memref<128xf32>, memref<128xf32>,  memref<128xf32>) -> ()
  "lmhlo.add"(%arg2, %1, %2) : (memref<128xf32>, memref<128xf32>,  memref<128xf32>) -> ()
  "lmhlo.add"(%1, %2, %arg3) : (memref<128xf32>, memref<128xf32>,  memref<128xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @test_reuse_sub_chunk_2
//  CHECK-DAG:   arith.constant 0
//  CHECK-DAG:   arith.constant 512
//  CHECK-NEXT:   memref.alloc
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   memref.view
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   lmhlo.add
//  CHECK-NEXT:   return

// CHECK-STAT-LABEL: test_reuse_sub_chunk_2
//  CHECK-STAT:   total_static_allocated_memory = 2048

func.func @test_noreuse_multi_memory_space(%arg0 : memref<512xf32, 1>, %arg1 : memref<512xf32, 2>) {
  %0 = memref.alloc() : memref<512xf32, 1>
  %1 = memref.alloc() : memref<512xf32, 2>
  "lmhlo.add"(%arg0, %arg0, %0) : (memref<512xf32, 1>, memref<512xf32, 1>,  memref<512xf32, 1>) -> ()
  "lmhlo.add"(%0, %0, %arg0) : (memref<512xf32, 1>, memref<512xf32, 1>,  memref<512xf32, 1>) -> ()
  "lmhlo.add"(%arg1, %arg1, %1) : (memref<512xf32, 2>, memref<512xf32, 2>,  memref<512xf32, 2>) -> ()
  "lmhlo.add"(%1, %1, %arg1) : (memref<512xf32, 2>, memref<512xf32, 2>,  memref<512xf32, 2>) -> ()
  return
}
// CHECK-STAT-LABEL: test_noreuse_multi_memory_space
//  CHECK-STAT:   total_static_allocated_memory = 4096


func.func @test_reuse_multi_memory_space(%arg0 : memref<512xf32, 1>, %arg1 : memref<512xf32, 2>) {
  %0 = memref.alloc() : memref<512xf32, 1>
  %1 = memref.alloc() : memref<512xf32, 2>
  %2 = memref.alloc() : memref<512xf32, 1>
  %3 = memref.alloc() : memref<512xf32, 2>
  "lmhlo.add"(%arg0, %arg0, %0) : (memref<512xf32, 1>, memref<512xf32, 1>,  memref<512xf32, 1>) -> ()
  "lmhlo.add"(%arg1, %arg1, %1) : (memref<512xf32, 2>, memref<512xf32, 2>,  memref<512xf32, 2>) -> ()
  "lmhlo.add"(%0, %0, %arg0) : (memref<512xf32, 1>, memref<512xf32, 1>,  memref<512xf32, 1>) -> ()
  "lmhlo.add"(%1, %1, %arg1) : (memref<512xf32, 2>, memref<512xf32, 2>,  memref<512xf32, 2>) -> ()
  "lmhlo.add"(%arg0, %arg0, %2) : (memref<512xf32, 1>, memref<512xf32, 1>,  memref<512xf32, 1>) -> ()
  "lmhlo.add"(%arg1, %arg1, %3) : (memref<512xf32, 2>, memref<512xf32, 2>,  memref<512xf32, 2>) -> ()
  "lmhlo.add"(%2, %2, %arg0) : (memref<512xf32, 1>, memref<512xf32, 1>,  memref<512xf32, 1>) -> ()
  "lmhlo.add"(%3, %3, %arg1) : (memref<512xf32, 2>, memref<512xf32, 2>,  memref<512xf32, 2>) -> ()
  return
}
// CHECK-STAT-LABEL: test_reuse_multi_memory_space
//  CHECK-STAT:   total_static_allocated_memory = 4096

func.func @test_basic_reuse_alloca(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>) -> memref<256xf32> attributes {__placeholder__byre.entry_point} {
  %0 = memref.alloca() : memref<256xf32>
  %1 = memref.alloca() : memref<256xf32>
  %2 = memref.alloca() : memref<256xf32>
  %3 = memref.alloca() : memref<256xf32>
  %4 = memref.alloca() : memref<256xf32>
  "lmhlo.add"(%arg0, %arg1, %0) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %0, %1) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %1, %2) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %2, %3) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %3, %4) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  return %4 : memref<256xf32>
}
// CHECK-ALLOCA-LABEL: func.func @test_basic_reuse_alloca
//  CHECK-ALLOCA-NEXT:   arith.constant
//  CHECK-ALLOCA-NEXT:   arith.constant
//  CHECK-ALLOCA-NEXT:   memref.alloca
//  CHECK-ALLOCA-NEXT:   memref.view
//  CHECK-ALLOCA-NEXT:   memref.view
//  CHECK-ALLOCA-NEXT:   lmhlo.add
//  CHECK-ALLOCA-NEXT:   lmhlo.add
//  CHECK-ALLOCA-NEXT:   lmhlo.add
//  CHECK-ALLOCA-NEXT:   lmhlo.add
//  CHECK-ALLOCA-NEXT:   lmhlo.add
//  CHECK-ALLOCA-NEXT:   return
