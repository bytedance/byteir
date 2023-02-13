// RUN: byteir-opt %s -cmae -cse -cse -split-input-file | FileCheck %s

func.func @common_arg(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>) {
  %c1= arith.constant 1 : index
  %0 = memref.load %arg0[%c1] : memref<4xf32>
  %1 = memref.load %arg1[%c1] : memref<4xf32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %arg2[%c1] : memref<4xf32>
  %3 = memref.load %arg0[%c1] : memref<4xf32>
  %4 = memref.load %arg1[%c1] : memref<4xf32>
  %5 = arith.addf %3, %4 : f32
  memref.store %5, %arg2[%c1] : memref<4xf32>
  return 
}
// CHECK-LABEL: func.func @common_arg
// CHECK-NEXT: arith.constant
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: return

// -----

func.func @common_local(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>) {
  %c1= arith.constant 1 : index
  %0 = memref.alloc() : memref<4xf32>
  %1 = memref.alloc() : memref<4xf32>
  %2 = memref.load %arg0[%c1] : memref<4xf32>
  %3 = memref.load %0[%c1] : memref<4xf32>
  %4 = arith.addf %2, %3 : f32
  memref.store %4, %arg1[%c1] : memref<4xf32>
  memref.store %4, %1[%c1] : memref<4xf32>
  %5 = memref.load %arg0[%c1] : memref<4xf32>
  %6 = memref.load %0[%c1] : memref<4xf32>
  %7 = arith.addf %5, %6 : f32
  memref.store %7, %arg1[%c1] : memref<4xf32>
  memref.store %7, %1[%c1] : memref<4xf32>
  return 
}
// CHECK-LABEL: func.func @common_local
// CHECK-NEXT: arith.constant
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: memref.store
// CHECK-NEXT: return

// -----

func.func private @external_func1(%arg0 : memref<4xf32>)
func.func @common_with_call_before(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>) {
  %c1= arith.constant 1 : index
  call @external_func1(%arg0) : (memref<4xf32>)->()
  %0 = memref.load %arg0[%c1] : memref<4xf32>
  %1 = memref.load %arg1[%c1] : memref<4xf32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %arg2[%c1] : memref<4xf32>
  %3 = memref.load %arg0[%c1] : memref<4xf32>
  %4 = memref.load %arg1[%c1] : memref<4xf32>
  %5 = arith.addf %3, %4 : f32
  memref.store %5, %arg2[%c1] : memref<4xf32>
  return 
}
// CHECK-LABEL: func.func @common_with_call_before
// CHECK-NEXT: arith.constant
// CHECK-NEXT: call
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: return

// -----

func.func private @external_func2(%arg0 : memref<4xf32>)
func.func @common_with_call_mid(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>) {
  %c1= arith.constant 1 : index
  %0 = memref.load %arg0[%c1] : memref<4xf32>
  %1 = memref.load %arg1[%c1] : memref<4xf32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %arg2[%c1] : memref<4xf32>
  call @external_func2(%arg0) : (memref<4xf32>)->()
  %3 = memref.load %arg0[%c1] : memref<4xf32>
  %4 = memref.load %arg1[%c1] : memref<4xf32>
  %5 = arith.addf %3, %4 : f32
  memref.store %5, %arg2[%c1] : memref<4xf32>
  return 
}
// CHECK-LABEL: func.func @common_with_call_mid
// CHECK-NEXT: arith.constant
// CHECK-NEXT: memref.load
// CHECK-NEXT: call
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: return

// -----

func.func @common_scope(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>, %cond : i1) {
  %c1= arith.constant 1 : index
  %0 = memref.load %arg0[%c1] : memref<4xf32>
  %1 = memref.load %arg1[%c1] : memref<4xf32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %arg2[%c1] : memref<4xf32>
  scf.if %cond {
    %3 = memref.load %arg0[%c1] : memref<4xf32>
    %4 = memref.load %arg1[%c1] : memref<4xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %arg2[%c1] : memref<4xf32>
  }
  %6 = memref.load %arg0[%c1] : memref<4xf32>
  %7 = memref.load %arg1[%c1] : memref<4xf32>
  %8 = arith.addf %6, %7 : f32
  memref.store %8, %arg2[%c1] : memref<4xf32>
  return 
}
// CHECK-LABEL: func.func @common_scope
// CHECK-NEXT: arith.constant
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: return

// -----

func.func @common_scope_raw(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %cond : i1) {
  %c1= arith.constant 1 : index
  %c2= arith.constant 2 : index
  %0 = memref.load %arg0[%c1] : memref<4xf32>
  %1 = memref.load %arg1[%c1] : memref<4xf32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %arg0[%c1] : memref<4xf32>
  scf.if %cond {
    %6 = memref.load %arg0[%c1] : memref<4xf32>
    %7 = memref.load %arg1[%c1] : memref<4xf32>
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %arg0[%c2] : memref<4xf32>
  }
  %3 = memref.load %arg0[%c1] : memref<4xf32>
  %4 = memref.load %arg1[%c1] : memref<4xf32>
  %5 = arith.addf %3, %4 : f32
  memref.store %5, %arg0[%c1] : memref<4xf32>
  return 
}
// CHECK-LABEL: func.func @common_scope_raw
// CHECK-NEXT: arith.constant
// CHECK-NEXT: arith.constant
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: return

// -----

func.func @common_2_scope_raw(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %cond : i1) {
  %c1= arith.constant 1 : index
  %c2= arith.constant 2 : index
  %0 = memref.load %arg0[%c1] : memref<4xf32>
  %1 = memref.load %arg1[%c1] : memref<4xf32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %arg0[%c1] : memref<4xf32>
  scf.if %cond {
    %3 = memref.load %arg0[%c1] : memref<4xf32>
    %4 = memref.load %arg1[%c1] : memref<4xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %arg0[%c2] : memref<4xf32>
  }
  scf.if %cond {
   %6 = memref.load %arg0[%c1] : memref<4xf32>
   %7 = memref.load %arg1[%c1] : memref<4xf32>
   %8 = arith.addf %6, %7 : f32
   memref.store %8, %arg0[%c1] : memref<4xf32>
  }
  %9 = memref.load %arg0[%c1] : memref<4xf32>
  %10 = memref.load %arg1[%c1] : memref<4xf32>
  %11 = arith.addf %9, %10 : f32
  memref.store %11, %arg0[%c1] : memref<4xf32>
  return 
}
// CHECK-LABEL: func.func @common_2_scope_raw
// CHECK-NEXT: arith.constant
// CHECK-NEXT: arith.constant
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK: scf.if
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: return
