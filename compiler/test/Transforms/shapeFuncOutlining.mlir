// RUN: byteir-opt %s -shape-func-outlining="entry-func-name=main" | FileCheck %s

module attributes {gpu.container_module, torch.debug_module_name = "MLPModule"} {
  memref.global "private" constant @__constant_20xf32 : memref<20xf32> = dense_resource<__elided__>
  memref.global "private" constant @__constant_10x20xf32 : memref<10x20xf32> = dense_resource<__elided__>
  func.func @main(%arg0: memref<?x10xf32>, %arg1: memref<?x20xf32>) -> memref<?x20xf32> attributes {__placeholder__byre.entry_point} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %0 = memref.get_global @__constant_10x20xf32 : memref<10x20xf32>
    %1 = memref.get_global @__constant_20xf32 : memref<20xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x10xf32>
    %alloc = memref.alloc(%dim) : memref<?x20xf32>
    byre.compute @MatmulOp_f32f32_f32(%arg0, %0, %alloc) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<?x10xf32>, memref<10x20xf32>, memref<?x20xf32>
    %alloc_0 = memref.alloc(%dim) : memref<?x20xf32>
    %2 = arith.cmpi sle, %dim, %c0 : index
    %3 = arith.subi %c0, %dim : index
    %4 = arith.subi %dim, %c1 : index
    %5 = arith.select %2, %3, %4 : index
    %6 = arith.divsi %5, %c4 : index
    %7 = arith.subi %c0, %6 : index
    %8 = arith.addi %6, %c1 : index
    %9 = arith.select %2, %7, %8 : index
    byre.compute @PTXOp(%alloc, %9, %1, %arg1, %alloc_0, %9, %c1, %c1, %c128, %c1, %c1, %c0) {byre_dynamic_launch_config, kernel_name = "Unknown0_kernel"} : memref<?x20xf32>, index, memref<20xf32>, memref<?x20xf32>, memref<?x20xf32>, index, index, index, index, index, index, index
    return %alloc_0 : memref<?x20xf32>
  }
}

// CHECK-LABEL: func.func private @ShapeComputaionFunc0
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM:.*]] = memref.dim %arg0, %[[C0]] : memref<?x10xf32>
// CHECK-DAG: %[[V0:.*]] = arith.cmpi sle, %[[DIM]], %[[C0]] : index
// CHECK-DAG: %[[V1:.*]] = arith.subi %[[DIM]], %[[C1]] : index
// CHECK-DAG: %[[V2:.*]] = arith.subi %[[C0]], %[[DIM]] : index
// CHECK-DAG: %[[V3:.*]] = arith.select %[[V0]], %[[V2]], %[[V1]] : index
// CHECK-DAG: %[[V4:.*]] = arith.divsi %[[V3]], %[[C4]] : index
// CHECK-DAG: %[[V5:.*]] = arith.addi %[[V4]], %[[C1]] : index
// CHECK-DAG: %[[V6:.*]] = arith.subi %[[C0]], %[[V4]] : index
// CHECK-DAG: %[[V7:.*]] = arith.select %[[V0]], %[[V6]], %[[V5]] : index
// CHECK-LABEL: func.func @main
// CHECK-NEXT: %[[V0:.*]]:5 = "byre.compute_shape"(%arg0) <{shape_fn = "ShapeComputaionFunc0"}> {device = "cpu"}
// CHECK: %[[ALLOC:.*]] = memref.alloc(%[[V0]]#{{.*}}) : memref<?x20xf32>
// CHECK: byre.compute @MatmulOp_f32f32_f32(%arg0, %1, %[[ALLOC]]) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<?x10xf32>, memref<10x20xf32>, memref<?x20xf32>
// CHECK: %[[ALLOC0:.*]] = memref.alloc(%[[V0]]#{{.*}}) : memref<?x20xf32>
// CHECK: byre.compute @PTXOp(%[[ALLOC]], %[[V0]]#{{.*}}, %2, %arg1, %[[ALLOC0]], %[[V0]]#{{.*}}, %[[V0]]#{{.*}}, %[[V0]]#{{.*}}, %[[V0]]#{{.*}}, %[[V0]]#{{.*}}, %[[V0]]#{{.*}}, %[[V0]]#{{.*}}) {byre_dynamic_launch_config, kernel_name = "Unknown0_kernel"} : memref<?x20xf32>, index, memref<20xf32>, memref<?x20xf32>, memref<?x20xf32>, index, index, index, index, index, index, index
