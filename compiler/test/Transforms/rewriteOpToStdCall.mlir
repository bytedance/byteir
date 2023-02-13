// RUN: byteir-opt %s --rewrite-op-to-std-call="call-table=linalg.matmul:matmul_impl" --split-input-file | FileCheck %s

// CHECK:  func.func private @matmul_impl(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
func.func @matmul(%A : memref<?x?xf32>, %B : memref<?x?xf32>, %C : memref<?x?xf32>) {
    // CHECK: call @matmul_impl({{.*}}, {{.*}}, {{.*}}) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>) outs(%C: memref<?x?xf32>)
    return
}
