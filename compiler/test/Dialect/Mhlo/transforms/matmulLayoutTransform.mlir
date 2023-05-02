// RUN: byteir-opt %s -matmul-layout-transform="transpose-const-only=0 target-layout=rcr" | FileCheck %s
// RUN: byteir-opt %s -matmul-layout-transform="transpose-const-only=1 target-layout=rcr" | FileCheck %s --check-prefix CONSTONLY

func.func @MatmulOp0(%arg0: tensor<128x256xf32>) -> tensor<128x30522xf32> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<256x30522xf32>
  %1 = "mhlo.dot"(%arg0, %0): (tensor<128x256xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
  return %1 : tensor<128x30522xf32>
}
// CHECK-LABEL: func.func @MatmulOp0
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.dot_general
// CHECK-NEXT: return
// CONSTONLY-LABEL: func.func @MatmulOp0
// CONSTONLY-NEXT: mhlo.constant
// CONSTONLY-NEXT: mhlo.dot_general
// CONSTONLY-NEXT: return
// TRANSPOSEOUTPUT-LABEL: func.func @MatmulOp0
// TRANSPOSEOUTPUT-NEXT: mhlo.constant
// TRANSPOSEOUTPUT-NEXT: mhlo.einsum
// TRANSPOSEOUTPUT-NEXT: mhlo.transpose
// TRANSPOSEOUTPUT-NEXT: return

func.func @MatmulOp1(%arg0: tensor<256x30522xf32>) -> tensor<128x30522xf32> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<128x256xf32>
  %1 = "mhlo.dot"(%0, %arg0): (tensor<128x256xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
  return %1 : tensor<128x30522xf32>
}
// CHECK-LABEL: func.func @MatmulOp1
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.dot_general
// CHECK-NEXT: return
// CONSTONLY-LABEL: func.func @MatmulOp1
// CONSTONLY-NEXT: mhlo.constant
// CONSTONLY-NEXT: mhlo.dot
// CONSTONLY-NEXT: return
// TRANSPOSEOUTPUT-LABEL: func.func @MatmulOp1
// TRANSPOSEOUTPUT-NEXT: mhlo.constant
// TRANSPOSEOUTPUT-NEXT: mhlo.einsum
// TRANSPOSEOUTPUT-NEXT: mhlo.transpose
// TRANSPOSEOUTPUT-NEXT: return
