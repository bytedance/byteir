// RUN: byteir-opt %s -linalg-prefetch="count=2" -cse -canonicalize | FileCheck %s -check-prefix=NOUNROLL
// RUN: byteir-opt %s -linalg-prefetch="count=2 unroll" -cse -canonicalize | FileCheck %s -check-prefix=UNROLL

#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
func.func @matmul_tiled(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %0 = memref.alloc() : memref<8x64xf32, 1>
  %1 = memref.alloc() : memref<64x64xf32, 2>
  %2 = memref.alloc() : memref<8x64xf32, 3>
  scf.for %arg3 = %c0 to %c128 step %c8 {
    %3 = memref.subview %arg0[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    %4 = memref.subview %arg2[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    linalg.copy {__byteir_prefetch__} ins(%3 : memref<8x64xf32, #map>) outs(%0 : memref<8x64xf32, 1>)
    linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%1 : memref<64x64xf32, 2>)
    linalg.matmul ins(%0, %1 : memref<8x64xf32, 1>, memref<64x64xf32, 2>) outs(%2 : memref<8x64xf32, 3>)
    linalg.copy ins(%2 : memref<8x64xf32, 3>) outs(%4 : memref<8x64xf32, #map>)
  }
  return
}
// NOUNROLL-LABEL: func.func @matmul_tiled
// NOUNROLL: %[[V0:.*]] = memref.alloc() : memref<8x64xf32, 1>
// NOUNROLL: %[[V1:.*]] = memref.alloc() : memref<64x64xf32, 2>
// NOUNROLL: %[[V2:.*]] = memref.alloc() : memref<8x64xf32, 3>
// NOUNROLL: %[[V3:.*]] = memref.subview %arg0[0, 0] [8, 64] [1, 1]
// NOUNROLL: linalg.copy ins(%[[V3]] : {{.*}}) outs(%[[V0]] : {{.*}})
// NOUNROLL: %[[V4:.*]] = memref.subview %arg0[8, 0] [8, 64] [1, 1]
// NOUNROLL: %[[V5:.*]] = memref.alloc() : memref<8x64xf32, 1>
// NOUNROLL: linalg.copy ins(%[[V4]] : {{.*}}) outs(%[[V5]] : {{.*}})
// NOUNROLL: %[[V6:.*]] = memref.alloc() : memref<8x64xf32, 1>
// NOUNROLL: scf.for %arg3 = %c0 to %c128 step %c8
// NOUNROLL:   %[[V7:.*]] = memref.subview %arg2[%arg3, 0] [8, 64] [1, 1]
// NOUNROLL:   %[[V8:.*]] = arith.addi %arg3, %c16
// NOUNROLL:   %[[V9:.*]] = arith.cmpi slt, %[[V8]], %c128
// NOUNROLL:   scf.if %[[V9]]
// NOUNROLL:     %[[V10:.*]] = memref.subview %arg0[%[[V8]], 0] [8, 64] [1, 1]
// NOUNROLL:     linalg.copy ins(%[[V10]] : {{.*}}) outs(%[[V6]] : {{.*}})
// NOUNROLL:   linalg.copy ins(%arg1 : {{.*}}) outs(%[[V1]] : {{.*}})
// NOUNROLL:   linalg.matmul ins(%[[V0]], %[[V1]]
// NOUNROLL:   linalg.copy ins(%[[V2]] : {{.*}}) outs(%[[V7]] : {{.*}})
// NOUNROLL:   linalg.copy ins(%[[V5]] : {{.*}}) outs(%[[V0]] : {{.*}})
// NOUNROLL:   linalg.copy ins(%[[V6]] : {{.*}}) outs(%[[V5]] : {{.*}})

// UNROLL-LABEL: func.func @matmul_tiled
// UNROLL: %[[V0:.*]] = memref.alloc() : memref<8x64xf32, 1>
// UNROLL: %[[V1:.*]] = memref.alloc() : memref<64x64xf32, 2>
// UNROLL: %[[V2:.*]] = memref.alloc() : memref<8x64xf32, 3>
// UNROLL: %[[V3:.*]] = memref.subview %arg0[0, 0] [8, 64] [1, 1]
// UNROLL: linalg.copy ins(%[[V3]] : {{.*}}) outs(%[[V0]] : {{.*}})
// UNROLL: %[[V4:.*]] = memref.subview %arg0[8, 0] [8, 64] [1, 1]
// UNROLL: %[[V5:.*]] = memref.alloc() : memref<8x64xf32, 1>
// UNROLL: linalg.copy ins(%[[V4]] : {{.*}}) outs(%[[V5]] : {{.*}})
// UNROLL: %[[V6:.*]] = memref.alloc() : memref<8x64xf32, 1>
// UNROLL: scf.for %arg3 = %c0 to %c128 step %c24
// UNROLL:   %[[V7:.*]] = memref.subview %arg2[%arg3, 0] [8, 64] [1, 1]
// UNROLL:   %[[V8:.*]] = arith.addi %arg3, %c16 : index
// UNROLL:   %[[V9:.*]] = arith.cmpi slt, %[[V8]], %c128
// UNROLL:   scf.if %[[V9]]
// UNROLL:     %[[V12:.*]] = memref.subview %arg0[%[[V8]], 0] [8, 64] [1, 1]
// UNROLL:     linalg.copy ins(%[[V12]] : {{.*}}) outs(%[[V6]] : {{.*}})
// UNROLL:   linalg.copy ins(%arg1 : {{.*}}) outs(%[[V1]] : {{.*}})
// UNROLL:   linalg.matmul ins(%[[V0]], %[[V1]]
// UNROLL:   linalg.copy ins(%[[V2]] : {{.*}}) outs(%[[V7]] : {{.*}})
// UNROLL:   %[[V10:.*]] = arith.addi %arg3, %c8
// UNROLL:   %[[V11:.*]] = arith.cmpi slt, %[[V10]], %c128 
// UNROLL:   scf.if %[[V11]]
// UNROLL:     %[[V12:.*]] = memref.subview %arg2[%[[V10]], 0] [8, 64] [1, 1] 
// UNROLL:     %[[V13:.*]] = arith.addi %arg3, %c24
// UNROLL:     %[[V14:.*]] = arith.cmpi slt, %[[V13]], %c128
// UNROLL:     scf.if %[[V14]]
// UNROLL:       %[[V15:.*]] = memref.subview %arg0[%[[V13]], 0] [8, 64] [1, 1]
// UNROLL:       linalg.copy ins(%[[V15]] : {{.*}}) outs(%[[V0]] : {{.*}})
// UNROLL:     linalg.copy ins(%arg1 : {{.*}}) outs(%[[V1]] : {{.*}})
// UNROLL:     linalg.matmul ins(%[[V5]], %[[V1]]
// UNROLL:     linalg.copy ins(%[[V2]] : {{.*}}) outs(%[[V12]] : {{.*}})
// UNROLL:   scf.if %[[V9]]
// UNROLL:     %[[V12:.*]] = memref.subview %arg2[%[[V8]], 0] [8, 64] [1, 1] 
// UNROLL:     %[[V13:.*]] = arith.addi %arg3, %c32
// UNROLL:     %[[V14:.*]] = arith.cmpi slt, %[[V13]], %c128
// UNROLL:     scf.if %[[V14]]
// UNROLL:       %[[V15:.*]] = memref.subview %arg0[%[[V13]], 0] [8, 64] [1, 1]
// UNROLL:       linalg.copy ins(%[[V15]] : {{.*}}) outs(%[[V5]] : {{.*}})
// UNROLL:     linalg.copy ins(%arg1 : {{.*}}) outs(%[[V1]] : {{.*}})
// UNROLL:     linalg.matmul ins(%[[V6]], %[[V1]]
// UNROLL:     linalg.copy ins(%[[V2]] : {{.*}}) outs(%[[V12]] : {{.*}})
