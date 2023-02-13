// RUN: byteir-opt %s -simplify-view | FileCheck %s

#map0 = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 64 + d1 + 512)>
#map2 = affine_map<(d0, d1) -> (d0 * 64 + d1 + 640)>
#map3 = affine_map<(d0, d1) -> (d0 * 64 + d1)>
#map4 = affine_map<(d0, d1) -> (d0 * 192 + d1)>
#map5 = affine_map<(d0, d1) -> (d0 * 384 + d1)>

// CHECK-LABEL: func.func @subview_no_canonical
func.func @subview_no_canonical(%arg0: memref<128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.subview %arg0[%c8, 0] [%c8, 64] [1, 1] : memref<128x64xf32> to memref<?x64xf32, #map0>
// CHECK: memref.subview %arg0
  %1 = memref.load %arg0[%c0, %c0] : memref<128x64xf32>
  %2 = arith.addf %1, %1 : f32
  %3 = memref.subview %0[%c2, 0] [%c2, 64] [1, 1] : memref<?x64xf32, #map0> to memref<?x64xf32, #map0>
// CHECK-NOT: memref.subview %arg0
  memref.store %2, %0[%c0, %c1] : memref<?x64xf32, #map0>
  memref.store %2, %3[%c0, %c1] : memref<?x64xf32, #map0>
  return 
}

// CHECK-LABEL: func.func @subview_of_subview
func.func @subview_of_subview(%arg0: memref<128x64xf32>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.subview %arg0[8, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map1>
  %1 = memref.load %arg0[%c0, %c0] : memref<128x64xf32>
  %2 = arith.addf %1, %1 : f32
  %3 = memref.subview %0[2, 0] [2, 64] [1, 1] : memref<8x64xf32, #map1> to memref<2x64xf32, #map2>
// CHECK: memref.subview %arg0[10, 0] [2, 64] [1, 1]
  memref.store %2, %0[%c0, %c1] : memref<8x64xf32, #map1>
  memref.store %2, %3[%c0, %c1] : memref<2x64xf32, #map2>
  return
}

// CHECK-LABEL: func.func @view_of_view
func.func @view_of_view(%arg0: memref<8192xi8>) {
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.view %arg0[%c128][] : memref<8192xi8> to memref<8000xi8>
  %1 = memref.view %0[%c128][] : memref<8000xi8> to memref<8x64xf32>
// CHECK: memref.view %arg0[%c256][]
  %2 = memref.load %1[%c0, %c0] : memref<8x64xf32>
  %3 = arith.addf %2, %2 : f32
  memref.store %3, %1[%c0, %c1] : memref<8x64xf32>
  return
}

// CHECK-LABEL: func.func @subview_of_view_contiguous_row_major
func.func @subview_of_view_contiguous_row_major(%arg0: memref<8192xi8>) {
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.view %arg0[%c128][] : memref<8192xi8> to memref<128x64xf32>
  %1 = memref.subview %0[8, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map1> // 128 + 8*64*4 = 2176
// CHECK: memref.view %arg0[%c2176][] : memref<8192xi8> to memref<8x64xf32>
  %2 = memref.load %0[%c0, %c0] : memref<128x64xf32>
  %3 = arith.addf %2, %2 : f32
  memref.store %3, %1[%c0, %c1] : memref<8x64xf32, #map1>
  return
}

// CHECK-LABEL: func.func @subview_of_view_not_contiguous_row_major
func.func @subview_of_view_not_contiguous_row_major(%arg0: memref<8192xi8>) {  // last dim multiple of 64 example
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.view %arg0[%c128][] : memref<8192xi8> to memref<128x64xf32>
  %1 = memref.subview %0[0, 0] [128, 32] [1, 1] : memref<128x64xf32> to memref<128x32xf32, #map3> // do nothing
// CHECK: memref.subview
  %2 = memref.load %0[%c0, %c0] : memref<128x64xf32>
  %3 = arith.addf %2, %2 : f32
  memref.store %3, %1[%c0, %c1] : memref<128x32xf32, #map3>
  return
}

// CHECK-LABEL: func.func @interleave
func.func @interleave(%arg0: memref<8192xi8>) {  // two access interleave access (128-elemet) example 
  %c128 = arith.constant 128 : index
  %c640 = arith.constant 640 : index  // 128 + 128*4
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.view %arg0[%c128][] : memref<8192xi8> to memref<64x192xf32>   // 192 = (128*4+128*2)/4 // do nothing
// CHECK: memref.view %arg0
  %1 = memref.view %arg0[%c640][] : memref<8192xi8> to memref<64x384xf16>   // 640 = 128 + 128*4; 384 = (128*4+128*2)/2 // do nothing
// CHECK: memref.view %arg0
  %2 = memref.subview %0[0, 0] [64, 128] [1, 1] : memref<64x192xf32> to memref<64x128xf32, #map4> // do nothing
// CHECK: memref.subview
  %3 = memref.subview %1[0, 0] [64, 128] [1, 1] : memref<64x384xf16> to memref<64x128xf16, #map5> // do nothing
// CHECK: memref.subview
  %4 = memref.load %2[%c0, %c0] : memref<64x128xf32, #map4>  
  %5 = arith.addf %4, %4 : f32
  memref.store %5, %2[%c0, %c1] : memref<64x128xf32, #map4> 
  %6 = memref.load %3[%c0, %c0] : memref<64x128xf16, #map5>  
  %7 = arith.addf %6, %6 : f16
  memref.store %7, %3[%c0, %c1] : memref<64x128xf16, #map5> 
  return
}

// CHECK-LABEL: func.func @many_contiguous
func.func @many_contiguous(%arg0: memref<8192xi8>) {
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.view %arg0[%c128][] : memref<8192xi8> to memref<8000xi8>
  %1 = memref.view %0[%c128][] : memref<8000xi8> to memref<128x64xf32>  // 256 = 128 + 128
  %2 = memref.subview %1[8, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map1>   // 256 + 8*64*4 = 2304
// CHECK: memref.view %arg0[%c2304][] : memref<8192xi8> to memref<8x64xf32>   
  %3 = memref.subview %2[2, 0] [2, 64] [1, 1] : memref<8x64xf32, #map1> to memref<2x64xf32, #map2> // 2304 + 2*64*4 = 2816
// CHECK: memref.view %arg0[%c2816][] : memref<8192xi8> to memref<2x64xf32>
  %4 = memref.load %2[%c0, %c0] : memref<8x64xf32, #map1>
  %5 = arith.addf %4, %4 : f32
  memref.store %5, %3[%c0, %c1] : memref<2x64xf32, #map2>
  return
}

// CHECK-LABEL: func @subview_of_subview_non_one_stride
func.func @subview_of_subview_non_one_stride(%input: memref<4x1024xf32>) -> memref<1x128xf32, strided<[2048, 2], offset: 3584>> {
  %0 = memref.subview %input[2, 256] [2, 256] [1, 2] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 2], offset: 2304>>
  %1 = memref.subview %0[1, 128] [1, 128] [2, 1] : memref<2x256xf32, strided<[1024, 2], offset: 2304>> to memref<1x128xf32, strided<[2048, 2], offset: 3584>>
// CHECK:  memref.subview %arg0[3, 512] [1, 128] [2, 2]
  return %1 : memref<1x128xf32, strided<[2048, 2], offset: 3584>>
}

// CHECK-LABEL: func.func @subview_of_subview_non_one_stride_dynamic_offset
func.func @subview_of_subview_non_one_stride_dynamic_offset(%input: memref<4x1024xf32>, %offset_0 : index, %offset_1 : index) -> memref<1x128xf32, strided<[2048, 2], offset: ?>> {
  %0 = memref.subview %input[%offset_0, 256] [2, 256] [1, 2] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 2], offset: ?>>
  %1 = memref.subview %0[%offset_1, 128] [1, 128] [2, 1] : memref<2x256xf32, strided<[1024, 2], offset: ?>> to memref<1x128xf32, strided<[2048, 2], offset: ?>>
// CHECK: %0 = affine.apply
// CHECK: memref.subview %arg0[%0, 512] [1, 128] [2, 2]
  return %1 : memref<1x128xf32, strided<[2048, 2], offset: ?>>
}