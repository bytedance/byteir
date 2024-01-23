// RUN: byteir-translate -emit-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: byteir-translate -emit-cpp -declare-var-at-top-cpp %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @emitc_call_unary() {
  %c0 = arith.constant 0 : i32
  %0 = emitc.call_opaque "max" (%c0) : (i32) -> i32
  %1 = emitc.call_opaque "unknowAdd" (%0, %c0) : (i32,i32) -> i32
  return
}
// CPP-DEFAULT: void emitc_call_unary() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = max([[V0]]);
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = unknowAdd([[V1]], [[V0]]);

// CPP-DECLTOP: void emitc_call_unary() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0]] = 0;
// CPP-DECLTOP-NEXT: [[V1]] = max([[V0]]);
// CPP-DECLTOP-NEXT: [[V2]] = unknowAdd([[V1]], [[V0]]);


func.func @emitc_call() {
  %0 = emitc.call_opaque "func_a" () : () -> i32
  %1 = emitc.call_opaque "func_b" () : () -> i32
  return
}
// CPP-DEFAULT: void emitc_call() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = func_a();
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = func_b();

// CPP-DECLTOP: void emitc_call() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0:]] = func_a();
// CPP-DECLTOP-NEXT: [[V1:]] = func_b();


func.func @emitc_call_two_results() {
  %0 = arith.constant 0 : index
  %1:2 = emitc.call_opaque "two_results" () : () -> (i32, i32)
  return
}
// CPP-DEFAULT: void emitc_call_two_results() {
// CPP-DEFAULT-NEXT: size_t [[V1:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: std::tie([[V2]], [[V3]]) = two_results();

// CPP-DECLTOP: void emitc_call_two_results() {
// CPP-DECLTOP-NEXT: size_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V1]] = 0;
// CPP-DECLTOP-NEXT: std::tie([[V2]], [[V3]]) = two_results();
