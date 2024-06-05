module {
  func.func @"__torch_mlir_shape_fn.byteir.flash_attn_fwd"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.float, %arg4: !torch.float, %arg5: !torch.bool, %arg6: !torch.bool) -> !torch.tuple<list<int>, list<int>, list<int>, list<int>, list<int>, list<int>, list<int>, list<int>> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %0 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %1 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %2 = torch.aten.__getitem__.t %arg0, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %3 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %4 = torch.prim.ListConstruct %0, %2, %1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.prim.ListConstruct %0, %2, %1, %3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %6 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %7 = torch.prim.TupleConstruct %arg0, %arg0, %arg1, %arg2, %arg0, %4, %5, %6 : !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>, list<int>, list<int>, list<int>, list<int>, list<int>, list<int>>
    return %7 : !torch.tuple<list<int>, list<int>, list<int>, list<int>, list<int>, list<int>, list<int>, list<int>>
  }
  func.func @"__torch_mlir_dtype_fn.byteir.flash_attn_fwd"(%arg0: !torch.tuple<int, int>, %arg1: !torch.tuple<int, int>, %arg2: !torch.tuple<int, int>, %arg3: !torch.float, %arg4: !torch.float, %arg5: !torch.bool, %arg6: !torch.bool) -> !torch.tuple<int, int, int, int, int, int, int, int> {
    %int4 = torch.constant.int 4
    %int6 = torch.constant.int 6
    %0:2 = torch.prim.TupleUnpack %arg0 : !torch.tuple<int, int> -> !torch.int, !torch.int
    %1 = torch.prim.TupleConstruct %0#1, %0#1, %0#1, %0#1, %0#1, %int6, %0#1, %int4 : !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tuple<int, int, int, int, int, int, int, int>
    return %1 : !torch.tuple<int, int, int, int, int, int, int, int>
  }
  func.func @"__torch_mlir_has_value_semantics_fn.byteir.flash_attn_fwd"() -> !torch.none {
    %none = torch.constant.none
    return %none : !torch.none
  }
  func.func @"__torch_mlir_shape_fn.byteir.flash_attn_bwd"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.float, %arg7: !torch.float, %arg8: !torch.bool, %arg9: !torch.list<int>) -> !torch.tuple<list<int>, list<int>, list<int>, list<int>, list<int>> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int127 = torch.constant.int 127
    %int128 = torch.constant.int 128
    %int3 = torch.constant.int 3
    %int31 = torch.constant.int 31
    %int32 = torch.constant.int 32
    %0 = torch.aten.__getitem__.t %arg1, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %1 = torch.aten.__getitem__.t %arg1, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %2 = torch.aten.__getitem__.t %arg1, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %3 = torch.aten.add.int %1, %int127 : !torch.int, !torch.int -> !torch.int
    %4 = torch.aten.floordiv.int %3, %int128 : !torch.int, !torch.int -> !torch.int
    %5 = torch.aten.mul.int %4, %int128 : !torch.int, !torch.int -> !torch.int
    %6 = torch.aten.__getitem__.t %arg1, %int3 : !torch.list<int>, !torch.int -> !torch.int
    %7 = torch.aten.add.int %6, %int31 : !torch.int, !torch.int -> !torch.int
    %8 = torch.aten.floordiv.int %7, %int32 : !torch.int, !torch.int -> !torch.int
    %9 = torch.aten.mul.int %8, %int32 : !torch.int, !torch.int -> !torch.int
    %10 = torch.prim.ListConstruct %0, %2, %5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %11 = torch.prim.ListConstruct %0, %2, %5, %9 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %12 = torch.prim.TupleConstruct %arg1, %arg2, %arg3, %10, %11 : !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>, list<int>, list<int>, list<int>>
    return %12 : !torch.tuple<list<int>, list<int>, list<int>, list<int>, list<int>>
  }
  func.func @"__torch_mlir_dtype_fn.byteir.flash_attn_bwd"(%arg0: !torch.tuple<int, int>, %arg1: !torch.tuple<int, int>, %arg2: !torch.tuple<int, int>, %arg3: !torch.tuple<int, int>, %arg4: !torch.tuple<int, int>, %arg5: !torch.tuple<int, int>, %arg6: !torch.float, %arg7: !torch.float, %arg8: !torch.bool, %arg9: !torch.tuple<int, int>) -> !torch.tuple<int, int, int, int, int> {
    %int6 = torch.constant.int 6
    %0:2 = torch.prim.TupleUnpack %arg1 : !torch.tuple<int, int> -> !torch.int, !torch.int
    %1:2 = torch.prim.TupleUnpack %arg2 : !torch.tuple<int, int> -> !torch.int, !torch.int
    %2:2 = torch.prim.TupleUnpack %arg3 : !torch.tuple<int, int> -> !torch.int, !torch.int
    %3 = torch.prim.TupleConstruct %0#1, %1#1, %2#1, %int6, %int6 : !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tuple<int, int, int, int, int>
    return %3 : !torch.tuple<int, int, int, int, int>
  }
  func.func @"__torch_mlir_has_value_semantics_fn.byteir.flash_attn_bwd"() -> !torch.none {
    %none = torch.constant.none
    return %none : !torch.none
  }
  func.func @"__torch_mlir_shape_fn.byteir.flash_attn_kvcache"(%arg0: !torch.list<int>, %arg1: !torch.list<int>, %arg2: !torch.list<int>, %arg3: !torch.list<int>, %arg4: !torch.list<int>, %arg5: !torch.list<int>, %arg6: !torch.float, %arg7: !torch.bool) -> !torch.tuple<list<int>, list<int>> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %0 = torch.aten.__getitem__.t %arg0, %int0 : !torch.list<int>, !torch.int -> !torch.int
    %1 = torch.aten.__getitem__.t %arg0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    %2 = torch.aten.__getitem__.t %arg0, %int2 : !torch.list<int>, !torch.int -> !torch.int
    %3 = torch.prim.ListConstruct %0, %2, %1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.prim.TupleConstruct %arg0, %3 : !torch.list<int>, !torch.list<int> -> !torch.tuple<list<int>, list<int>>
    return %4 : !torch.tuple<list<int>, list<int>>
  }
  func.func @"__torch_mlir_dtype_fn.byteir.flash_attn_kvcache"(%arg0: !torch.tuple<int, int>, %arg1: !torch.tuple<int, int>, %arg2: !torch.tuple<int, int>, %arg3: !torch.tuple<int, int>, %arg4: !torch.tuple<int, int>, %arg5: !torch.tuple<int, int>, %arg6: !torch.float, %arg7: !torch.bool) -> !torch.tuple<int, int> {
    %int6 = torch.constant.int 6
    %0:2 = torch.prim.TupleUnpack %arg0 : !torch.tuple<int, int> -> !torch.int, !torch.int
    %1 = torch.prim.TupleConstruct %0#1, %int6 : !torch.int, !torch.int -> !torch.tuple<int, int>
    return %1 : !torch.tuple<int, int>
  }
  func.func @"__torch_mlir_has_value_semantics_fn.byteir.flash_attn_kvcache"() -> !torch.none {
    %none = torch.constant.none
    return %none : !torch.none
  }
}
