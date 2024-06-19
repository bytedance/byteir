Testcase(
    contents=[Content(stages=(Input, E2E), content=r"""
func.func @main(%arg0 : tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>  {
  %0 = mhlo.convert %arg0 : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>
  return %0 : tensor<1x224x224x3xf16>
}
    """)],
    pipelines=[
        InputPipeline(r"""
// CHECK-LABEL: func.func @main
"""),
        HostOptPipeline(r"""
// CHECK-LABEL: func.func @Unknown
//   CHECK: %[[COLLAPSE0:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]] : memref<1x224x224x3xf32> into memref<150528xf32>
//   CHECK: %[[COLLAPSE1:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]] : memref<1x224x224x3xf16> into memref<150528xf16>
//   CHECK-NEXT: scf.for %arg2 = %c0 to %c150528 step %c1 {
//   CHECK-NEXT:   %[[LOAD:.*]] = memref.load %[[COLLAPSE0]]
//   CHECK-NEXT:   %[[TRUNCF:.*]] = arith.truncf %[[LOAD]] : f32 to f16
//   CHECK-NEXT:   memref.store %[[TRUNCF]], %[[COLLAPSE1]]
//   CHECK-NEXT: }
"""),
        ToLLVMPipeline(r"""
// CHECK: llvm.func
"""),
        ToLLVMIRPipeline(r"""
// CHECK-LABEL: define void @_mlir_ciface_Unknown
"""),
        ByreHostPipeline(r"""
// CHECK-LABEL: func.func @main
"""),
        TotalPipeline(r"""
// CHECK-LABEL: define void @_mlir_ciface_Unknown
"""),
        ByreOutPipeline(r"""
// CHECK-LABEL: func.func @main
"""),
    ]
)