Testcase(
    contents=[Content(stages=(Input, E2E), content=r"""
func.func @main() -> tensor<1x97xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<[1, 97]> : tensor<2xi64>
    %3 = "mhlo.rng"(%0, %1, %2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
    return %3 : tensor<1x97xf32>
}
    """)],
    pipelines=[
        InputPipeline(r"""
// CHECK-LABEL: func.func @main
"""),
        HostOptPipeline(r"""
// CHECK-LABEL: func.func @Unknown
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