load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")


package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)


gentbl_cc_library(
    name = "tfext_pass_pipelines_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorFlowExtensionPipelines",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "tfext_pass_pipelines",
    srcs = [
        "customized_tf_to_mhlo.cc",
    ],
    hdrs = [
        "customized_tf_to_mhlo.h",
        "passes.h",
        "passes_detail.h",
    ],
    textual_hdrs = [
        "passes.h.inc",
    ],
    deps = [
        "//tf_mlir_ext/transforms:tfext_passes",
        ":tfext_pass_pipelines_inc_gen",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/mlir_hlo:mlir_hlo",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:tf_xla_passes",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ]
)
