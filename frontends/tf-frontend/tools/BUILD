load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "tf-frontend",
    srcs = ["tf_frontend_main.cc"],
    deps = [
        "//byteir/Dialect/Ace:ace_dialect",
        "//tf_mlir_ext/pipelines:tfext_pass_pipelines",
        "//utils:graphdef_opt",
        "//utils:misc",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:AllTranslations",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_xla_mlir_translate",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "@org_tensorflow//tensorflow/c:c_api_no_xla",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)


tf_cc_binary(
    name = "tf-ext-opt",
    srcs = ["tf_ext_opt_main.cc"],
    deps = [
        "//byteir/Dialect/Ace:ace_dialect",
        "//tf_mlir_ext/pipelines:tfext_pass_pipelines",
        "//tf_mlir_ext/transforms:tfext_passes",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@local_xla//xla/mlir_hlo:all_passes",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_test_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:tf_xla_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:lift_tflite_flex_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "@org_tensorflow//tensorflow/core:lib",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:ShapeDialect",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

