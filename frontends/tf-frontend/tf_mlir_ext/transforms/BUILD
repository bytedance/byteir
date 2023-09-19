load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")


package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)


gentbl_cc_library(
    name = "tfext_pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorFlowExtension",
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

td_library(
    name = "tfext_transform_patterns_td_files",
    srcs = [
        "fuse_tf_ops.td",
        "rewrite_to_custom_call.td",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    includes = ["../../external/tensorflow"], # patch tensorflow_ops_td_files's `includes`
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tfext_rewrite_to_custom_call_pattern_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "rewrite_to_custom_call.inc",
        )
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "rewrite_to_custom_call.td",
    
    deps = [
        ":tfext_transform_patterns_td_files",
    ],
)

gentbl_cc_library(
    name = "tfext_fuse_tf_ops_pattern_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "fuse_tf_ops.inc",
        )
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "fuse_tf_ops.td",
    
    deps = [
        ":tfext_transform_patterns_td_files",
    ],
)


cc_library(
    name = "tfext_passes",
    srcs = [
        "constant_folding.cc",
        "fuse_tf_ops.cc",
        "mhlo_legalize_tf_ext.cc",
        "process_dynamic_stitch_as_static.cc",
        "reshape_movedown_string.cc",
        "remove_control_flow.cc",
        "rewrite_to_if.cc",
        "rewrite_func_attr_to_byteir.cc",
        "rewrite_to_custom_call.cc",
        "tf_fallback_to_custom_call.cc",
    ],
    hdrs = [
      "constant_folding.h",
      "fuse_tf_ops.h",
      "passes.h",
      "mhlo_legalize_tf_ext.h",
      "process_dynamic_stitch_as_static.h",
      "reshape_movedown_string.h",
      "remove_control_flow.h",
      "rewrite_to_if.h",
      "rewrite_func_attr_to_byteir.h",
      "rewrite_to_custom_call.h",
      "tf_fallback_to_custom_call.h",
    ],
    textual_hdrs = [
       "passes_detail.h",
       "passes.h.inc",
       "fuse_tf_ops.inc",
       "rewrite_to_custom_call.inc",
    ],
    deps = [
      "//utils:attributes",
      "//tf_mlir_ext/utils:tfext_utils",
      "//byteir/Dialect/Ace:ace_dialect",
      ":tfext_pass_inc_gen",
      ":tfext_fuse_tf_ops_pattern_gen",
      ":tfext_rewrite_to_custom_call_pattern_gen",
      "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
      "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:shape_inference_utils",
      "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow",
      "@org_tensorflow//tensorflow/compiler/mlir/lite:validators",
      "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo",
      "@llvm-project//llvm:Support",
      "@llvm-project//mlir:Dialect",
      "@llvm-project//mlir:IR",
      "@llvm-project//mlir:Support",
    ]
)
