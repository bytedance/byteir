package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tfext_utils",
    srcs = [
        "customcall.cc",
        "dce.cc",
        "utils.cc",
    ],
    hdrs = [
      "customcall.h",
      "dce.h",
      "utils.h",
    ],
    deps = [
      "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow",
      "@llvm-project//llvm:Support",
      "@llvm-project//mlir:Dialect",
      "@llvm-project//mlir:IR",
      "@llvm-project//mlir:Support",
    ]
)