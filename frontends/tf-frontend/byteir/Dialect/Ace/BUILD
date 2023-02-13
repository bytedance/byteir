load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")


package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

td_library(
    name = "ace_ir_td_files",
    srcs = [
        "AceBase.td",
        "AceOps.td",
    ],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "ace_dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "AceOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "AceOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls", "-dialect=ace"],
            "AceOpsDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs", "-dialect=ace"],
            "AceOpsDialect.cpp.inc",
        ),
        (
            ["-gen-typedef-decls", "-typedefs-dialect=ace"],
            "AceOpsTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs", "-typedefs-dialect=ace"],
            "AceOpsTypes.cpp.inc"
        ),
        (
            ["-gen-attrdef-decls"],
            "AceOpsAttributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "AceOpsAttributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "AceOps.td",
    deps = [
        ":ace_ir_td_files",
    ],
)

cc_library(
    name = "ace_dialect",
    srcs = [
        "AceDialect.h",
        "AceDialect.cpp",
    ],
    textual_hdrs = [
        "AceOps.h.inc",
        "AceOpsDialect.h.inc",
    ],
    deps = [
        ":ace_dialect_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)