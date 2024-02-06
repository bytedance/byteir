load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")


package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

td_library(
    name = "ace_ir_td_files",
    srcs = [
        "include/byteir/Dialect/Ace/AceBase.td",
        "include/byteir/Dialect/Ace/AceOps.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "ace_dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/byteir/Dialect/Ace/AceOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/byteir/Dialect/Ace/AceOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls", "-dialect=ace"],
            "include/byteir/Dialect/Ace/AceOpsDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs", "-dialect=ace"],
            "include/byteir/Dialect/Ace/AceOpsDialect.cpp.inc",
        ),
        (
            ["-gen-typedef-decls", "-typedefs-dialect=ace"],
            "include/byteir/Dialect/Ace/AceOpsTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs", "-typedefs-dialect=ace"],
            "include/byteir/Dialect/Ace/AceOpsTypes.cpp.inc"
        ),
        (
            ["-gen-attrdef-decls"],
            "include/byteir/Dialect/Ace/AceOpsAttributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/byteir/Dialect/Ace/AceOpsAttributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/byteir/Dialect/Ace/AceOps.td",
    deps = [
        ":ace_ir_td_files",
    ],
)

cc_library(
    name = "ace_dialect",
    srcs = [
        "include/byteir/Dialect/Ace/AceDialect.h",
        "lib/Dialect/Ace/IR/AceDialect.cpp",
    ],
    textual_hdrs = [
        "include/byteir/Dialect/Ace/AceDialect.h",
        "include/byteir/Dialect/Ace/AceOps.h.inc",
        "include/byteir/Dialect/Ace/AceOpsDialect.h.inc",
    ],
    includes = ["include"],
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
