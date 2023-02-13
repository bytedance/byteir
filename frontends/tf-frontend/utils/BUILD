load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "graphdef_opt",
    srcs = [
        "graphdef_opt.cc",
    ],
    hdrs = [
        "graphdef_opt.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core/grappler/clusters:single_machine",
        "@org_tensorflow//tensorflow/core/grappler/optimizers:meta_optimizer",
        "@org_tensorflow//tensorflow/tools/graph_transforms:transform_graph_lib",
    ],
)

cc_library(
    name = "misc",
    srcs = [
        "misc.cc",
    ],
    hdrs = [
        "misc.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/platform:status",
    ]
)
