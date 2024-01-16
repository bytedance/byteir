import os
import mhlo_tools
import numpy as np
import argparse


def get_config(config: str):
    if config == "":
        return {}
    if config == "fuse_tf_ops":
        # FIXME: remove black_list when tf.Conv3D with dilates could run on AMD cpu
        return {"black_list": ["dilated_conv3d"]}
    if config == "dilated_conv2d":
        # FIXME: remove black_list when support mhlo.convolution dynamic shape
        return {"black_list": ["dilated_conv1"]}
    if config == "fallback":
        return {
            "inputs_dict": {
                "test_string_to_ace_custom_call": [
                    np.array(["aa\t", "bb"], dtype=np.str_)
                ],
                "test_dynamic_partition": [
                    np.random.randn(4, 4).astype(np.float32),
                    np.random.randint(0, 2, size=(4,)).astype(np.int32),
                ],
            }
        }
    if config == "process_dynamic_stitch_as_static":
        return {
            "inputs_dict": {
                "dynamic_stitch_with_multiple_data": [
                    np.random.randn(512, 368).astype(np.float16),
                    np.random.randn(512, 368).astype(np.float16),
                    np.random.randn(512, 368).astype(np.float16),
                    np.random.randint(0, 2, size=(512,)).astype(np.int32),
                ],
                "dynamic_stitch_with_zero_fill": [
                    np.random.randn(4, 5).astype(np.float32),
                    np.random.randn(4, 5).astype(np.float32),
                    np.random.randint(0, 2, size=(4, 10)).astype(np.bool_),
                    np.random.randint(0, 2, size=(4,)).astype(np.int32),
                ],
            }
        }
    if config == "rewrite_to_custom_call":
        return {
            "inputs_dict": {
                "dynamic_mask_stitch": [
                    np.random.rand(4, 4).astype(np.float32),
                    np.random.randint(0, 2, size=(4,)).astype(np.int32),
                ]
            },
            "decimal_dict": {
                "log_softmax_case0": 6,
                "erf_case0": 6,
                "gelu_erf_case0": 6,
                "gelu_erf_case1": 6,
                "gelu_tanh_case0": 6,
                "gelu_tanh_case1": 2,
                "gelu_tanh_case2": 6,
                "layer_norm": 6,
                "layer_norm_negative_axis": 6,
                "layer_norm_without_beta": 6,
                "layer_norm_swap_add": 6,
                "layer_norm_swap_mul": 6,
                "layer_norm_swap_squarediff": 6,
                "layer_norm_V2": 6,
                "layer_norm_V3_disable_minimize_broadcast": 6,
                "layer_norm_V4": 2,
                "layer_norm_V4_swap_squarediff": 2,
                "layer_norm_with_cast": 2,
                "layer_norm_with_cast_v2": 2,
                "layer_norm_with_cast_disable_minimize_broadcast": 2,
                "l2_norm_V1": 6,
                "l2_norm_V1_swap_mul": 6,
                "l2_norm_V2": 3,
                "l2_norm_V2_swap_mul": 3,
                "l2_norm_V3": 6,
                "onehot_case0": 6,
            },
            "black_list": [
                "dynamic_partition",
                "dynamic_stitch",
            ],
        }
    raise NotImplementedError("No such config name {}".format(config))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before_pass_file")
    parser.add_argument("after_pass_file")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()
    mhlo_tools.testing.pass_numerical_test(
        args.before_pass_file, args.after_pass_file, **get_config(args.config)
    )
