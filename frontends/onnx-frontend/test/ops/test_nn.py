import pytest
from test.base import TestBase
from test.ops.utils import build_onnx


class TestOpsNN(TestBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        self.setup_base(tmpdir_factory, "test/ops/data/nn")

    def test_conv(self):
        input_shape_dtype = [
            ["X", (1, 1, 5, 5), "float32"],
            ["W", (1, 1, 3, 3), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (1, 1, 5, 5), "float32"],
        ]
        proto = build_onnx("Conv", input_shape_dtype, output_shape_dtype, kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        self.run(model_filename="conv.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_conv_transpose(self):
        input_shape_dtype = [
            ["X", (1, 3, 3, 3), "float32"],
            ["W", (3, 2, 3, 3), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (1, 2, 5, 5), "float32"],
        ]
        proto = build_onnx(
            "ConvTranspose", input_shape_dtype, output_shape_dtype
        )
        self.run(model_filename="conv_transpose.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_conv_transpose_group(self):
        input_shape_dtype = [
            ["X", (1, 6, 8, 14), "float32"],
            ["W", (6, 2, 4, 4), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (1, 4, 16, 28), "float32"],
        ]
        proto = build_onnx(
            "ConvTranspose", input_shape_dtype, output_shape_dtype,
            group=2, kernel_shape=[4, 4], output_padding=[0, 0], pads=[1, 1, 1, 1], strides=[2, 2]
        )
        self.run(model_filename="conv_transpose_group.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_batch_normalization(self):
        input_shape_dtype = [
            ["input_0", (2, 3, 2, 2), "float32"],
        ]
        self.run(model_filename="batch_normalization.onnx", input_shape_dtype=input_shape_dtype)

    def test_average_pool(self):
        input_shape_dtype = [
            ["X", (1, 2, 11, 4), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (1, 2, 5, 3), "float32"],
        ]
        proto = build_onnx(
            "AveragePool", input_shape_dtype, output_shape_dtype,
            kernel_shape=[2, 2], strides=[2, 2],pads=[0, 1, 0, 1]
        )
        self.run(model_filename="average_pool.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_instance_normalization(self):
        input_shape_dtype = [
            ["input", (1, 5, 3), "float32"],
            ["scale", (5,), "float32"],
            ["B", (5,), "float32"],
        ]
        output_shape_dtype = [
            ["output", (1, 5, 3), "float32"],
        ]
        proto = build_onnx(
            "InstanceNormalization", input_shape_dtype, output_shape_dtype,
            epsilon=9.999999974752427e-7
        )
        self.run(model_filename="instance_norm.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)
