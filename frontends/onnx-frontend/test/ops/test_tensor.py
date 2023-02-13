import numpy as np
import pytest
import onnx
from test.base import TestBase
from test.ops.utils import build_onnx


class TestOpsTensor(TestBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        self.setup_base(tmpdir_factory, "test/ops/data/tensor")

    def test_concat(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
            ["input_1", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="concat.onnx",
                 input_shape_dtype=input_shape_dtype)

    def test_shape(self):
        input_shape_dtype = [
            ["X", (3, 2, 4, 5), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (4,), "int64"],
        ]
        proto = build_onnx("Shape", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="shape.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_gather(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
            ["Y", (2, 2), "int64"],
        ]
        output_shape_dtype = [
            ["Z", (2, 2, 2), "float32"],
        ]
        proto = build_onnx("Gather", input_shape_dtype, output_shape_dtype, axis=0)

        np.random.seed(0)
        input_data = {
            "X": np.random.rand(3, 2).astype(np.float32),
            "Y": np.array([[0, 1], [1, 2]], dtype=np.int64),
        }
        self.run(model_filename="gather.onnx", model_onnx_pb=proto, input_data=input_data)

    def test_split(self):
        input_shape_dtype = [
            ["X", (12, 2), "float32"],
        ]
        output_shape_dtype = [
            ["output_0", (4, 2), "float32"],
            ["output_1", (4, 2), "float32"],
            ["output_2", (4, 2), "float32"],
        ]
        proto = build_onnx("Split", input_shape_dtype, output_shape_dtype, axis=0)
        self.run(model_filename="split.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_transpose(self):
        input_shape_dtype = [
            ["X", (5, 5, 1, 32), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (5, 1, 32, 5), "float32"],
        ]
        perm = [0, 2, 3, 1]
        proto = build_onnx("Transpose", input_shape_dtype, output_shape_dtype, perm=perm)
        self.run(model_filename="transpose.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_unsqueeze(self):
        input_shape_dtype = [
            ["X", (20, 10, 5), "float32"],
            ["axes", (2,), "int64"],
        ]
        output_shape_dtype = [
            ["Y", (20, 1, 10, 5, 1), "float32"],
        ]
        axes_tensor = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [2], np.array([1, 4]))
        proto = build_onnx("Unsqueeze", input_shape_dtype, output_shape_dtype, initializer=[axes_tensor])

        input_shape_dtype.pop()
        self.run(model_filename="unsqueeze.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_concat_dynamic_shape(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
            ["input_1", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="concat_dynamic_shape.onnx", input_shape_dtype=input_shape_dtype)

    def test_arg_max(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="arg_max.onnx", input_shape_dtype=input_shape_dtype)

    def test_arg_min(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="arg_min.onnx", input_shape_dtype=input_shape_dtype)

    def test_pad(self):
        input_shape_dtype = [
            ["X", (1, 3, 5, 5), "float32"],
            ["pads", (8,), "int64"],
            ["constant_value", tuple(), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (1, 3, 7, 7), "float32"],
        ]
        pads_tensor = onnx.helper.make_tensor(
            "pads", onnx.TensorProto.INT64, [8], np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        constant_value_tensor = onnx.helper.make_tensor(
            "constant_value", onnx.TensorProto.FLOAT, [], np.array([2.0]))
        proto = build_onnx(
            "Pad", input_shape_dtype, output_shape_dtype,
            initializer=[pads_tensor, constant_value_tensor], mode="constant"
        )
        input_shape_dtype.pop()
        input_shape_dtype.pop()
        self.run(model_filename="pad.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)
