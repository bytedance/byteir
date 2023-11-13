import numpy as np
import pytest
from test.base import TestBase
from test.ops.utils import build_onnx, build_reduce_sum_axis_one
import onnx


class TestOpsMath(TestBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        self.setup_base(tmpdir_factory, "test/ops/data/math")

    def test_erf(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 2), "float32"],
        ]
        proto = build_onnx("Erf", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="erf.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_clip(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="clip.onnx", input_shape_dtype=input_shape_dtype)

    def test_abs(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 2), "float32"],
        ]
        proto = build_onnx("Abs", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="abs.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_cast(self):
        input_shape_dtype = [
            ["input", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["output", (3, 2), "float64"],
        ]
        proto = build_onnx("Cast", input_shape_dtype, output_shape_dtype, to=getattr(onnx.TensorProto, "DOUBLE"))
        self.run(model_filename="cast.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_equal(self):
        input_shape_dtype = [
            ["X", (3, 2), "int64"],
            ["Y", (3, 2), "int64"],
        ]
        output_shape_dtype = [
            ["Z", (3, 2), "bool"],
        ]
        proto = build_onnx("Equal", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="equal.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_sqrt(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 2), "float32"],
        ]
        proto = build_onnx("Sqrt", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="sqrt.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_pow(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
            ["Y", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Z", (3, 2), "float32"],
        ]
        proto = build_onnx("Pow", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="pow.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)
        
    def test_prelu(self):
        input_shape_dtype = [
            ["x", (3, 4, 5), "float32"],
            ["slope", (5, ), "float32"],
        ]
        output_shape_dtype = [
            ["y", (3, 4, 5), "float32"],
        ]
        proto = build_onnx("PRelu", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="prelu.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_mat_mul(self):
        input_shape_dtype = [
            ["X", (3, 2, 5, 4), "float32"],
            ["Y", (3, 2, 4, 3), "float32"]
        ]
        
        output_shape_dtype = [
            ["Z", (3, 2, 5, 3), "float32"],
        ]
        proto = build_onnx("MatMul", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="mat_mul.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_reduce_max(self):
        input_shape_dtype = [
            ["X", (3, 2, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 1, 2), "float32"],
        ]
        proto = build_onnx("ReduceMax", input_shape_dtype, output_shape_dtype, axes=[1], keepdims=1)
        self.run(model_filename="reduce_max.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_reduce_min(self):
        input_shape_dtype = [
            ["X", (3, 2, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 1, 2), "float32"],
        ]
        proto = build_onnx("ReduceMin", input_shape_dtype, output_shape_dtype, axes=[1], keepdims=1)
        self.run(model_filename="reduce_min.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_reduce_mean(self):
        input_shape_dtype = [
            ["X", (3, 2, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 1, 2), "float32"],
        ]
        proto = build_onnx("ReduceMean", input_shape_dtype, output_shape_dtype, axes=[1], keepdims=1)
        self.run(model_filename="reduce_mean.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_reduce_sum(self):
        input_shape_dtype = [
            ["X", (3, 2, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 1, 2), "float32"],
        ]
        proto = build_reduce_sum_axis_one(input_shape_dtype, output_shape_dtype)
        np.random.seed(0)
        input_data = {
            "X": np.random.rand(3, 2, 2).astype(np.float32),
        }
        self.run(model_filename="reduce_sum.onnx", model_onnx_pb=proto, input_data=input_data)

    def test_reduce_l2(self):
        input_shape_dtype = [
            ["X", (3, 2, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 1, 2), "float32"],
        ]
        proto = build_onnx("ReduceL2", input_shape_dtype, output_shape_dtype, axes=[1], keepdims=1)
        self.run(model_filename="reduce_l2.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_global_average_pool(self):
        input_shape_dtype = [
            ["X", (3, 2, 2, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 2, 1, 1), "float32"],
        ]
        proto = build_onnx("GlobalAveragePool", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="globalavg.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_softmax(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="softmax.onnx", input_shape_dtype=input_shape_dtype)

    def test_log(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 2), "float32"],
        ]
        proto = build_onnx("Log", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="log.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_tanh(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 2), "float32"],
        ]
        proto = build_onnx("Tanh", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="tanh.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_leaky_relu(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (3, 2), "float32"],
        ]
        proto = build_onnx("LeakyRelu", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="leaky_relu.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_max(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
            ["Y", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Z", (3, 2), "float32"],
        ]
        proto = build_onnx("Max", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="max.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_gelu(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="gelu.onnx", input_shape_dtype=input_shape_dtype)
