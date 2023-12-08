import numpy as np
import pytest
import onnx
from test.base import TestBase
from test.ops.utils import build_onnx


class TestOpsRNN(TestBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        self.setup_base(tmpdir_factory, "test/ops/data/rnn")

    def test_lstm(self):
        input_shape_dtype = [
            ["X", (10, 16, 512), "float32"],
            ["W", (2, 1024, 512), "float32"],
            ["R", (2, 1024, 256), "float32"],
            ["B", (2, 2048), "float32"],
            ["", None, None],
            ["initial_h", (2, 16, 256), "float32"],
            ["initial_c", (2, 16, 256), "float32"],
            ["", None, None],
        ]
        output_shape_dtype = [
            ["Y", (10, 2, 16, 256), "float32"],
        ]
        proto = build_onnx("LSTM", input_shape_dtype, output_shape_dtype,
                           direction="bidirectional", hidden_size=256)
        input_shape_dtype = [input_shape_dtype[0], input_shape_dtype[1],
                             input_shape_dtype[2], input_shape_dtype[3],
                             input_shape_dtype[5], input_shape_dtype[6],]
        self.run(model_filename="lstm.onnx", model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype, decimal=3)