import numpy as np
import pytest
import onnx
from test.base import TestBase
from test.ops.utils import build_onnx


class TestOpsQuantize(TestBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        self.setup_base(tmpdir_factory, "test/ops/data/quantize")

    def test_quantize_dequantize(self):
        input_shape_dtype = [
            ["data", (16, 3, 224, 224), "float32"],
        ]
        self.run(model_filename="quantize_dequantize.onnx", input_shape_dtype=input_shape_dtype)
