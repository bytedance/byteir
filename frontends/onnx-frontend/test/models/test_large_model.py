import os
import pytest
from test.base import TestBase
from test.env import LARGE_MODEL_PATH

LARGE_MODEL_PATH = os.environ["LARGE_MODEL_PATH"]

class TestLargeModel(TestBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        self.data_dir = LARGE_MODEL_PATH
        tmp_dir = tmpdir_factory.mktemp(self.data_dir.replace("/", "_"))
        self.tmp_dir = str(tmp_dir)

    def test_mnist(self):
        input_shape_dtype = [
            ["image", (1, 1, 28, 28), "float32"],
        ]
        # bs=1 in mnist.onnx
        self.run(model_filename="mnist.onnx", input_shape_dtype=input_shape_dtype)

    def test_resnet(self):
        input_shape_dtype = [
            ["data", (8, 3, 224, 224), "float32"],
        ]
        # bs=8 in resnet50-v1-7.onnx
        self.run(model_filename="resnet50-v1-7.onnx", input_shape_dtype=input_shape_dtype)

    def test_transformer(self):
        input_shape_dtype = [
            ["fbank", (2, 128, 80), "float32"],
        ]
        self.run(model_filename="transformer_encoder_-1x128x80.onnx", input_shape_dtype=input_shape_dtype, batch_size=2)
