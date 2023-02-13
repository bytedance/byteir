import onnx
import os.path as osp
import pytest
import subprocess

from test.ops.utils import build_onnx
from test.env import ONNX_FRONTEND_PATH

class TestModelsBatchSize:

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        dir = "test/models/data/batch_size"
        self.tmp_dir = str(tmpdir_factory.mktemp(dir.replace("/", "_")))

    def check(self, model_filename, model_onnx_pb, bs, ref):
        onnx_path = osp.join(self.tmp_dir, model_filename)
        onnx.save(model_onnx_pb, onnx_path)

        cmd_opts = [ONNX_FRONTEND_PATH]
        cmd_opts.append(f"-batch-size={bs}")
        cmd_opts.append(onnx_path)

        p = subprocess.run(
            cmd_opts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        out, err = p.stdout, p.stderr
        assert not err, str(err)
        assert ref in out

    def test_dynamic_batch_size(self):
        input_shape_dtype = [
            ["X", (None, 2), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (None, 2), "float32"],
        ]
        proto = build_onnx("Abs", input_shape_dtype, output_shape_dtype)
        self.check("dynamic_batch_size.onnx", proto, 3, "tensor<3x2xf32>")
