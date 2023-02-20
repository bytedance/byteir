import json
import os.path as osp
import pytest
import re
import subprocess

from typing import Dict, List, Optional
import numpy.typing as npt

from mhlo_tools.ir_executor import Interpreter
import numpy as np
import onnx
from onnxsim import simplify
import onnxruntime as ort
import pickle

from test.env import ONNX_FRONTEND_PATH

CUSTOM_CALL_OPS = [
    "arg_max",
    "arg_min",
    "layer_norm",
    "erf",
    "gelu",
    "instance_norm",
    "l2_norm",
    "quantize",
    "dequantize",
    "softmax",
]
class TestBase:
    def setup_base(self, tmpdir_factory, dir):
        self.data_dir = dir
        self.tmp_dir = str(tmpdir_factory.mktemp(dir.replace("/", "_")))

    def convert_onnx_to_onnx_sim(self, onnx_path, onnx_sim_path, input_data):
        input_shapes = {
            k: list(v.shape) for k, v in input_data.items()
        }

        model_original = onnx.load(onnx_path)
        model_sim, check = simplify(
            model_original,
            # check_n=1,
            perform_optimization=True,
            input_shapes=input_shapes)
        # assert check, "Failed to simplify onnx model"

        onnx.save(model_sim, onnx_sim_path)

    def convert_onnx_sim_to_mhlo_ir(self, onnx_path, mhlo_ir_path):
        if not osp.exists(onnx_path):
            raise ValueError("onnx_path {} not exists".format(onnx_path))

        cmd_opts = [ONNX_FRONTEND_PATH]
        cmd_opts.append(onnx_path)
        cmd_opts.append(f"-custom-call-ops={','.join(CUSTOM_CALL_OPS)}")
        cmd_opts.append("-invokeOnnxVersionConverter")
        cmd_opts.append("-mlir-print-op-generic")
        # cmd_opts.append(f"-o={mhlo_ir_path}")
        p = subprocess.run(
            cmd_opts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        out, err = p.stdout, p.stderr
        assert not err, str(err)
        with open(mhlo_ir_path, "w") as f:
            f.write(out)

    def onnx_mhlo_test_helper(self, onnx_path, mhlo_ir_path, input_data: Dict[str, npt.NDArray]):
        # TODO: handle a model with multiple final outputs
        onnx_model = onnx.load(onnx_path)
        onnx_init_names = set([init.name for init in onnx_model.graph.initializer])
        input_names = [inp.name for inp in onnx_model.graph.input if inp.name not in onnx_init_names]

        # onnx
        ort_sess = ort.InferenceSession(onnx_path)
        onnx_outputs: List[npt.NDArray] = ort_sess.run(None, input_data)

        # load mhlo mlir
        interp = Interpreter.load_from_file(mhlo_ir_path)
        # generate all golden references
        mhlo_data: List[npt.NDArray] = [input_data[name] for name in input_names]
        mhlo_outputs: List[npt.NDArray] = interp.call_function("main", mhlo_data)

        np.testing.assert_almost_equal(onnx_outputs, mhlo_outputs, decimal=4)

    def run(self,
            model_filename: Optional[str] = None,
            model_onnx_pb = None,
            input_data: Optional[Dict[str, npt.NDArray]] = None,
            input_filename: Optional[str] = None,
            input_shape_dtype: Optional[List[List]] = None):
        assert self.data_dir is not None, "self.data_dir not initialized in derived class"
        assert osp.isdir(self.data_dir), "self.data_dir (" + \
            self.data_dir + ") is not a directory"
        assert self.tmp_dir is not None, "self.tmp_dir not initialized in derived class"
        assert osp.isdir(self.tmp_dir), "self.tmp_dir (" + \
            self.tmp_dir + ") is not a directory"

        if input_filename is not None:
            # set inputs
            content = pickle.load(
                open(osp.join(self.data_dir, input_filename), 'rb'))
            input_data = {k: np.array(v) for k, v in content.items()}

        if input_shape_dtype is not None:
            np.random.seed(0)
            input_data = {
                name: np.random.randn(*shape).astype(dtype)
                for name, shape, dtype in input_shape_dtype
            }

        try:
            if model_filename.endswith(".onnx"):
                # from onnx pb file
                base_filename = model_filename[:-len(".onnx")]

                onnx_path = osp.join(self.data_dir, model_filename)  # to read from data_dir
                onnx_sim_path = osp.join(self.tmp_dir, base_filename + "_sim.onnx")
                mhlo_ir_path = osp.join(self.tmp_dir, base_filename + ".mhlo.mlir")

                if model_onnx_pb is None:
                    self.convert_onnx_to_onnx_sim(onnx_path, onnx_sim_path, input_data)
                else:
                    # skip onnx-simplifier
                    onnx_path = osp.join(self.tmp_dir, model_filename)
                    onnx_sim_path = osp.join(self.tmp_dir, base_filename + "_sim.onnx")
                    # save onnx pb to tmp_dir
                    onnx.save(model_onnx_pb, onnx_path)
                    onnx.save(model_onnx_pb, onnx_sim_path)

                self.convert_onnx_sim_to_mhlo_ir(onnx_sim_path, mhlo_ir_path)
                self.onnx_mhlo_test_helper(onnx_path, mhlo_ir_path, input_data)
            else:
                raise ValueError(
                "Model file {} has an unkown extension name".format(model_filename))
        except Exception as e:
            print("Error occurs in UT, data_dir {}, tmp_dir {}".format(self.data_dir, self.tmp_dir))
            raise e
