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
import onnxruntime as ort
import pickle

from test.env import ONNX_FRONTEND_PATH

CUSTOM_CALL_OPS = [
    "arg_max",
    "arg_min",
    "layer_norm",
    "dequantize",
    "erf",
    "gelu",
    "l2_norm",
    "quantize",
    "resize",
    "softmax",
]
class TestBase:
    def setup_base(self, tmpdir_factory, dir):
        self.data_dir = dir
        self.tmp_dir = str(tmpdir_factory.mktemp(dir.replace("/", "_")))

    def convert_onnx_to_stablehlo_ir(self, onnx_path, stablehlo_ir_path, batch_size, onnx_frontend_option):
        if not osp.exists(onnx_path):
            raise ValueError("onnx_path {} not exists".format(onnx_path))

        cmd_opts = [ONNX_FRONTEND_PATH]
        cmd_opts.append(onnx_path)
        cmd_opts.append(f"-batch-size={batch_size}")
        cmd_opts.append(f"-custom-call-ops={','.join(CUSTOM_CALL_OPS)}")
        cmd_opts.append("-invokeOnnxVersionConverter")
        # mhlo-tools not accept generic format for now
        # cmd_opts.append("-mlir-print-op-generic")
        if onnx_frontend_option != "":
            cmd_opts.append(onnx_frontend_option)

        # cmd_opts.append(f"-o={stablehlo_ir_path}")
        p = subprocess.run(
            cmd_opts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        out, err = p.stdout, p.stderr
        assert not err, str(err)
        with open(stablehlo_ir_path, "w") as f:
            f.write(out)

    def onnx_stablehlo_test_helper(self, onnx_path, stablehlo_ir_path, input_data: Dict[str, npt.NDArray], decimal):
        # TODO: handle a model with multiple final outputs
        onnx_model = onnx.load(onnx_path)
        onnx_init_names = set([init.name for init in onnx_model.graph.initializer])
        input_names = [inp.name for inp in onnx_model.graph.input if inp.name not in onnx_init_names]

        # onnx
        ort_sess = ort.InferenceSession(onnx_path)
        onnx_outputs: List[npt.NDArray] = ort_sess.run(None, input_data)

        stablehlo_data: List[npt.NDArray] = [input_data[name] for name in input_names]
        # load stablehlo mlir
        with Interpreter.load_from_file(stablehlo_ir_path, True) as interp:
            # generate all golden references
            stablehlo_outputs: List[npt.NDArray] = interp.call_function("main", stablehlo_data)
            for onnx_output, stablehlo_output in zip(onnx_outputs, stablehlo_outputs):
                np.testing.assert_almost_equal(onnx_output, stablehlo_output, decimal=decimal)

    def run(self,
            model_filename: Optional[str] = None,
            model_onnx_pb = None,
            input_data: Optional[Dict[str, npt.NDArray]] = None,
            input_filename: Optional[str] = None,
            input_shape_dtype: Optional[List[List]] = None,
            batch_size: int = 1,
            decimal: int = 4,
            onnx_frontend_option: str = ""):
        assert self.data_dir is not None, "self.data_dir not initialized in derived class"
        assert self.tmp_dir is not None, "self.tmp_dir not initialized in derived class"
        assert osp.isdir(self.tmp_dir), "self.tmp_dir (" + \
            self.tmp_dir + ") is not a directory"

        if input_filename is not None:
            assert osp.isdir(self.data_dir), "self.data_dir (" + \
                self.data_dir + ") is not a directory"
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
                stablehlo_ir_path = osp.join(self.tmp_dir, base_filename + ".stablehlo.mlir")

                if model_onnx_pb is not None:
                    onnx_path = osp.join(self.tmp_dir, model_filename)
                    # save onnx pb to tmp_dir
                    onnx.save(model_onnx_pb, onnx_path)

                self.convert_onnx_to_stablehlo_ir(onnx_path, stablehlo_ir_path, batch_size, onnx_frontend_option)
                self.onnx_stablehlo_test_helper(onnx_path, stablehlo_ir_path, input_data, decimal)
            else:
                raise ValueError(
                "Model file {} has an unkown extension name".format(model_filename))
        except Exception as e:
            print("Error occurs in UT, data_dir {}, tmp_dir {}".format(self.data_dir, self.tmp_dir))
            raise e
