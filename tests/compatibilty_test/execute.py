# Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import brt
from brt.utils import brt_dtype_to_torch_dtype

import torch
import numpy as np
import os
import re

from reporting import TestResult


class BRTBackend:

    def __init__(self, device, brt_file_path):
        _stream = None
        self.device = None
        if device == "CPU":
            self.session = brt.Session(device=device.upper(), )
            self.device = "cpu"
            _stream = None
        else:
            raise NotImplementedError(
                f"Compatible test for {device} not implement")

        self.session.load(brt_file_path)
        self.req = self.session.new_request_context(_stream)

    def _check(self, result, golden, atol=1e-06):
        return torch.allclose(result, golden, atol=atol)

    def _generate_torch_outputs(self):
        outputs = []
        for offset in self.session.get_output_arg_offsets():
            outputs.append(
                torch.empty(self.session.get_static_shape(offset),
                            dtype=brt_dtype_to_torch_dtype(
                                self.session.get_data_type(offset)),
                            device=self.device))
        return outputs

    def compare(self, inputs, goldens):
        outputs = self._generate_torch_outputs()
        assert len(self.session.get_input_arg_offsets()) == len(inputs)
        assert len(self.session.get_output_arg_offsets()) == len(outputs)
        assert len(outputs) == len(goldens)
        for offset, arg in zip(self.session.get_input_arg_offsets(), inputs):
            assert list(self.session.get_static_shape(offset)) == list(
                arg.shape)
            self.req.bind_arg(offset, arg.data_ptr())
        for offset, ret in zip(self.session.get_output_arg_offsets(), outputs):
            assert list(self.session.get_static_shape(offset)) == list(
                ret.shape)
            self.req.bind_arg(offset, ret.data_ptr())
        self.req.finish_io_binding()
        self.req.run()
        self.req.sync()
        return all(self._check(o, g) for o, g in zip(outputs, goldens))


def run_and_check_mlir(name, testdir, target):
    inps_pattern = re.compile(r'inputs\.\d\.npz$')
    outs_pattern = re.compile(r'outputs\.\d\.npz$')
    inp_files = [
        file for file in os.listdir(testdir) if inps_pattern.search(file)
        and os.path.isfile(os.path.join(testdir, file))
    ]
    out_files = [
        file for file in os.listdir(testdir) if outs_pattern.search(file)
        and os.path.isfile(os.path.join(testdir, file))
    ]
    assert len(inp_files) == len(out_files)
    brt_file_path = os.path.join(testdir, name + ".rt.mlir")

    _device = None
    if target == "cpu":
        _device = "CPU"

    brt_backend = BRTBackend(device=_device, brt_file_path=brt_file_path)

    cmp_res = []
    for idx in range(len(inp_files)):
        input_file = os.path.join(testdir, f"inputs.{idx}.npz")
        target_file = os.path.join(testdir, f"outputs.{idx}.npz")
        inp = np.load(input_file, allow_pickle=True)
        inp = [torch.from_numpy(inp[f]).contiguous().to(_device.lower()) for f in inp.files]
        tgt = np.load(target_file, allow_pickle=True)
        tgt = [torch.from_numpy(tgt[f]).contiguous().to(_device.lower()) for f in tgt.files]
        if brt_backend.compare(inp, tgt):
            cmp_res.append(TestResult(name + str(idx), numerical_error=None))
        else:
            cmp_res.append(
                TestResult(
                    name + str(idx),
                    numerical_error=
                    f"input is {input_file}, output not match {target_file}"))

    return cmp_res
