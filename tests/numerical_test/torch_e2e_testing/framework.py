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
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Any, Callable, List, NamedTuple, Union, Dict

import torch

TorchScriptValue = Union[int, float, List['TorchScriptValue'],
                         Dict['TorchScriptValue',
                              'TorchScriptValue'], torch.Tensor]


class TraceItem(NamedTuple):
    # The externally visible symbol name that is called.
    # For example `"forward"` or `"submodule.forward"`.
    symbol: str
    # The inputs to the call.
    inputs: List[TorchScriptValue]
    # The output from the call.
    # In Python, there is only one output from a function. It might be a tuple
    # in case of "multiple results".
    # Sometimes this field is treated as golden outputs from a test.
    # Sometimes this field is treated as ignored, such as the input trace
    # provided to `TestConfig.run`.
    output: TorchScriptValue


# A trace of invocations to the program.
# This is an ordered sequence of external invocations to a program's
# public boundary.
Trace = List[TraceItem]

# Clone all the tensor values.


def clone_torch_script_value(v: TorchScriptValue):
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, tuple):
        return tuple(clone_torch_script_value(field) for field in v)
    if isinstance(v, list):
        return [clone_torch_script_value(item) for item in v]
    if isinstance(v, dict):
        return {
            clone_torch_script_value(key): clone_torch_script_value(val)
            for key, val in v.items()
        }
    if isinstance(v, float) or isinstance(v, int) or isinstance(v, str):
        return v
    assert False, "unhandled cloning of TorchScriptValue value type"


class _Tracer:
    """Wrapper around a `torch.nn.Module` that records calls into it.

    The inputs and outputs of each call are recorded in a Trace. Recursive
    property accesses are also traced.
    """

    def __init__(self, wrapped, property_base_path: List[str], trace: Trace):
        self.__wrapped__ = wrapped
        self.__trace__ = trace
        self.__property_base_path__ = property_base_path

    def __call__(self, *args, **kwargs):
        # Clone the inputs to capture the original tensors values. This is
        # needed because inplace mutation might happen to the input tensors.
        inputs = [clone_torch_script_value(arg) for arg in args]
        output = self.__wrapped__(*args, **kwargs)
        self.__trace__.append(
            TraceItem(symbol=".".join(self.__property_base_path__),
                      inputs=inputs,
                      output=output))
        return output

    def __getattr__(self, name):
        return _Tracer(getattr(self.__wrapped__, name),
                       self.__property_base_path__ + [name], self.__trace__)


class TestUtils:
    """Utilities for executing a test.

    Test cases are provided an instance of this class to make test cases
    more succinct.

    For reproducibility, this class also resets the random seed.
    TODO: Figure out how to seed reset properly scoped to just a test case
    (such as when running tests in parallel)
    """

    def __init__(self):
        torch.manual_seed(0)

    # TODO: Add zeros/ones/etc. as convenient.
    def rand(self, *sizes, low=0.0, high=1.0):
        return torch.empty(sizes).uniform_(low, high).cuda()

    def randint(self, *sizes, low=0, high=10, dtype=torch.int64):
        return torch.randint(low, high, sizes, dtype=dtype).cuda()


class Test(NamedTuple):
    """A description of a test as produced by the test frontend.
    """
    # Stable name for error reporting.
    #
    # This name's stability is also useful for backend, which want to
    # generate their own lower-level test suites based on this framework.
    #
    # It is expected that those backends will need additional
    # metadata to describe their test configurations, so having a unique
    # key to keep that information associated is important.
    unique_name: str
    # A callable which produces the module under test.
    # This is a callable to allow lazily creating the module.
    program_factory: Callable[[], torch.nn.Module]
    # A callable which provides external stimuli to the module.
    # The first parameter is a torch.nn.Module (or a `_Tracer` wrapping that
    # module, actually).
    # The secon parameter is a `TestUtils` instance for convenience.
    program_invoker: Callable[[Any, TestUtils], None]


def generate_golden_trace(test: Test) -> Trace:
    """Generate a trace with the original program.

    If the original program is deterministic, then this the produced trace is
    suitable as a golden trace to compare against.
    """
    trace = []
    tracer = _Tracer(test.program_factory(), [], trace)
    test.program_invoker(tracer, TestUtils())
    return trace
