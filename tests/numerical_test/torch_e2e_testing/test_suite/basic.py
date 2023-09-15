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

import torch
from ..registry import register_test_case
from ..framework import TestUtils


class ElementwiseAddModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


@register_test_case(module_factory=lambda: ElementwiseAddModule())
def ElementwiseAddModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand())


class MatmulF16Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)


@register_test_case(module_factory=lambda: MatmulF16Module())
def MatmulF16Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(256, 512).to(torch.float16),
                   tu.rand(512, 1024).to(torch.float16))


class MatmulF32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)


@register_test_case(module_factory=lambda: MatmulF32Module())
def MatmulF32Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 6), tu.rand(6, 10))


class BatchMatmulF32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.bmm(a, b)


@register_test_case(module_factory=lambda: BatchMatmulF32Module())
def BatchMatmulF32Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 6), tu.rand(2, 6, 10))


class BatchMatmulAddF32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        return c + torch.bmm(a, b)


@register_test_case(module_factory=lambda: BatchMatmulAddF32Module())
def BatchMatmulAddF32Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 6), tu.rand(2, 6, 10), tu.rand(2, 5, 10))
