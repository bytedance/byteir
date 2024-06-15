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


class ReductionPaddingModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a,):
        return torch.ops.aten.mean(a)


@register_test_case(module_factory=lambda: ReductionPaddingModule())
def ReductionPaddingModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1023))

class ReductionOneSizeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a,):
        return torch.ops.aten.mean(a,dim=(1))


@register_test_case(module_factory=lambda: ReductionOneSizeModule())
def ReductionOneSizeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024,1))

class  Large1DReductionModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a,):
        return torch.ops.aten.mean(a)

@register_test_case(module_factory=lambda: Large1DReductionModule())
def Large1DReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10000))

class ParallelReductionModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a,):
        return torch.ops.aten.sum(a, 1)

@register_test_case(module_factory=lambda: ParallelReductionModule())
def ParallelReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10000, 1000))

class ReductionParallelModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a,):
        return torch.ops.aten.sum(a, 0)

@register_test_case(module_factory=lambda: ReductionParallelModule())
def ReductionParallelModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(60, 10000))

class RngUniformModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a,):
        a = torch.rand(32,10240000)*0.001
        mean = torch.ops.aten.mean(a,dim=1)
        std = torch.ops.aten.std(a,dim=1)
        return torch.cat([mean,std])


@register_test_case(module_factory=lambda: RngUniformModule())
def RngUniformModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(32,10240000))

class RngNormalModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a,):
        a = torch.randn(32,10240000)*0.001
        mean = torch.ops.aten.mean(a,dim=1)
        std = torch.ops.aten.std(a,dim=1)
        return torch.cat([mean,std])


@register_test_case(module_factory=lambda: RngNormalModule())
def RngNormalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(32,10240000))
