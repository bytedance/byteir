import torch
from torch.testing import FileCheck

import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.testing._internal.common_utils import run_tests

from utils import with_comms, DistributedTestBase, skip_unless_torch_version_bigger_than

import torch_frontend
from torch_frontend import compile_dynamo_model


class AllReduceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return funcol.all_reduce(x, "sum", [0, 1, 2, 3])


class AllGatherModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return funcol.all_gather_tensor(x, 0, [0, 1, 2, 3])


class ReduceScatterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return funcol.reduce_scatter_tensor(x, "sum", 0, [0, 1, 2, 3])


class BroadcastModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return funcol.broadcast(x, 2, [0, 1, 2, 3])


class DistributedCollectiveTest(DistributedTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_reduce_scatter(self):
        module = ReduceScatterModule()
        inputs = [torch.tensor([1, 2, 3, 4], dtype=torch.float32)]
        prog = torch.export.export(module, tuple(inputs))
        if dist.get_rank() == 0:
            module = compile_dynamo_model(prog, "stablehlo")
            ir = module.operation.get_asm()
            FileCheck().check("@main").check("ccl.reduce_scatter").check(
                "axis = 0"
            ).check('reduction = "sum"').check("replica_groups = [[0, 1, 2, 3]]").check(
                "-> tensor<1xf32>"
            ).run(
                ir
            )

    @with_comms
    def test_all_reduce(self):
        module = AllReduceModule()
        inputs = [torch.tensor([1, 2, 3, 4], dtype=torch.float32)]
        prog = torch.export.export(module, tuple(inputs))
        if dist.get_rank() == 0:
            module = compile_dynamo_model(prog, "stablehlo")
            ir = module.operation.get_asm()
            FileCheck().check("@main").check("ccl.all_reduce").check(
                'reduction = "sum"'
            ).check("replica_groups = [[0, 1, 2, 3]]").check("-> tensor<4xf32>").run(ir)

    @with_comms
    def test_all_gather(self):
        module = AllGatherModule()
        inputs = [torch.tensor([1, 2, 3, 4], dtype=torch.float32)]
        prog = torch.export.export(module, tuple(inputs))
        if dist.get_rank() == 0:
            module = compile_dynamo_model(prog, "stablehlo")
            ir = module.operation.get_asm()
            FileCheck().check("@main").check("ccl.all_gather").check("axis = 0").check(
                "replica_groups = [[0, 1, 2, 3]]"
            ).check("-> tensor<16xf32>").run(ir)

    @with_comms
    @skip_unless_torch_version_bigger_than(torch_version="2.2")
    def test_broadcast(self):
        module = BroadcastModule()
        inputs = [torch.tensor([1, 2, 3, 4], dtype=torch.float32)]
        prog = torch.export.export(module, tuple(inputs))
        if dist.get_rank() == 0:
            module = compile_dynamo_model(prog, "stablehlo")
            ir = module.operation.get_asm()
            FileCheck().check("@main").check("ccl.broadcast").check(
                "replica_groups = [[2, 0, 1, 3]]"
            ).check("-> tensor<4xf32>").run(ir)

    # TODO: add test for send/recv


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, world_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.fc2 = torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim)

    def forward(self, x):
        return funcol.all_reduce(
            self.fc2(self.fc1(x)), "sum", list(range(self.world_size))
        )


class DistributedCollectiveE2ETest(DistributedTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_mlp_e2e(self):
        module = MLP(hidden_dim=4, world_size=self.world_size)
        x = torch.rand(3, 4)
        prog = torch.export.export(module, (x,))

        module = compile_dynamo_model(prog, "stablehlo")

        if dist.get_rank() == 0:
            ir = module.operation.get_asm()
            print(ir)


if __name__ == "__main__":
    run_tests()
