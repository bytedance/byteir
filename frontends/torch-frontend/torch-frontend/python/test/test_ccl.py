import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import os

import torch_frontend
from torch_frontend import compile_exported_program

def setup(world_size, rank, port="4991", addr="localhost"):
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.barrier()
    dist.destroy_process_group()

def run(
    rank,
    world_size,
    module,
    inputs,
    constraints=None,
):
    setup(world_size, rank)
    prog = torch.export.export(module, tuple(inputs), constraints=constraints)
    if rank == 0:
        module = compile_exported_program(prog, "stablehlo")
        print(module.operation.get_asm())
    cleanup()

# ==============================================================================

class AllReduceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return funcol.all_reduce(x, "sum", [0, 1, 2, 3])

def test_all_reduce():
    module = AllReduceModule()
    world_size = 4
    inputs = [torch.tensor([1, 2, 3, 4], dtype=torch.float32)]
    mp.spawn(run, args=(world_size, module, inputs), nprocs=world_size, join=True)

# ==============================================================================

class AllGatherModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return funcol.all_gather_tensor(x, 0, [0, 1, 2, 3])

def test_all_gather():
    module = AllGatherModule()
    world_size = 4
    inputs = [torch.tensor([1, 2, 3, 4], dtype=torch.float32)]
    mp.spawn(run, args=(world_size, module, inputs), nprocs=world_size, join=True)

# ==============================================================================

class ReduceScatterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return funcol.reduce_scatter_tensor(x, "sum", 0, [0, 1, 2, 3])

def test_reduce_scatter():
    module = ReduceScatterModule()
    world_size = 4
    inputs = [torch.tensor([1, 2, 3, 4], dtype=torch.float32)]
    mp.spawn(run, args=(world_size, module, inputs), nprocs=world_size, join=True)

# ==============================================================================
# TODO: add test for send/recv
