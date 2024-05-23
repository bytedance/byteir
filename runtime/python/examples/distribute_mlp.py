import brt
import byteir

import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed._functional_collectives as funcol

from torch_frontend import compile_dynamo_model

stablehlo_ir_name = "stablehlo_ccl.mlir"
ccl_ir_name = "ccl.mlir"


class MLP(nn.Module):
    def __init__(self, hidden_dim, world_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.fc2 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            # Initialize weights with fixed values
            self.fc1.weight.fill_(0.01)
            self.fc1.bias.fill_(0.02)
            self.fc2.weight.fill_(0.03)
            self.fc2.bias.fill_(0.04)

    def forward(self, x):
        return funcol.all_reduce(
            self.fc2(self.fc1(x)), "sum", list(range(self.world_size))
        )


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def brt_infer(nranks, host, port, rank, data):
    local_rank = rank
    d_session = brt.DistributedSession(rank, nranks, host, port)
    config = [ccl_ir_name] * nranks
    ir = d_session.load_config(config)
    d_session.load(ccl_ir_name)

    req = d_session.new_request_context(None)

    assert len(d_session.get_input_arg_offsets()) == 1
    for offset in d_session.get_input_arg_offsets():
        req.bind_arg(offset, data.data_ptr())

    outputs = []

    for offset in d_session.get_output_arg_offsets():
        outputs.append(
            torch.empty(
                d_session.get_static_shape(offset), dtype=torch.float32, device="cuda"
            )
        )
        req.bind_arg(offset, outputs[-1].data_ptr())

    assert len(d_session.get_output_arg_offsets()) == 1

    req.finish_io_binding()
    req.run()
    req.sync()

    return outputs[0]

def infer(rank, world_size, hidden_dim, brt_host, brt_port):
    setup(rank, world_size)

    model = MLP(hidden_dim, world_size).to(rank)

    # Initialize data based on rank
    if rank == 0:
        data = torch.ones(10, hidden_dim)
    elif rank == 1:
        data = torch.ones(10, hidden_dim) * 2
    elif rank == 2:
        data = torch.ones(10, hidden_dim) * 3
    elif rank == 3:
        data = torch.ones(10, hidden_dim) * 4
    else:
        data = torch.zeros(10, hidden_dim)

    data = data.to(rank)

    with torch.no_grad():
        outputs = model(data)

    outputs_brt = brt_infer(world_size, brt_host, brt_port, rank, data)

    assert torch.allclose(outputs_brt, outputs)

    cleanup()


def main():
    hidden_dim = 16
    world_size = 4
    host = "localhost"

    module = MLP(hidden_dim, world_size)
    x = torch.rand(10, hidden_dim)
    prog = torch.export.export(module, (x,), constraints=None)

    module = compile_dynamo_model(prog, "stablehlo")

    ir = module.operation.get_asm()

    with open(stablehlo_ir_name, "w") as f:
        f.write(ir)

    byteir.compile(stablehlo_ir_name, ccl_ir_name, entry_func="main")

    port = brt.get_free_port()
    brt.create_server(world_size, port)

    mp.spawn(
        infer, args=(world_size, hidden_dim, host, port), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    main()
