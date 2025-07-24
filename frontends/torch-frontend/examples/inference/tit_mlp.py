import os

import torch
from torch import nn
import torch_frontend
import byteir

from brt_backend import BRTBackend

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 512,dtype=torch.float32)
        self.linear2 = nn.Linear(512, 256,dtype=torch.float32)
        self.linear3 = nn.Linear(256, 128,dtype=torch.float32)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        return x

workspace = "./workspace"
os.makedirs(workspace, exist_ok=True)
with torch.no_grad():
    torch.backends.cuda.matmul.allow_tf32=False
    model = MLP().cuda().eval()
    inputs = [torch.randn(128, 256, dtype=torch.float32).cuda()]
    traced_model = torch.jit.trace(model, inputs)

    stablehlo_file = workspace + "/model.stablehlo.mlir"
    byre_file = workspace + "/model.byre.mlir"
    module = torch_frontend.compile(traced_model, inputs, "stablehlo")
    with open(stablehlo_file, "w") as f:
        f.write(module.operation.get_asm())

    byteir.compile(stablehlo_file, byre_file, entry_func="forward", target="cuda_with_triton")

    backend = BRTBackend("cuda", byre_file)
    byteir_outputs = backend.execute(inputs)

    torch_outputs = model(*inputs)
    torch_jit_outputs = traced_model(*inputs)
    if len(byteir_outputs) == 1:
        byteir_outputs = byteir_outputs[0]

    torch.testing.assert_close(torch_outputs, torch_jit_outputs, rtol=1e-3, atol=1e-3)
    try:
        torch.testing.assert_close(torch_outputs, byteir_outputs, rtol=1e-3, atol=1e-3)
    except AssertionError as e:
        diff=torch.abs(torch_outputs-byteir_outputs)
        print("diff:",diff)
        raise e
    print("byteir tit backend success")