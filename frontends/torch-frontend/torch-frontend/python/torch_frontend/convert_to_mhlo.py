from typing import Optional, Sequence, Union, List
import torch

from torch_frontend import torch_mlir
from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects.mhlo import register_mhlo_dialect

_CUSTOM_OPS_IN_TORCH = [
    "torch.aten._softmax",
    "torch.aten._log_softmax",
    "torch.aten.native_layer_norm",
    "torch.aten.gelu",
    "torch.aten.softmax.int",
    "torch.aten.argmax",
    "torch.aten.max.dim",
    "torch.aten.one_hot",
]

def convert_to_mhlo_via_torch_mlir(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    backend_legal_ops: Optional[Sequence[str]] = None,
    use_tracing: bool = False,
    verbose: bool = False,
):
    if backend_legal_ops is None:
        backend_legal_ops = _CUSTOM_OPS_IN_TORCH
    torch_mlir.BACKEND_LEGAL_OPS[torch_mlir.OutputType.TORCH] = backend_legal_ops
    module = torch_mlir.compile(
        model,
        example_inputs,
        output_type=torch_mlir.OutputType.TORCH,
        use_tracing=use_tracing,
        verbose=verbose,
    )
    with module.context:
        pm = PassManager.parse("builtin.module(torch-to-mhlo-pipeline)")
        pm.run(module)
    return module

