from typing import Optional, Sequence, Union, List
import torch

from torch_frontend import torch_mlir
from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects.mhlo import register_mhlo_dialect

_CUSTOM_OPS_IN_TORCH = [
    "aten._softmax",
    "aten._log_softmax",
    "aten.native_layer_norm",
    "aten.layer_norm",
    "aten.gelu",
    "aten.softmax.int",
    "aten.argmax",
    "aten.max.dim",
    "aten.one_hot",
    "aten.topk",
]

def compile(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    output_type: str,
    backend_legal_ops: Optional[Sequence[str]] = None,
    verbose: bool = False,
):
    if output_type not in ["raw", "torch", "mhlo"]:
        raise NotImplemented("unsupported output type {}".format(output_type))
    if backend_legal_ops is None:
        backend_legal_ops = _CUSTOM_OPS_IN_TORCH

    module = torch_mlir.compile(
        model,
        example_inputs,
        output_type=torch_mlir.OutputType.RAW,
        use_tracing=False,
        verbose=False,
    )
    if output_type == "raw":
        return module

    with module.context:
        option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + "}"
        PassManager.parse(f"builtin.module(torchscript-to-torch-pipeline{option_string})").run(module.operation)
    if output_type == "torch":
        return module

    with module.context:
        PassManager.parse("builtin.module(torch-to-mhlo-pipeline)").run(module.operation)
    return module


def convert_to_mhlo_via_torch_mlir(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    backend_legal_ops: Optional[Sequence[str]] = None,
    use_tracing: bool = False,
    verbose: bool = False,
):
    if backend_legal_ops is None:
        backend_legal_ops = _CUSTOM_OPS_IN_TORCH
    # torch_mlir.BACKEND_LEGAL_OPS[torch_mlir.OutputType.TORCH] = backend_legal_ops
    module = torch_mlir.compile(
        model,
        example_inputs,
        output_type=torch_mlir.OutputType.RAW,
        use_tracing=use_tracing,
        verbose=False,
    )

    with module.context:
        option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + "}"
        PassManager.parse(f"builtin.module(torchscript-to-torch-pipeline{option_string})").run(module.operation)

    with module.context:
        PassManager.parse("builtin.module(torch-to-mhlo-pipeline)").run(module.operation)

    return module

