from typing import Optional, Sequence, Union, List, Tuple
import torch
import sys

from torch_frontend import torch_mlir
from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects.mhlo import register_mhlo_dialect

_CUSTOM_OPS_IN_TORCH = [
    "aten._softmax",
    "aten.softmax.int",
    "aten._log_softmax",
    "aten.log_softmax.int",
    "aten.native_layer_norm",
    "aten.layer_norm",
    "aten.gelu",
    "aten.argmax",
    "aten.max.dim",
    "aten.one_hot",
    "aten.topk",
    "byteir.flash_attn_fwd",
    "byteir.flash_attn_bwd",
    "pytorch.gemv_upmem",
]

# def pytorch〇gemv_upmem

def byteir〇flash_attn_fwd〡shape(q: List[int], k: List[int], v: List[int], dropout_p: float, softmax_scale: float, casual: bool, return_softmax: bool) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int]]:
    batch_size = q[0]
    seqlen_q = q[1]
    num_heads = q[2]
    seqlen_k = k[1]
    softmax_lse = [batch_size, num_heads, seqlen_q]
    softmax_return = [batch_size, num_heads, seqlen_q, seqlen_k]
    rng_shape = [2]
    return q, q, k, v, q, softmax_lse, softmax_return, rng_shape

def byteir〇flash_attn_fwd〡dtype(q_rank_dtype: Tuple[int, int], k_rank_dtype: Tuple[int, int], v_rank_dtype: Tuple[int, int], dropout_p: float, softmax_scale: float, casual: bool, return_softmax: bool) -> Tuple[int, int, int, int, int, int, int, int]:
    q_rank, q_dtype = q_rank_dtype
    return q_dtype, q_dtype, q_dtype, q_dtype, q_dtype, torch.float32, q_dtype, torch.int64

def byteir〇flash_attn_fwd〡has_value_semantics() -> None:
    return

def byteir〇flash_attn_bwd〡shape(dout: List[int], q: List[int], k: List[int], v: List[int], out: List[int], softmax_lse: List[int], dropout_p: float, softmax_scale: float, casual: bool, rng: List[int]) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    batch_size = q[0]
    seqlen_q = q[1]
    num_heads = q[2]
    seqlen_q_rounded = ((seqlen_q+127)//128)*128
    head_size = q[3]
    head_size_rounded = ((head_size+31)//32)*32
    d_softmax = [batch_size, num_heads, seqlen_q_rounded]
    dq_accum = [batch_size, num_heads, seqlen_q_rounded, head_size_rounded]
    return q, k, v, d_softmax, dq_accum

def byteir〇flash_attn_bwd〡dtype(dout_rank_dtype: Tuple[int, int], q_rank_dtype: Tuple[int, int], k_rank_dtype: Tuple[int, int], v_rank_dtype: Tuple[int, int], out_rank_dtype: Tuple[int, int], softmax_lse_rank_dtype: Tuple[int, int], dropout_p: float, softmax_scale: float, casual: bool, rng_rank_dtype: Tuple[int, int]) -> Tuple[int, int, int, int, int]:
    dq_rank, dq_dtype = q_rank_dtype
    dk_rank, dk_dtype = k_rank_dtype
    dv_rank, dv_dtype = v_rank_dtype
    return dq_dtype, dk_dtype, dv_dtype, torch.float32, torch.float32

def byteir〇flash_attn_bwd〡has_value_semantics() -> None:
    return

extra_library = [
    byteir〇flash_attn_fwd〡shape, byteir〇flash_attn_fwd〡dtype, byteir〇flash_attn_fwd〡has_value_semantics,
    byteir〇flash_attn_bwd〡shape, byteir〇flash_attn_bwd〡dtype, byteir〇flash_attn_bwd〡has_value_semantics]

def compile(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    output_type: str,
    backend_legal_ops: Optional[Sequence[str]] = None,
    verbose: bool = False,
    debug: bool = False,
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

    if debug:
        print("// IR Dump After RAW")
        print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))
        print()
        sys.stdout.flush()

    if debug:
        module.context.enable_multithreading(False)

    extra_library_file_name = torch_mlir._canon_extra_library(extra_library)
    if verbose:
        cmdline_option_string = "backend-legal-ops=" + ",".join(backend_legal_ops) + " extra-library=" + extra_library_file_name
        print(f'[RUN] ./build/bin/torch-frontend-opt --torchscript-to-torch-pipeline="{cmdline_option_string}"')
    with module.context:
        option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + " extra-library=" + extra_library_file_name + "}"
        pm = PassManager.parse(f"builtin.module(torchscript-to-torch-pipeline{option_string})")
        if debug:
            pm.enable_ir_printing()
        pm.run(module.operation)
    if output_type == "torch":
        return module

    if verbose:
        print('[RUN] ./build/bin/torch-frontend-opt --torch-to-mhlo-pipeline')
    with module.context:
        pm = PassManager.parse("builtin.module(torch-to-mhlo-pipeline)")
        if debug:
           pm.enable_ir_printing() 
        pm.run(module.operation)
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

