from typing import Optional, Sequence, Union, List, Tuple
from enum import Enum
import sys

import torch
import torch.export

import torch_mlir
from torch_mlir.extras.fx_importer import FxImporter
from torch_frontend import ir
from torch_frontend.passmanager import PassManager
from torch_frontend.dialects.builtin import ModuleOp
from torch_frontend._mlir_libs._stablehlo import serialize_portable_artifact

_CUSTOM_OPS_IN_TORCH = [
    "aten._softmax",
    "aten.softmax.int",
    "aten._log_softmax",
    "aten.log_softmax.int",
    "aten.native_layer_norm",
    "aten.layer_norm",
    "aten.group_norm",
    "aten.native_group_norm",
    "aten.gelu",
    "aten.argmax",
    "aten.max.dim",
    "aten.one_hot",
    "aten.topk",
    "byteir.flash_attn_fwd",
    "byteir.flash_attn_bwd",
]

class DebugType(Enum):
    NO_DEBUG = 0
    PRINT_AFTER_FALIURE = 1
    PRINT_AFTER_ONLY_CHANGE = 2

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
    seqlen_q_rounded = ((seqlen_q + 127) // 128) * 128
    head_size = q[3]
    head_size_rounded = ((head_size + 31) // 32) * 32
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
    byteir〇flash_attn_fwd〡shape,
    byteir〇flash_attn_fwd〡dtype,
    byteir〇flash_attn_fwd〡has_value_semantics,
    byteir〇flash_attn_bwd〡shape,
    byteir〇flash_attn_bwd〡dtype,
    byteir〇flash_attn_bwd〡has_value_semantics,
]


def compile(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    output_type: str,
    backend_legal_ops: Optional[Sequence[str]] = None,
    debug: DebugType = DebugType.NO_DEBUG,
) -> ModuleOp:
    """
    Args:
        output_type: str type
            `raw`
            `torch`
            `stablehlo`
            `stablehlo+0.16.2`(stablehlo version, could specify other version)
        debug: DebugType, one of
            `NO_DEBUG: no debug message`,
            `PRINT_AFTER_FALIURE: print after failure`,
            `PRINT_AFTER_ONLY_CHANGE: print after pass only on change`
    """
    if output_type not in ["raw", "torch"] and "stablehlo" not in output_type:
        raise NotImplementedError(f"unsupported output type {output_type}")
    assert isinstance(debug, DebugType), "unsupported debug type"
    if backend_legal_ops is None:
        backend_legal_ops = _CUSTOM_OPS_IN_TORCH

    ### compile to raw by torch_mlir.torchscript
    from torch_mlir import torchscript
    module = torchscript.compile(
        model,
        example_inputs,
        output_type=torchscript.OutputType.RAW,
        use_tracing=False,
        verbose=False,
    )
    module_str = module.operation.get_asm(enable_debug_info=True)

    context = ir.Context()
    module = ir.Module.parse(module_str, context)
    if output_type == "raw":
        return module

    if debug == DebugType.PRINT_AFTER_ONLY_CHANGE:
        print("// IR Dump After RAW")
        print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))
        print()
        sys.stdout.flush()

    if debug != DebugType.NO_DEBUG:
        module.context.enable_multithreading(False)

    extra_library_file_name = torchscript._canon_extra_library(extra_library)
    with module.context:
        option_string = (
            "{backend-legal-ops=" + ",".join(backend_legal_ops) + " extra-library=" + extra_library_file_name + "}"
        )
        pm = PassManager.parse(f"builtin.module(torchscript-to-torch-pipeline{option_string})")
        if debug == DebugType.PRINT_AFTER_FALIURE:
            pm.enable_ir_printing(
                print_before_pass=False,
                print_after_pass=True,
                print_after_only_on_change=False,
                print_after_only_on_failure=True,
                large_elements_limit=10,
            )
        if debug == DebugType.PRINT_AFTER_ONLY_CHANGE:
            pm.enable_ir_printing(
                print_before_pass=False,
                print_after_pass=True,
                print_after_only_on_change=True,
                print_after_only_on_failure=False,
                large_elements_limit=10,
            )
        pm.run(module.operation)
    if output_type == "torch":
        return module

    with module.context:
        pm = PassManager.parse("builtin.module(torch-to-mhlo-pipeline)")
        if debug == DebugType.PRINT_AFTER_FALIURE:
            pm.enable_ir_printing(
                print_before_pass=False,
                print_after_pass=True,
                print_after_only_on_change=False,
                print_after_only_on_failure=True,
                large_elements_limit=10,
            )
        if debug == DebugType.PRINT_AFTER_ONLY_CHANGE:
            pm.enable_ir_printing(
                print_before_pass=False,
                print_after_pass=True,
                print_after_only_on_change=True,
                print_after_only_on_failure=False,
                large_elements_limit=10,
            )
        pm.run(module.operation)
    if output_type == "stablehlo":
        return module
    return serialize_portable_artifact(module.operation.get_asm(), output_type.split('+')[1])


def compile_dynamo_model(
    model: Union[torch.export.ExportedProgram, torch.fx.GraphModule],
    output_type: str,
    backend_legal_ops: Optional[Sequence[str]] = None,
    debug: DebugType = DebugType.NO_DEBUG,
) -> ModuleOp:
    """
    Args:
        output_type: str type
            `raw`
            `torch`
            `stablehlo`
            `stablehlo+0.16.2`(stablehlo version, could specify other version)
        debug: DebugType, one of
            `NO_DEBUG: no debug message`,
            `PRINT_AFTER_FALIURE: print after failure`,
            `PRINT_AFTER_ONLY_CHANGE: print after pass only on change`
    """
    if output_type not in ["raw", "torch"] and "stablehlo" not in output_type:
        raise NotImplementedError(f"unsupported output type {output_type}")
    assert isinstance(debug, DebugType), "unsupported debug type"
    if backend_legal_ops is None:
        backend_legal_ops = _CUSTOM_OPS_IN_TORCH

    ### compile to raw by torch_mlir.extras.fx_importer
    torch_mlir_context = torch_mlir.ir.Context()
    from torch_mlir.dialects.torch import register_dialect as register_torch_dialect
    register_torch_dialect(torch_mlir_context)
    fx_importer = FxImporter(context=torch_mlir_context)
    # for torch.export
    if isinstance(model, torch.export.ExportedProgram):
        fx_importer.import_frozen_program(model)
    # for torch.compile
    elif isinstance(model, torch.fx.GraphModule):
        fx_importer.import_graph_module(model)
    else:
        raise RuntimeError("unsupported model type")
    module_str = fx_importer.module_op.operation.get_asm(enable_debug_info=True)

    context = ir.Context()
    module = ir.Module.parse(module_str, context)
    if output_type == "raw":
        return module
    if debug == DebugType.PRINT_AFTER_ONLY_CHANGE:
        print("// IR Dump After RAW")
        print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))
        print()
        sys.stdout.flush()

    if debug != DebugType.NO_DEBUG:
        module.context.enable_multithreading(False)

    with module.context:
        # We still need torch-function-to-torch-pipeline help us do something, e.g.,
        # decompose ops, like aten.addmm, aten.t and so on.
        option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + "}"
        pm = PassManager.parse(f"builtin.module(torch-function-to-torch-pipeline{option_string})")
        if debug == DebugType.PRINT_AFTER_FALIURE:
            pm.enable_ir_printing(
                print_before_pass=False,
                print_after_pass=True,
                print_after_only_on_change=False,
                print_after_only_on_failure=True,
                large_elements_limit=10,
            )
        if debug == DebugType.PRINT_AFTER_ONLY_CHANGE:
            pm.enable_ir_printing(
                print_before_pass=False,
                print_after_pass=True,
                print_after_only_on_change=True,
                print_after_only_on_failure=False,
                large_elements_limit=10,
            )
        pm.run(module.operation)

    if output_type == "torch":
        return module

    with module.context:
        pm = PassManager.parse("builtin.module(torch-to-mhlo-pipeline)")
        if debug == DebugType.PRINT_AFTER_FALIURE:
            pm.enable_ir_printing(
                print_before_pass=False,
                print_after_pass=True,
                print_after_only_on_change=False,
                print_after_only_on_failure=True,
                large_elements_limit=10,
            )
        if debug == DebugType.PRINT_AFTER_ONLY_CHANGE:
            pm.enable_ir_printing(
                print_before_pass=False,
                print_after_pass=True,
                print_after_only_on_change=True,
                print_after_only_on_failure=False,
                large_elements_limit=10,
            )
        pm.run(module.operation)

    if output_type == "stablehlo":
        return module
    return serialize_portable_artifact(module.operation.get_asm(), output_type.split('+')[1])


def convert_to_mhlo_via_torch_mlir(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    backend_legal_ops: Optional[Sequence[str]] = None,
    use_tracing: bool = False,
    verbose: bool = False,
) -> ModuleOp:
    """
    Deprecated, use torch_frontend.compile or torch_frontend.compile_dynamo_model instead.
    """
    if backend_legal_ops is None:
        backend_legal_ops = _CUSTOM_OPS_IN_TORCH
    # torch_mlir.BACKEND_LEGAL_OPS[torch_mlir.OutputType.TORCH] = backend_legal_ops
    from torch_mlir import torchscript
    module = torchscript.compile(
        model,
        example_inputs,
        output_type=torchscript.OutputType.RAW,
        use_tracing=use_tracing,
        verbose=False,
    )
    module_str = module.operation.get_asm(enable_debug_info=True)

    context = ir.Context()
    module = ir.Module.parse(module_str, context)
    with module.context:
        option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + "}"
        PassManager.parse(f"builtin.module(torchscript-to-torch-pipeline{option_string})").run(module.operation)

    with module.context:
        PassManager.parse("builtin.module(torch-to-mhlo-pipeline)").run(module.operation)

    return module
