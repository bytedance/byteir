from typing import Optional, Sequence, Union, List, Tuple
from enum import Enum
import sys
import os

import torch
import torch.export

import torch_mlir
from torch_mlir.extras.fx_importer import FxImporter

from . import ir
from .passmanager import PassManager
from ._mlir_libs._stablehlo import serialize_portable_artifact

from .extra_shape_fn import byteir_extra_library

GENERIC_CUSTOM_OPS = [
    "byteir.softmax",
    "byteir.log_softmax",
    "byteir.layer_norm",
    "byteir.l2_norm",
    "byteir.gelu",
    "byteir.arg_max",
    "byteir.arg_min",
    "byteir.one_hot",
    "byteir.topk",
    "byteir.non_zero",
    "byteir.resize",
]

MATH_CUSTOM_OPS = [
    "math.asin",
    "math.asinh",
    "math.sinh",
    "math.atan",
    "math.tan",
    "math.acos",
    "math.acosh",
    "math.cosh",
    "math.erf",
    "math.trunc",
]

BYTEIR_CUSTOM_OPS = [
    "byteir.flash_attn_fwd",
    "byteir.flash_attn_kvcache",
    "byteir.flash_attn_bwd",
]

CUSTOM_OP_MAP = {
    "byteir.softmax": ["aten._softmax", "aten.softmax.int"],
    "byteir.log_softmax": ["aten._log_softmax", "aten.log_softmax.int"],
    "byteir.layer_norm": ["aten.native_layer_norm", "aten.layer_norm", "aten.group_norm", "aten.native_group_norm"],
    "byteir.l2_norm": ["aten.linalg_vector_norm", "aten.expand_as"],
    "byteir.gelu": ["aten.gelu"],
    "byteir.arg_max": ["aten.max.dim"],
    "byteir.arg_min": ["aten.min.dim"],
    "byteir.one_hot": ["aten.one_hot"],
    "byteir.topk": ["aten.topk"],
    "byteir.non_zero": ["aten.nonzero"],
    "byteir.resize": ["aten.upsample_nearest2d.vec"],
    "math.asin": ["aten.asin"],
    "math.asinh": ["aten.asinh"],
    "math.sinh": ["aten.sinh"],
    "math.atan": ["aten.atan"],
    "math.tan": ["aten.tan"],
    "math.acos": ["aten.acos"],
    "math.acosh": ["aten.acosh"],
    "math.cosh": ["aten.cosh"],
    "math.erf": ["aten.erf"],
    "math.trunc": ["aten.trunc"],
    # torch.operator
    "byteir.flash_attn_fwd": ["byteir.flash_attn_fwd"],
    "byteir.flash_attn_kvcache": ["byteir.flash_attn_kvcache"],
    "byteir.flash_attn_bwd": ["byteir.flash_attn_bwd"],
}

# ops which should not be decomposed by torch-mlir 
OPS_NOT_DECOMPOSE_BY_TORCH_MLIR = [
    "aten.randn.generator",
    "aten.normal_functional",
    "aten.amax",
    "aten.amin",
] + [
    "aten.var.dim",
    "aten.var.correction",
]


class DebugType(Enum):
    NO_DEBUG = 0
    PRINT_AFTER_FAILURE = 1
    PRINT_AFTER_ONLY_CHANGE = 2


def _get_debug_parameters(debug: DebugType):
    assert isinstance(debug, DebugType), "unknown debug type"
    # note: if you want to set `print_module_scope = True``,
    # you should set `module.context.enable_multithreading(False)`
    debug_parameters = {}
    if debug == DebugType.PRINT_AFTER_FAILURE:
        debug_parameters = {
            "print_before_pass": False,
            "print_after_pass": True,
            "print_after_only_on_change": False,
            "print_after_only_on_failure": True,
            "print_module_scope": False,
            "large_elements_limit": 10,
        }
    elif debug == DebugType.PRINT_AFTER_ONLY_CHANGE:
        debug_parameters = {
            "print_before_pass": False,
            "print_after_pass": True,
            "print_after_only_on_change": True,
            "print_after_only_on_failure": False,
            "print_module_scope": False,
            "large_elements_limit": 10,
        }
    return debug_parameters

def _print_verbose(module: ir.Module, pipeline_msg: str):
    print(pipeline_msg)
    print(module.operation.get_asm(large_elements_limit=10))
    print()

def _get_extra_library_file(backend_legal_ops: Sequence[str]):
    CUR_DIR = os.path.abspath(os.path.dirname(__file__))
    for extra_op in byteir_extra_library:
        if extra_op in backend_legal_ops:
            return str(os.path.join(CUR_DIR, "tools", "extra_fn.mlir"))
    return ""

def _get_aten_ops_by_map(backend_legal_ops: Sequence[str]):
    aten_ops = []
    for op in backend_legal_ops:
        if op in CUSTOM_OP_MAP:
            aten_ops += CUSTOM_OP_MAP[op]
        else:
            print(f"Warning: unknown custom op {op}")
    return aten_ops

def compile(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    output_type: str,
    entry_func_name: str = "forward",
    backend_legal_ops: Optional[Sequence[str]] = None,
    verbose: bool = False,
    debug: DebugType = DebugType.NO_DEBUG,
) -> ir.Module:
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
    if backend_legal_ops is None:
        backend_legal_ops = GENERIC_CUSTOM_OPS
    backend_legal_aten_ops = _get_aten_ops_by_map(backend_legal_ops)
    debug_parameters = _get_debug_parameters(debug)

    ############################################
    # compile to raw by torch_mlir.torchscript
    ############################################
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
    _print_verbose(module, "// IR Dump After JIT IR Importer") if verbose else ...
    if output_type == "raw":
        return module

    ############################################
    # compile raw to torch
    ############################################
    extra_library_file_name = _get_extra_library_file(backend_legal_ops)
    with module.context:
        option_string = (
            "{backend-legal-ops="
            + ",".join(backend_legal_aten_ops + OPS_NOT_DECOMPOSE_BY_TORCH_MLIR)
            + " extra-library="
            + extra_library_file_name
            + "}"
        )
        pm = PassManager.parse(
            f"builtin.module(torchscript-to-torch-pipeline{option_string})"
        )
        if debug != DebugType.NO_DEBUG:
            pm.enable_ir_printing(**debug_parameters)
        pm.run(module.operation)
    _print_verbose(module, "// IR Dump After Torch Backend Pipeline") if verbose else ...
    
    # customize decompose on torch dialect
    with module.context:
        option_string = (
            "{legal-ops="
            + ",".join(backend_legal_aten_ops)
            + "}"
        )
        option_string_1 = (
            "{legal-ops="
            + ",".join(backend_legal_aten_ops + OPS_NOT_DECOMPOSE_BY_TORCH_MLIR)
            + "}"
        )
        pm = PassManager.parse(
            f"builtin.module(func.func(decompose-on-torch{option_string}),"
            + "canonicalize,"
            + f"func.func(torch-decompose-complex-ops{option_string_1}),"
            + "canonicalize)"
        )
        if debug != DebugType.NO_DEBUG:
            pm.enable_ir_printing(**debug_parameters)
        pm.run(module.operation)
    _print_verbose(module, "// IR Dump After Customize Decomposition on Torch Dialect") if verbose else ...

    if output_type == "torch":
        return module

    ############################################
    # lowering torch to stablehlo
    ############################################
    with module.context:
        # FIXME: change `backend-legal-ops` to another name?
        option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + "}"
        pm = PassManager.parse(f"builtin.module(torch-to-stablehlo-pipeline{option_string})")
        if debug != DebugType.NO_DEBUG:
            pm.enable_ir_printing(**debug_parameters)
        pm.run(module.operation)
    with module.context:
        option_string = "{target-name=" + entry_func_name + "}"
        pm = PassManager.parse(
            f"builtin.module(rewrite-entry-func-name{option_string})"
        )
        if debug != DebugType.NO_DEBUG:
            pm.enable_ir_printing(**debug_parameters)
        pm.run(module.operation)
    _print_verbose(module, "// IR Dump After Torch to Stablehlo Pipeline") if verbose else ...
    if output_type == "stablehlo":
        return module

    ############################################
    # serialize stablehlo to target version
    ############################################
    return serialize_portable_artifact(
        module.operation.get_asm(), output_type.split("+")[1]
    )


def compile_dynamo_model(
    model: Union[torch.export.ExportedProgram, torch.fx.GraphModule],
    output_type: str,
    entry_func_name: str = "main",
    backend_legal_ops: Optional[Sequence[str]] = None,
    verbose: bool = False,
    debug: DebugType = DebugType.NO_DEBUG,
) -> ir.Module:
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
    if backend_legal_ops is None:
        backend_legal_ops = GENERIC_CUSTOM_OPS
    backend_legal_aten_ops = _get_aten_ops_by_map(backend_legal_ops)
    debug_parameters = _get_debug_parameters(debug)

    ##################################################
    # compile to raw by torch_mlir.extras.fx_importer
    ##################################################
    torch_mlir_context = torch_mlir.ir.Context()
    from torch_mlir.dialects.torch import register_dialect as register_torch_dialect

    register_torch_dialect(torch_mlir_context)
    fx_importer = FxImporter(context=torch_mlir_context)
    # for torch.export
    if isinstance(model, torch.export.ExportedProgram):
        fx_importer.import_frozen_program(model, func_name=entry_func_name)
    # for torch.compile
    elif isinstance(model, torch.fx.GraphModule):
        fx_importer.import_stateless_graph(model.graph, func_name=entry_func_name)
    else:
        raise RuntimeError("unsupported model type")
    module_str = fx_importer.module_op.operation.get_asm(enable_debug_info=True)

    context = ir.Context()
    module = ir.Module.parse(module_str, context)
    _print_verbose(module, "// IR Dump After FX Importer") if verbose else ...
    if output_type == "raw":
        return module

    ############################################
    # compile raw to torch
    ############################################
    extra_library_file_name = _get_extra_library_file(backend_legal_ops)
    with module.context:
        # We still need torch-function-to-torch-pipeline help us do something, e.g.,
        # decompose ops, like aten.addmm, aten.t and so on.
        option_string = (
            "{shape-dtype-refine=false"
            + " backend-legal-ops="
            + ",".join(backend_legal_aten_ops + OPS_NOT_DECOMPOSE_BY_TORCH_MLIR)
            + " extra-library="
            + extra_library_file_name
            + "}"
        )
        pm = PassManager.parse(
            f"builtin.module(torch-function-to-torch-pipeline{option_string})"
        )
        if debug != DebugType.NO_DEBUG:
            pm.enable_ir_printing(**debug_parameters)
        pm.run(module.operation)
    _print_verbose(module, "// IR Dump After Torch Backend Pipeline") if verbose else ...

    # customize decompose on torch dialect
    with module.context:
        option_string = (
            "{legal-ops="
            + ",".join(backend_legal_aten_ops)
            + "}"
        )
        option_string_1 = (
            "{legal-ops="
            + ",".join(backend_legal_aten_ops + OPS_NOT_DECOMPOSE_BY_TORCH_MLIR)
            + "}"
        )
        pm = PassManager.parse(
            f"builtin.module(func.func(decompose-on-torch{option_string}),"
            + "canonicalize,"
            + f"func.func(torch-decompose-complex-ops{option_string_1}),"
            + "canonicalize)"
        )
        if debug != DebugType.NO_DEBUG:
            pm.enable_ir_printing(**debug_parameters)
        pm.run(module.operation)
    _print_verbose(module, "// IR Dump After Customize Decomposition on Torch Dialect") if verbose else ...

    if output_type == "torch":
        return module

    ############################################
    # lowering torch to stablehlo
    ############################################
    with module.context:
        option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + "}"
        pm = PassManager.parse(f"builtin.module(torch-to-stablehlo-pipeline{option_string})")
        if debug != DebugType.NO_DEBUG:
            pm.enable_ir_printing(**debug_parameters)
        pm.run(module.operation)
    _print_verbose(module, "// IR Dump After Torch to Stablehlo Pipeline") if verbose else ...
    if output_type == "stablehlo":
        return module

    ############################################
    # serialize stablehlo to target version
    ############################################
    return serialize_portable_artifact(
        module.operation.get_asm(), output_type.split("+")[1]
    )
