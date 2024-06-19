import functools
import os
import logging

from typing import Optional, Any, Callable, Dict, List, Sequence, Tuple, Union
from itertools import count

import torch
from torch._dynamo import (
    utils as dynamo_utils, )
from torch._dynamo.utils import counters
from torch._dynamo.utils import detect_fake_mode
from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    FakeTensor,
    FakeTensorConverter,
)

import torch_frontend
from torch_frontend import get_none_indices
from torch_frontend.compile import DebugType

try:
    import byteir
    import brt
except ImportError:
    ...

from .compilation_cache import (
    compiled_fx_graph_hash,
    ByteIRFxGraphCache,
)
from .compiled_function import (
    CompiledArtifact,
    ByteIRFunction,
)
from .utils import (
    dump_tensors_meta_info,
    BypassFxGraphCache,
    OrderedSetHolder,
    TensorMetadata,
    extract_tensor_metadata,
    maybe_get_fake_mode,
)
from . import config

log = logging.getLogger(__name__)
g_graph_counter = count(0)

BACKEND_LEGAL_OPS = ["aten.max.dim"]


#@dynamo_utils.dynamo_timed(phase_name="byteir_compile")
def inner_compile(gm: torch.fx.GraphModule,
                  example_inputs: List[torch.Tensor],
                  workdir: str = None,
                  compiler_type: str = "forward",
                  **kwargs) -> CompiledArtifact:

    graph_id = next(g_graph_counter)
    log.info(f"byteir compiling {compiler_type} graph {graph_id}")

    if workdir is None:
        key = compiled_fx_graph_hash(gm, example_inputs, kwargs)
        workdir = ByteIRFxGraphCache._get_tmp_dir_for_key(key)

    stablehlo_fiel_name = f"model.stablehlo.mlir"
    byre_file_name = f"model.byre.mlir"
    stablehlo_file = f"{workdir}/{stablehlo_fiel_name}"
    byre_file = f"{workdir}/{byre_file_name}"

    os.makedirs(workdir, exist_ok=True)

    if config.byteir_save_fxgraph:
        # save fx graph, example inputs and outs
        fxg_dir_name = f"fx_graph_{compiler_type}_{graph_id}"
        fx_graph_folder = f"{workdir}/{fxg_dir_name}/"
        os.makedirs(fx_graph_folder, exist_ok=True)
        with maybe_disable_fake_tensor_mode():
            gm.to_folder(folder=fx_graph_folder, module_name="FxModule")
        with detect_fake_mode(example_inputs):
            #with FakeTensorMode(allow_non_fake_inputs=True):
            fake_outs = gm(*example_inputs)
        dump_tensors_meta_info(
            example_inputs,
            os.path.join(fx_graph_folder, "inputs_meta_info.pkl"))
        dump_tensors_meta_info(
            fake_outs, os.path.join(fx_graph_folder, "outputs_meta_info.pkl"))

    # FIXME. torch-mlir importer requires disable FakeTensorMode.
    with maybe_disable_fake_tensor_mode():
        if not os.path.exists(stablehlo_file):
            module = torch_frontend.compile_dynamo_model(
                gm,
                output_type="stablehlo",
                backend_legal_ops=BACKEND_LEGAL_OPS)
            with open(stablehlo_file, "w") as f:
                print(module.operation.get_asm(), file=f)
        if not os.path.exists(byre_file):
            byteir.compile(stablehlo_file,
                           byre_file,
                           verbose=False,
                           target="cuda")
            #byteir.compile(stablehlo_file, byre_file, verbose=False, target="cuda_with_ait")

        byre_session = brt.Session(alloc_func=caching_allocator_alloc,
                                   free_func=caching_allocator_delete)
        byre_session.load(byre_file)
    log.debug("#### byteir compile success")
    none_indices = get_none_indices(gm)

    compiled_artifact = CompiledArtifact(byre_file, none_indices)

    return compiled_artifact


def byteir_fx_compiler(gm: torch.fx.GraphModule,
                       example_inputs,
                       is_backward=False):
    """
    The main entry function of byteir torch compiler backend.
    """

    compiler_type = "backward" if is_backward else "forward"
    log.info(
        f"########################### {'FORWARD' if not is_backward else 'BACKWARD'} ###########################"
    )
    log.info(torch._guards.TracingContext.get())

    if config.byteir_not_use_cache:
        compiled_artifact = inner_compile(gm, example_inputs)
        byre_session = brt.Session(alloc_func=caching_allocator_alloc,
                                   free_func=caching_allocator_delete)
        byre_session.load(compiled_artifact.byre_file)
        byre_func = ByteIRFunction(byre_session,
                                   compiled_artifact.none_indices)
    else:
        byre_func = ByteIRFxGraphCache.Load(
            functools.partial(inner_compile, compiler_type=compiler_type), gm,
            example_inputs)

    log.debug(f"Counters:\n{counters}")
    return byre_func
