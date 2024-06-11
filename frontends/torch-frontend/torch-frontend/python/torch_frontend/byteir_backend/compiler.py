import functools
from typing import Optional, Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
from functorch.compile import min_cut_rematerialization_partition, default_partition
from torch._decomp import register_decomposition, get_decompositions, core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import (
    detect_fake_mode, )
from torch._inductor.fx_passes.joint_graph import joint_graph_passes
from torch._inductor.virtualized import V

from .inner_compile import (byteir_fx_compiler)
from .partitioners import fuse_aware_min_cut_partition


def byteir_decompositions():
    aten = torch.ops.aten

    decomposition_table = get_decompositions([
        aten.binary_cross_entropy_with_logits,
        aten.squeeze.dims,
        aten.log_sigmoid_forward,
        aten.threshold_backward,
        aten.slice_backward,
        aten.sigmoid_backward,
    ])

    return decomposition_table


def byteir_partition_fn(graph, joint_inputs, **kwargs):
    #joint_graph_passes(graph)
    return fuse_aware_min_cut_partition(graph,
                                        joint_inputs,
                                        **kwargs,
                                        compiler="inductor")


def byteir_compiler(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
):
    _byteir_compiler = aot_autograd(
        fw_compiler=functools.partial(byteir_fx_compiler, is_backward=False),
        bw_compiler=functools.partial(byteir_fx_compiler, is_backward=True),
        decompositions=byteir_decompositions,
        partition_fn=byteir_partition_fn,
        #partition_fn=min_cut_rematerialization_partition,
        #partition_fn=default_partition,
        keep_inference_input_mutations=False,
    )

    fake_mode = detect_fake_mode(
        example_inputs_) or torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True)
    tracing_context = (torch._guards.TracingContext.get()
                       or torch._guards.TracingContext(fake_mode))

    with V.set_fake_mode(fake_mode), torch._guards.tracing(tracing_context):
        return _byteir_compiler(model_, example_inputs_)
