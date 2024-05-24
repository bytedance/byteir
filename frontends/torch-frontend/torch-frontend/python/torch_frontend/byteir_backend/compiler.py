import functools

import torch
from functorch.compile import min_cut_rematerialization_partition, default_partition
from torch._decomp import register_decomposition, get_decompositions, core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

from .inner_compile import (byteir_fx_compiler)


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


byteir_compiler = aot_autograd(
    fw_compiler=functools.partial(byteir_fx_compiler, is_backward=False),
    bw_compiler=functools.partial(byteir_fx_compiler, is_backward=True),
    decompositions=byteir_decompositions,
    partition_fn=min_cut_rematerialization_partition,
    #partition_fn=default_partition,
)
