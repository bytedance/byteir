import functools
import logging
from typing import Optional, Any, Callable, Dict, List, Sequence, Tuple, Union

import torch

from .compiler import byteir_compiler

log = logging.getLogger(__name__)


def debug_backend(gm: torch.fx.GraphModule,
                  example_inputs: List[torch.Tensor]):
    """
    compare results between byteir compiled function and eager mode graph.
    """
    _opt_gm = byteir_compiler(gm, example_inputs)

    def f(*inputs):
        opt_inputs = []
        for inp in inputs:
            _opt_inp = torch.empty_strided(size=inp.size(),
                                           stride=inp.stride(),
                                           storage_offset=inp.storage_offset())
            opt_inputs.append(_opt_inp.copy_(inp))

        eager_inputs = inputs
        eager_res = gm(*eager_inputs)
        opt_res = _opt_gm(*opt_inputs)

        # compare results
        # TODO: check meta info as well as numercial.
        try:
            torch.testing.assert_close(eager_res, opt_res)
        except Exception as e:
            log.error(f"******* debug backend fail *******")
            raise e

        print(f"******* debug backend pass *******")
        return eager_res

    return f
