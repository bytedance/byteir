import sys
import importlib.util

try:
    import torch_mlir
except ImportError:
    torch_mlir_module_spec = importlib.util.find_spec(__name__ + ".torch_mlir")
    torch_mlir_module = importlib.util.module_from_spec(torch_mlir_module_spec)
    sys.modules["torch_mlir"] = torch_mlir_module
    torch_mlir_module_spec.loader.exec_module(torch_mlir_module)
    del torch_mlir_module_spec
    del torch_mlir_module
except Exception:
    raise
else:
    # FIXME: handle reimport
    raise ImportError(
        "Found already installed or imported torch_mlir which might has conflicted with torch_frontend"
    )


from ._torch_frontend_registry import *

del sys
del importlib
del _torch_frontend_registry

from .ts_utils import register_decomposition_in_torchscript
from .fx_utils import list_decomposed_ops, preprocess_fx_graph
from .convert_to_mhlo import convert_to_mhlo_via_torch_mlir
