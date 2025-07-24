from typing import List, Optional, Union
import logging
import importlib

from tritontemplate import compiler,backend
from tritontemplate.compiler.kernel import TritonExecutor

_LOGGER = logging.getLogger(__name__)

def compile_kernel(
    op: compiler.base.Operation,
    device: str='cuda',
    workdir: str='./workshop',
    enable_tf32: bool=False
)->TritonExecutor:
    try:
        _ = importlib.import_module(f'tritontemplate.backend.{device}')
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f'Target {device} not found')
    return op.compile(device, workdir,enable_tf32)
    