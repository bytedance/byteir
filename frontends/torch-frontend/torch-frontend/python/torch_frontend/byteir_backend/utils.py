import base64
import contextlib
import dataclasses
import hashlib
import os
import functools
import time
from typing import Optional, Any, Callable, Dict, List, Sequence, Tuple, Union
import pickle

from dataclasses import dataclass

import torch

from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    FakeTensor,
)
from torch._prims_common import suggest_memory_format

## Helper classes portting from torch2.4
class BypassFxGraphCache(Exception):
    """
    Exception to indicate that the FxGraphCache should be bypassed.
    """

    pass

@dataclasses.dataclass
class OrderedSetHolder:
    """
    See FxGraphHashDetails. Holds a sorted list to support stable hashing
    of set kwargs.
    """

    items: List[Any]

@dataclass(frozen=True)
class TensorMetadata:
    """
    The Tensor metadata relevant to hashing FakeTensors when caching.
    """

    dtype: torch.dtype
    shape: torch.Size
    stride: Tuple[Any, ...]
    device: torch.device
    layout: torch.layout
    memory_format: Optional[torch.memory_format]
    storage_offset: int
    storage_bytes: Optional[int]
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_inference: bool
    is_sparse: bool  # read: is sparse COO
    is_coalesced: Optional[bool]
    dense_dim: Optional[int]
    sparse_dim: Optional[int]



## Helper functions portting from torch2.4
def is_sparse_coo(t):
    return isinstance(t, torch.Tensor) and t.layout is torch.sparse_coo


def is_sparse_compressed_layout(layout):
    return layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }


def is_sparse_compressed(t):
    return isinstance(t, torch.Tensor) and is_sparse_compressed_layout(t.layout)


def is_sparse_any(t):
    return is_sparse_coo(t) or is_sparse_compressed(t)

def extract_tensor_metadata(t: torch.Tensor) -> "TensorMetadata":
    """
    Extract the TensorMetadata of a tensor.
    """
    memory_format: Optional[torch.memory_format] = suggest_memory_format(t)
    if is_sparse_any(t) or not t.is_contiguous(memory_format=memory_format):
        memory_format = None

    return TensorMetadata(
        dtype=t.dtype,
        shape=t.shape,
        stride=t.stride() if t.layout == torch.strided else (),
        device=t.device,
        layout=t.layout,
        memory_format=memory_format,
        storage_offset=t.storage_offset(),
        # Only set storage_bytes for tensors that have storage (not sparse)
        storage_bytes=t.untyped_storage().nbytes() if not t.is_sparse else None,
        requires_grad=t.requires_grad,
        is_quantized=t.is_quantized,
        is_conj=t.is_conj(),
        is_neg=t.is_neg(),
        is_inference=t.is_inference(),
        is_sparse=t.is_sparse,
        is_coalesced=t.is_coalesced() if t.is_sparse else None,
        dense_dim=t.dense_dim() if t.is_sparse else None,
        sparse_dim=t.sparse_dim() if t.is_sparse else None,
    )

def sha256_hash(data: bytes) -> str:
    # [:51] to strip off the "Q====" suffix common to every hash value.
    return base64.b32encode(hashlib.sha256(data).digest())[:51].decode("utf-8").lower()

def _ident(x: Any) -> Any:
    return x

def _reduce_fake_tensor(t):
    """
    See FxGraphCachePickler. Custom reducer to pickle FakeTensors.
    """
    metadata = extract_tensor_metadata(t)
    return (_ident, (metadata,))

def _reduce_symint(s):
    """
    See FxGraphCachePickler. Custom reducer to pickle SymInts.
    """
    # For hashing purposes, we only care about the name of the symbol and
    # not the backed value. We evaluate guards stored with a cached graph
    # to ensure a cached entity with SymInt args is safe to reuse.
    return (_ident, (str(s),))

def maybe_get_fake_mode(t):
    if isinstance(t, FakeTensor):
        return t.fake_mode
    if is_traceable_wrapper_subclass(t):
        inner_tensor_names, _ = t.__tensor_flatten__()
        modes = [
            maybe_get_fake_mode(getattr(t, t_name)) for t_name in inner_tensor_names
        ]
        m = modes[0]
        assert all(m is x for x in modes)
        return m
    elif isinstance(t, torch.Tensor) and torch._is_functional_tensor(t):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(t, reapply_views)
        return maybe_get_fake_mode(unwrapped)
    elif isinstance(t, torch.Tensor) and is_functorch_wrapped_tensor(t):
        unwrapped = torch._C._functorch.get_unwrapped(t)
        return maybe_get_fake_mode(unwrapped)
    return None



def record_execution_time(stage: str = "Unknown"):

    def decorator(func):
        assert isinstance(func, Callable)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _start = time.time()
            outs = func(*args, **kwargs)
            _end = time.time()
            print(f"{stage} stage consume: {(_end - _start)} s")
            return outs

        return wrapper

    return decorator


def dump_tensors_meta_info(tensors: list[torch.Tensor, FakeTensor],
                           save_path: str):
    _meta_infos = []
    for t in tensors:
        if t is None:
            _meta_infos.append(None)
        else:
            _meta_infos.append(extract_tensor_metadata(t))
    with open(save_path, "wb") as f:
        pickle.dump(_meta_infos, f)

def cal_storage_size(size, stride, storage_offset):
    _size = storage_offset
    assert len(size) == len(stride)
    for i in range(0, len(size)):
        _size = _size + (size[i] - 1) * stride[i]
    _size = _size + 1
    return _size

def create_real_tensor(size, dtype, layout, device, requires_grad, stride,
                       storage_offset):
    storage_size = cal_storage_size(size, stride, storage_offset)

    if dtype == torch.int32:
        rt = torch.randint(low=0,
                           high=5,
                           size=(1, storage_size),
                           dtype=dtype,
                           layout=layout,
                           device=device,
                           requires_grad=requires_grad)
    elif dtype == torch.int64:
        rt = torch.randint(low=0,
                           high=2,
                           size=(1, storage_size),
                           dtype=dtype,
                           layout=layout,
                           device=device,
                           requires_grad=requires_grad)
    elif dtype == torch.bool:
        rt = torch.randint(low=0,
                           high=2,
                           size=(1, storage_size),
                           dtype=torch.int32,
                           layout=layout,
                           device=device,
                           requires_grad=requires_grad)
        rt = rt.to(dtype)
    else:
        rt = torch.rand(size=(1, storage_size),
                        dtype=dtype,
                        layout=layout,
                        device=device,
                        requires_grad=requires_grad)
    rt = torch.as_strided(rt,
                          size=size,
                          stride=stride,
                          storage_offset=storage_offset)

    return rt


def create_real_tensors_from_meta_info(pkl: str):
    rets = []
    if not os.path.exists(pkl):
        return rets
    with open(pkl, "rb") as f:
        meta_infos = pickle.load(f)
        for meta in meta_infos:
            if meta is None:
                rt = None
            else:
                rt = create_real_tensor(meta.shape, meta.dtype, meta.layout,
                                        meta.device, meta.requires_grad,
                                        meta.stride, meta.storage_offset)
            rets.append(rt)
    return rets


def create_real_tensor_from_fake(ft: FakeTensor):
    rt = create_real_tensor(size=ft.size(),
                            dtype=ft.dtype,
                            layout=ft.layout,
                            device=ft.device,
                            requires_grad=ft.requires_grad,
                            stride=ft.stride(),
                            storage_offset=ft.storage_offset())
    return rt
