import os
import functools
import time
import pickle

import torch

from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    FakeTensor,
    extract_tensor_metadata,
)


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
