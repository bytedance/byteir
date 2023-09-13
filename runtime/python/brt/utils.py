import brt

def brt_dtype_to_torch_dtype(dtype: brt.DType):
    import torch
    if dtype == brt.DType.float32:
        return torch.float32
    if dtype == brt.DType.int32:
        return torch.int32
    if dtype == brt.DType.int64:
        return torch.int64
    if dtype == brt.DType.uint8:
        return torch.uint8
    if dtype == brt.DType.float16:
        return torch.float16
    if dtype == brt.DType.float64:
        return torch.float64
    if dtype == brt.DType.bool:
        return torch.bool
    if dtype == brt.DType.int8:
        return torch.int8
    if dtype == brt.DType.int16:
        return torch.int16
    raise RuntimeError("unsupporetd data type: {}".format(int(dtype)))

