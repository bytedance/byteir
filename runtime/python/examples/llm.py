import _brt
import torch
from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete
import numpy as np
import os
import sys
import time

LLM_MODEL_PATH = sys.argv[1]

def ToTorchDType(dtype : _brt.DType):
    if dtype == _brt.DType.float16:
        return torch.float16
    if dtype == _brt.DType.float32:
        return torch.float32
    if dtype == _brt.DType.float64:
        return torch.float64
    if dtype == _brt.DType.bool:
        return torch.bool
    if dtype == _brt.DType.uint8:
        return torch.uint8
    if dtype == _brt.DType.int32:
        return torch.int32
    if dtype == _brt.DType.int64:
        return torch.int64
    return "unsupporetd data type"

def main():
    session = _brt.Session(alloc_func=caching_allocator_alloc, free_func=caching_allocator_delete)
    model_path = os.path.join(os.path.dirname(__file__), LLM_MODEL_PATH)
    session.load(model_path)
    req = session.new_request_context(torch.cuda.current_stream()._as_parameter_.value)

    inputs = []
    outputs = []

    for offset in session.get_input_arg_offsets():
        dtype = ToTorchDType(session.get_data_type(offset))
        if dtype == torch.int64:
            inputs.append(torch.zeros(session.get_static_shape(offset), dtype=dtype, device="cuda"))
        else:
            data = np.random.random(size=session.get_static_shape(offset))
            inputs.append(torch.tensor(data, dtype=dtype, device="cuda"))
        req.bind_arg(offset, inputs[-1].data_ptr())

    for offset in session.get_output_arg_offsets():
        dtype = ToTorchDType(session.get_data_type(offset))
        outputs.append(torch.empty(session.get_static_shape(offset), dtype=dtype, device="cuda"))
        req.bind_arg(offset, outputs[-1].data_ptr())

    req.finish_io_binding()
    t0 = time.time()
    req.run()
    req.sync()
    t1 = time.time()
    print("E2E execution time {} ms".format((t1 - t0) * 1000))


if __name__ == "__main__":
    main()
