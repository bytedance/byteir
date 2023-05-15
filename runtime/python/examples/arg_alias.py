import _brt
import torch
from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete
import numpy as np
import os

def copy():
    session = _brt.Session(alloc_func=caching_allocator_alloc, free_func=caching_allocator_delete)
    model_path = os.path.join(os.path.dirname(__file__), "arg_alias.mlir")
    session.load(model_path)
    req = session.new_request_context(torch.cuda.current_stream()._as_parameter_.value)

    inputs = []
    outputs = []

    for offset in session.get_input_arg_offsets():
        data = np.random.random(size=session.get_static_shape(offset))
        inputs.append(torch.tensor(data, dtype=torch.float32, device="cuda"))
        req.bind_arg(offset, inputs[-1].data_ptr())

    for offset in session.get_output_arg_offsets():
        outputs.append(torch.empty(session.get_static_shape(offset), dtype=torch.float32, device="cuda"))
        req.bind_arg(offset, outputs[-1].data_ptr())

    req.finish_io_binding()
    req.run()
    req.sync()

    torch.testing.assert_close(inputs[0], outputs[0])
    torch.testing.assert_close(inputs[1], outputs[1])

def not_copy():
    session = _brt.Session(alloc_func=caching_allocator_alloc, free_func=caching_allocator_delete)
    model_path = os.path.join(os.path.dirname(__file__), "arg_alias.mlir")
    session.load(model_path)
    req = session.new_request_context(torch.cuda.current_stream()._as_parameter_.value)

    inputs = []
    outputs = []

    input_arg_offsets = session.get_input_arg_offsets()
    output_arg_offsets = session.get_output_arg_offsets()

    for offset in input_arg_offsets:
        data = np.random.random(size=session.get_static_shape(offset))
        inputs.append(torch.tensor(data, dtype=torch.float32, device="cuda"))
        req.bind_arg(offset, inputs[-1].data_ptr())

    for offset in output_arg_offsets:
        alias_offset = session.get_graph_arg_alias_offset(offset)
        if alias_offset != -1:
            if alias_offset in input_arg_offsets:
                req.bind_arg(offset, inputs[input_arg_offsets.index(alias_offset)].data_ptr())
                outputs.append(inputs[input_arg_offsets.index(alias_offset)])
            elif alias_offset in output_arg_offsets:
                req.bind_arg(offset, outputs[output_arg_offsets.index(alias_offset)].data_ptr())
                outputs.append(outputs[output_arg_offsets.index(alias_offset)])
            else:
                raise Exception("alias offset should fall in inputs or outputs")
        else:
            outputs.append(torch.empty(session.get_static_shape(offset), dtype=torch.float32, device="cuda"))
            req.bind_arg(offset, outputs[-1].data_ptr())

    req.finish_io_binding()
    req.run()
    req.sync()

    torch.testing.assert_close(inputs[0], outputs[0])
    torch.testing.assert_close(inputs[1], outputs[1])

if __name__ == "__main__":
    copy()
    not_copy()
