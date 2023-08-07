import brt
import torch
from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete
import numpy as np
import os

def main():
    session = brt.Session(alloc_func=caching_allocator_alloc, free_func=caching_allocator_delete)
    ait_dir = os.path.join(os.path.dirname(__file__), "../../test/test_files/AITOp")
    model_path = os.path.join(ait_dir, "bmm_permute_entry.mlir")
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

    b = inputs[0].size(dim=0)
    m = inputs[0].size(dim=1)
    N = inputs[1].size(dim=2)
    d1 = outputs[0].size(dim=2)

    Y_l = torch.bmm(inputs[0], inputs[1])
    Y_r = Y_l.reshape(b // d1, d1, m, N)
    Y_pt = torch.permute(Y_r, [0, 2, 1, 3])

    torch.testing.assert_close(Y_pt, outputs[0], atol=1e-3, rtol=1e-3)
    print("bmm_permute numerical test pass")

if __name__ == "__main__":
    main()
