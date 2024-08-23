import brt
from brt.utils import brt_dtype_to_torch_dtype
import torch

class BRTBackend:
    def __init__(self, device, brt_file_path):
        assert device == "cuda" or device == "cpu"
        from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete

        _allocator_alloc = caching_allocator_alloc if device == "cuda" else None
        _allocator_delete = caching_allocator_delete if device == "cuda" else None
        _stream = (
            torch.cuda.current_stream()._as_parameter_.value
            if device == "cuda"
            else None
        )
        self.session = brt.Session(
            device=device.upper(),
            alloc_func=_allocator_alloc,
            free_func=_allocator_delete,
        )
        self.session.load(brt_file_path)
        self.req = self.session.new_request_context(_stream)
        self.device = device

    def execute(self, inputs):
        # TODO(lyq): how to support dynamic shape?
        assert len(self.session.get_input_arg_offsets()) == len(inputs)
        outputs = []
        for offset, arg in zip(self.session.get_input_arg_offsets(), inputs):
            assert list(self.session.get_static_shape(offset)) == list(arg.shape)
            assert brt_dtype_to_torch_dtype(self.session.get_data_type(offset)) == arg.dtype
            self.req.bind_arg(offset, arg.data_ptr())
        for offset in self.session.get_output_arg_offsets():
            shape = self.session.get_static_shape(offset)
            outputs.append(torch.empty(shape, dtype=brt_dtype_to_torch_dtype(self.session.get_data_type(offset)), device=self.device))
            self.req.bind_arg(offset, outputs[-1].data_ptr())
        
        self.req.finish_io_binding()
        self.req.run()
        self.req.sync()
        
        return outputs
