import brt
from brt.utils import brt_dtype_to_torch_dtype
import torch

import time

# BRTBackend for static shape and single device
class BRTBackend:
    def __init__(self, byre_file_path, device):
        assert device == "cuda" or device == "cpu"
        if device == "cuda":
            from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete
            _allocator_alloc = caching_allocator_alloc
            _allocator_delete = caching_allocator_delete
            _stream = torch.cuda.current_stream()._as_parameter_.value
        else:
            _allocator_alloc = None
            _allocator_delete = None
            _stream = None
        self.session = brt.Session(
            device=device.upper(),
            alloc_func=_allocator_alloc,
            free_func=_allocator_delete,
        )
        self.session.load(byre_file_path)
        self.req = self.session.new_request_context(_stream)
        self.device = device

        # for static shape model, just cache shape and dtype info
        self.input_arg_offsets = self.session.get_input_arg_offsets()
        self.input_shapes = []
        self.input_dtypes = []
        for offset in self.input_arg_offsets:
            self.input_shapes.append(self.session.get_static_shape(offset))
            self.input_dtypes.append(brt_dtype_to_torch_dtype(self.session.get_data_type(offset)))
        self.output_arg_offsets = self.session.get_output_arg_offsets()
        self.output_shapes = []
        self.output_dtypes = []
        for offset in self.output_arg_offsets:
            self.output_shapes.append(self.session.get_static_shape(offset))
            self.output_dtypes.append(brt_dtype_to_torch_dtype(self.session.get_data_type(offset)))

    def _check_shape_dtype(self, tensors, shapes, dtypes):
        assert len(tensors) == len(shapes)
        assert len(tensors) == len(dtypes)
        for tensor, shape, dtype in zip(tensors, shapes, dtypes):
            assert list(shape) == list(tensor.shape)
            assert dtype == tensor.dtype

    def _bind_inputs(self, inputs):
        inputOffsetAndData = []
        for offset, input in zip(self.input_arg_offsets, inputs):
            inputOffsetAndData.append((offset, input.data_ptr()))
        self.req.bind_args(inputOffsetAndData)

    def _bind_outputs(self, outputs):
        outputOffsetAndData = []
        for offset, output in zip(self.output_arg_offsets, outputs):
            outputOffsetAndData.append((offset, output.data_ptr()))
        self.req.bind_args(outputOffsetAndData)

    def run(self, inputs, check=True):
        if check:
            self._check_shape_dtype(inputs, self.input_shapes, self.input_dtypes)

        # alloc outputs
        outputs = []
        for shape, dtype in zip(self.output_shapes, self.output_dtypes):
            outputs.append(torch.empty(shape, dtype=dtype, device=self.device))

        self._bind_inputs(inputs)
        self._bind_outputs(outputs)

        # run
        self.req.finish_io_binding()
        self.req.run()
        self.req.sync()
        
        return outputs

    def profile(self, inputs, check=True, warmup_trials=10, run_trials=50):
        if check:
            self._check_shape_dtype(inputs, self.input_shapes, self.input_dtypes)

        # alloc outputs
        outputs = []
        for shape, dtype in zip(self.output_shapes, self.output_dtypes):
            outputs.append(torch.empty(shape, dtype=dtype, device=self.device))

        self._bind_inputs(inputs)
        self._bind_outputs(outputs)
        self.req.finish_io_binding()

        # warmup
        for _ in range(warmup_trials):
            self.req.run()
            self.req.sync()
        
        start = time.time()
        for _ in range(run_trials):
            self.req.run()
            self.req.sync()
        end = time.time()
        avg = ((end - start) * 1000) / run_trials

        return outputs, avg

    def run_with_outputs(self, inputs, outputs, check=True):
        if check:
            self._check_shape_dtype(inputs, self.input_shapes, self.input_dtypes)
            self._check_shape_dtype(outputs, self.output_shapes, self.output_dtypes)

        self._bind_inputs(inputs)
        self._bind_outputs(outputs)

        self.req.finish_io_binding()
        self.req.run()
        self.req.sync()

    def profile_with_outputs(self, inputs, outputs, check=True, warmup_trials=10, run_trials=50):
        if check:
            self._check_shape_dtype(inputs, self.input_shapes, self.input_dtypes)
            self._check_shape_dtype(outputs, self.output_shapes, self.output_dtypes)
        
        self._bind_inputs(inputs)
        self._bind_outputs(outputs)
        self.req.finish_io_binding()

        # warmup
        for _ in range(warmup_trials):
            self.req.run()
            self.req.sync()
        
        start = time.time()
        for _ in range(run_trials):
            self.req.run()
            self.req.sync()
        end = time.time()
        avg = ((end - start) * 1000) / run_trials

        return avg


# TODO: add BRTDynamicShapeBackend and BRTNCCLBackend
