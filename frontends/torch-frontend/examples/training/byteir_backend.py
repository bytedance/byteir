import os
import functools

import torch
from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from functorch.compile import aot_module
from torch._decomp import get_decompositions

# byteir components
import brt
import byteir
from torch_frontend import compile
from torch_frontend import list_decomposed_ops, preprocess_fx_graph

TRACE = False
cnt = 0
model_name = "model"
TEMP_FOLDER = "./temp"
byteir_target = "cuda" # enable ait: "cuda_with_ait"

# byteir runtime backend function for torch
class ByteIRFunction:
    def __init__(self, module_path, output_shapes, output_dtypes):
        self._session = brt.Session(
            alloc_func=caching_allocator_alloc,
            free_func=caching_allocator_delete)
        self._session.load(module_path)
        self._output_shapes = output_shapes
        self._output_dtypes = output_dtypes
        self._req = self._session.new_request_context(
            torch.cuda.current_stream()._as_parameter_.value)

    def __call__(self, *inputs):
        if TRACE:
            for i in inputs:
                print(f'In ByteIRFunction, Inputs: {i}')

        device = inputs[0].device
        rets = [torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in zip(self._output_shapes, self._output_dtypes)]
        for offset, arg in zip(self._session.get_input_arg_offsets(), inputs):
            assert list(self._session.get_static_shape(offset)) == list(arg.shape)
        for offset, ret in zip(self._session.get_output_arg_offsets(), rets):
            assert list(self._session.get_static_shape(offset)) == list(ret.shape)

        for i, tensor in zip(self._session.get_input_arg_offsets(), inputs):
            self._req.bind_arg(i, tensor.data_ptr())
        for i, tensor in zip(self._session.get_output_arg_offsets(), rets):
            self._req.bind_arg(i, tensor.data_ptr())

        self._req.finish_io_binding()
        self._req.run()
        self._req.sync()

        if TRACE:
            for r in rets:
                print(f'In ByteIRFunction, Outputs: {r}')

        return rets

def unsafe_index_put_pattern(self, indices, values, accumulate):
    return torch.ops.aten._unsafe_index_put(self, indices, values, accumulate)

def unsafe_index_put_replacement(self, indices, values, accumulate):
    return  torch.ops.aten.index_put_.hacked_twin(self, indices, values, accumulate)

def byteir_compile_fx_inner(graph: torch.fx.GraphModule, inputs, is_backward, ban_lst=[]):
    category = 'backward' if is_backward else 'forward'

    print("\n\n============")
    print(f"{category} Part")
    print("============\n\n")
    # print(graph.code)
    fx_graph = preprocess_fx_graph(graph)
    torch.fx.replace_pattern(fx_graph, unsafe_index_put_pattern, unsafe_index_put_replacement)

    backend_legal_ops = [
        "aten._softmax",
        "aten.softmax.int",
        "aten.log_softmax.int",
        "aten._log_softmax",
        # "aten.native_layer_norm",
        # "aten.layer_norm",
        "aten.gelu",
        "aten.argmax",
        "aten.max.dim",
        "aten.one_hot",
        "aten.topk",
    ]
    with maybe_disable_fake_tensor_mode():
        compiled_graph = compile(fx_graph, inputs, 'mhlo', backend_legal_ops=backend_legal_ops)

    global cnt
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER + f"/{model_name}_{category}", exist_ok=True)
    mlir_file_name = f'{TEMP_FOLDER}/{model_name}_{category}_{cnt}.{compile_type}.mlir'
    output_mlir_file_name = f'{TEMP_FOLDER}/{model_name}_{category}/{model_name}_{category}_{cnt}.rt.mlir'
    cnt = cnt + 1
    with open(mlir_file_name, "w+") as fout:
        compiled_graph.operation.print(file=fout,
                                       large_elements_limit=None)

    with maybe_disable_fake_tensor_mode():
        byteir.compile(mlir_file_name, output_mlir_file_name, entry_func='forward', target=byteir_target)

    with torch.inference_mode():
        outputs = graph(*inputs)  # This is problematic. Must convert to use fake tensors.
    mhlo_ret_dtypes = [t.dtype for t in outputs]
    mhlo_ret_shapes = [t.shape for t in outputs]

    print(output_mlir_file_name)
    return ByteIRFunction(output_mlir_file_name, mhlo_ret_shapes, mhlo_ret_dtypes)

def byteir_compile_fx(model: torch.fx.GraphModule, inputs):
    decompose_list = list_decomposed_ops()
    decompositions = get_decompositions(decompose_list)
    module = aot_module(model, fw_compiler=functools.partial(byteir_compile_fx_inner, is_backward=False),
                bw_compiler=functools.partial(byteir_compile_fx_inner, is_backward=True), 
                decompositions=decompositions)
    return module
