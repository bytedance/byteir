import os
import copy
import torch
import sys
import functools
from typing import List

from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

if 'BYTEIR_FRONTEND_PATH' in os.environ:
    sys.path.append(os.getenv('BYTEIR_FRONTEND_PATH'))

if 'BYTEIR_RUNTIME_PATH' in os.environ:
    sys.path.append(os.getenv('BYTEIR_RUNTIME_PATH'))

import brt
import byteir

from torch_frontend import compile
from torch_frontend import list_decomposed_ops, preprocess_fx_graph, fx_replace_attn_pattern, replace_flash_attn 

from functorch.compile import aot_module
from torch._decomp import get_decompositions

from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete

class ByteIRFunction:
    def __init__(self, module_path, output_shapes, output_dtypes, none_indices):
        self._session = brt.Session(
            alloc_func=caching_allocator_alloc,
            free_func=caching_allocator_delete)
        self._session.load(module_path)
        self._output_shapes = output_shapes
        self._output_dtypes = output_dtypes
        self._req = self._session.new_request_context(
            torch.cuda.current_stream()._as_parameter_.value)
        self._none_indices = none_indices

    def __call__(self, *inputs):
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

        # add None results to return values
        results = []
        none_ptr = 0
        ret_ptr = 0
        for i in range(len(rets) + len(self._none_indices)):
            if none_ptr < len(self._none_indices) and i == self._none_indices[none_ptr]:
                results.append(None)
                none_ptr += 1
            else:
                results.append(rets[ret_ptr])
                ret_ptr += 1
        return results

def get_none_indices(fx_g: torch.fx.GraphModule) -> List[int]:
    none_indices = []
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, (list, tuple)):
                node_arg = list(node_arg)
                node_args_len = len(node_arg)
                for i in range(node_args_len):
                    if node_arg[i] is None:
                        none_indices.append(i)
                break

    return none_indices

def byteir_compile_fx_inner(graph: torch.fx.GraphModule, inputs, is_backward, ban_lst=[]):
    category = 'backward' if is_backward else 'forward'

    print("\n\n============")
    print(f"{category} Part")
    print("============\n\n")
    # print(graph.code)
    none_indices = get_none_indices(graph)
    fx_graph = preprocess_fx_graph(graph)
    # print(fx_graph.code)
    # compiled_graph = convert_to_mhlo_via_torch_mlir(fx_graph, inputs, use_tracing=False)

    compile_type = 'mhlo'
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
        "byteir.flash_attn_fwd",
        "byteir.flash_attn_bwd",
    ]
    with maybe_disable_fake_tensor_mode():
        compiled_graph = compile(fx_graph, inputs, compile_type, backend_legal_ops=backend_legal_ops)
    # print(compiled_graph)

    model_name = "test"
    TEMP_FOLDER="./temp"
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER + f"/{model_name}_{category}", exist_ok=True)
    mlir_file_name = f'{TEMP_FOLDER}/{model_name}_{category}.{compile_type}.mlir'
    output_mlir_file_name = f'{TEMP_FOLDER}/{model_name}_{category}/{model_name}_{category}.rt.mlir'
    with open(mlir_file_name, "w+") as fout:
        compiled_graph.operation.print(file=fout,
                                       large_elements_limit=None)

    with maybe_disable_fake_tensor_mode():
        byteir.compile(mlir_file_name, output_mlir_file_name, entry_func='forward', target='cuda_with_ait')

    outputs = FakeTensorProp(graph).propagate(*inputs)
    mhlo_ret_dtypes = [t.dtype for t in outputs]
    mhlo_ret_shapes = [t.shape for t in outputs]

    print(output_mlir_file_name)
    return ByteIRFunction(output_mlir_file_name, mhlo_ret_shapes, mhlo_ret_dtypes, none_indices)


from torch._inductor.virtualized import V
from torch._dynamo.utils import detect_fake_mode
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.fx_passes.joint_graph import joint_graph_passes
from functorch.compile import min_cut_rematerialization_partition

def byteir_compile_fx(model_: torch.fx.GraphModule, example_inputs_):
    # TODO: can add logging before/after the call to create_aot_dispatcher_function
    # in torch._functorch/aot_autograd.py::aot_module_simplified::aot_function_simplified::new_func
    # once torchdynamo is merged into pytorch
    fake_mode = detect_fake_mode(example_inputs_) or torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True
    )
    tracing_context = (
        torch._guards.TracingContext.get() or torch._guards.TracingContext(fake_mode)
    )
    decompose_list = list_decomposed_ops()
    decompositions = get_decompositions(decompose_list)
    # preprocess flash attention
    # replace attention pattern to scaled_dot_product_attention
    model_ = fx_replace_attn_pattern(model_)
    # replace scaled_dot_product_attention to byteir.flash_attn
    model_ = replace_flash_attn(model_)
    def partition_fn(graph, joint_inputs, **kwargs):
        joint_graph_passes(graph)
        return min_cut_rematerialization_partition(
            graph, joint_inputs, **kwargs, compiler="inductor"
        )

    with V.set_fake_mode(fake_mode), torch._guards.tracing(tracing_context):
        return aot_autograd(
            fw_compiler=functools.partial(byteir_compile_fx_inner, is_backward=False),
            bw_compiler=functools.partial(byteir_compile_fx_inner, is_backward=True),
            inference_compiler=functools.partial(byteir_compile_fx_inner, is_backward=False),
            decompositions=decompositions,
            partition_fn=partition_fn,
            keep_inference_input_mutations=True,
        )(model_, example_inputs_)
