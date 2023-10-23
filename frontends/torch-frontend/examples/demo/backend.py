import os
import torch
import functools

from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport

import brt
import byteir

import torch_frontend
from torch_frontend import list_decomposed_ops, preprocess_fx_graph, fx_replace_attn_pattern, replace_flash_attn, get_none_indices


TRACE = False

submodule_cnt = 0
MODEL_NAME = ''
FLASH = False


from functorch.compile import aot_module
from torch._decomp import get_decompositions

from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete

class ByteIRInferenceFunction:
    def __init__(self, module_path):
        self._session = brt.Session(alloc_func=caching_allocator_alloc,
                                    free_func=caching_allocator_delete)
        self._session.load(module_path)
        self._req = self._session.new_request_context(
            torch.cuda.current_stream()._as_parameter_.value)

    def __call__(self, *inputs):
        device = inputs[0].device
        from brt.utils import brt_dtype_to_torch_dtype
        results = [torch.empty(self._session.get_static_shape(offset),
                               dtype=brt_dtype_to_torch_dtype(self._session.get_data_type(offset)),
                               device=device) for offset in self._session.get_output_arg_offsets()]
        
        for offset, input in zip(self._session.get_input_arg_offsets(), inputs):
            self._req.bind_arg(offset, input.data_ptr())
        for offset, output in zip(self._session.get_output_arg_offsets(), results):
            self._req.bind_arg(offset, output.data_ptr())
        self._req.finish_io_binding()
        self._req.run()
        self._req.sync()
        return results

class ByteIRFunction:
    def __init__(self, module_path, output_shapes, output_dtypes, none_indices, device):
        self._session = brt.Session(
            alloc_func=caching_allocator_alloc,
            free_func=caching_allocator_delete)
        self._session.load(module_path)
        self._output_shapes = output_shapes
        self._output_dtypes = output_dtypes
        self._req = self._session.new_request_context(
            torch.cuda.current_stream()._as_parameter_.value)
        self._none_indices = none_indices
        self.device = device

    def __call__(self, *inputs):
        if TRACE:
            for i in range(len(inputs)):
                input = inputs[i]
                print("In ByteIRFunction, Inputs["+str(i)+"]", input)

        rets = [torch.empty(shape, dtype=dtype, device=self.device)
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
            for i in range(len(rets)):
                r = rets[i]
                print("In ByteIRFunction, Outputs["+str(i)+"]", r)

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
        if len(results) == 1:
            return results[0]
        return results

class ByteIROperatorSupport(OperatorSupport):
    def __init__(self, fallback_ops):
        super().__init__()
        self._fallback_ops = fallback_ops

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        unsupported_ops = [
            torch.ops.aten.view_as_complex.default,
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten.all.default, # TODO support in torch mlir
        ]
        return node.op in [
          "call_function", "call_module", "call_method"
        ] and node not in self._fallback_ops and node.target not in unsupported_ops

def is_complex_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_complex()

class FallBackNodeCollector(torch.fx.Interpreter):

  def __init__(self, module):
    super().__init__(module)
    self._fallback_ops = []

  def run_node(self, n: torch.fx.Node):
    result = super().run_node(n)
    # fallback if inputs are complex tensors,
    if n.op in ["call_function", "call_module", "call_method"]:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        for arg in args:
            if isinstance(arg, torch.Tensor) and is_complex_tensor(arg):
                self._fallback_ops.append(n)
                break
    return result

  def get_fallback_ops(self):
    return self._fallback_ops


def extract_internal(graph, is_backward):
    def byteir_runner(*inputs):
        category = 'backward' if is_backward else 'forward'

        print("\n\n============")
        print(f"{category} Part")
        print("============\n\n")
        none_indices = get_none_indices(graph)
        fx_graph = preprocess_fx_graph(graph)

        compile_type = 'stablehlo'
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
            compiled_graph = torch_frontend.compile(fx_graph, inputs, compile_type, backend_legal_ops=backend_legal_ops)

        model_name = MODEL_NAME
        global submodule_cnt
        TEMP_FOLDER="./temp"
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        os.makedirs(TEMP_FOLDER + f"/{model_name}_{category}_{submodule_cnt}", exist_ok=True)
        mlir_file_name = f'{TEMP_FOLDER}/{model_name}_{category}_{submodule_cnt}.{compile_type}.mlir'
        output_mlir_file_name = f'{TEMP_FOLDER}/{model_name}_{category}_{submodule_cnt}/{model_name}_{category}_{submodule_cnt}.rt.mlir'
        submodule_cnt = submodule_cnt + 1
        with open(mlir_file_name, "w+") as fout:
            compiled_graph.operation.print(file=fout,
                                        large_elements_limit=None)

        with maybe_disable_fake_tensor_mode():
            byteir.compile(mlir_file_name, output_mlir_file_name, entry_func='forward', target='cuda_with_ait')

        outputs = FakeTensorProp(graph).propagate(*inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        mhlo_ret_dtypes = [t.dtype for t in outputs]
        mhlo_ret_shapes = [t.shape for t in outputs]

        print(output_mlir_file_name)
        runner = ByteIRFunction(output_mlir_file_name, mhlo_ret_shapes, mhlo_ret_dtypes, none_indices, device=outputs[0].device)
        return runner(*inputs)
    return byteir_runner


def partition_graphs(gm, inputs):
    collector = FallBackNodeCollector(gm)
    collector.run(*inputs)
    fallback_ops = collector.get_fallback_ops()
    # print("fallback_ops", fallback_ops)
    supported_ops = ByteIROperatorSupport(fallback_ops)
    partitioner = CapabilityBasedPartitioner(gm,
                                            supported_ops,
                                            allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    print("num graphs:", len(partitions))
    fused_graph = partitioner.fuse_partitions(partitions)
    return fused_graph


def fallback_inner(model_: torch.fx.GraphModule, inputs, is_backward):
    partitioned_graph = partition_graphs(model_, inputs)
    # partitioned_graph.graph.print_tabular()
    # compile each submodule and replace it with a call
    for node in partitioned_graph.graph.nodes:
        if node.op == "call_module" and "fused_" in node.name:
            fused_module = getattr(partitioned_graph, node.name)
            partitioned_graph.delete_submodule(node.target)
            with partitioned_graph.graph.inserting_after(node):
                compile_submodule = extract_internal(fused_module, is_backward)
                new_node = partitioned_graph.graph.call_function(
                    compile_submodule, node.args, None)
                node.replace_all_uses_with(new_node)
            partitioned_graph.graph.erase_node(node)
    partitioned_graph.recompile()
    return partitioned_graph


from torch._inductor.virtualized import V
from torch._dynamo.utils import detect_fake_mode
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.fx_passes.joint_graph import joint_graph_passes


def fuse_aware_byteir_compile_fx(model_: torch.fx.GraphModule, example_inputs_):
    from partitioners import fuse_aware_min_cut_partition
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

    def partition_fn(graph, joint_inputs, **kwargs):
        joint_graph_passes(graph)
        return fuse_aware_min_cut_partition(
            graph, joint_inputs, **kwargs, compiler="inductor"
        )

    if FLASH:
        # preprocess flash attention
        # replace attention pattern to scaled_dot_product_attention
        model_ = fx_replace_attn_pattern(model_)
        # replace scaled_dot_product_attention to byteir.flash_attn
        model_ = replace_flash_attn(model_)

    with V.set_fake_mode(fake_mode), torch._guards.tracing(tracing_context):
        return aot_autograd(
            fw_compiler=functools.partial(fallback_inner, is_backward=False),
            bw_compiler=functools.partial(fallback_inner, is_backward=True),
            inference_compiler=functools.partial(fallback_inner, is_backward=False),
            decompositions=decompositions,
            partition_fn=partition_fn,
            keep_inference_input_mutations=True,
        )(model_, example_inputs_)
