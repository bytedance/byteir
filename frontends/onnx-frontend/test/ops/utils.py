import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


def build_onnx(node_name, input_shape_dtype, output_shape_dtype, initializer=[], **kwargs):
    # Create inputs (ValueInfoProto)
    input_infos = [
        helper.make_tensor_value_info(name, NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(shape))
        for name, shape, dtype in input_shape_dtype if shape is not None and dtype is not None
    ]
    input_names = [
        name for name, _, _ in input_shape_dtype
    ]
    # Create outputs (ValueInfoProto)
    output_infos = [
        helper.make_tensor_value_info(name, NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(shape))
        for name, shape, dtype in output_shape_dtype
    ]
    output_names = [
        name for name, _, _ in output_shape_dtype
    ]

    # Create a node (NodeProto)
    node_def = helper.make_node(
        node_name,
        input_names,
        output_names,
        **kwargs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        nodes=[node_def],
        name="test-model",
        inputs=input_infos,
        outputs=output_infos,
        initializer=initializer,
    )

    op = onnx.OperatorSetIdProto()
    op.version = 17

    # Create the model (ModelProto)
    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", opset_imports=[op])

    onnx.checker.check_model(model_def)
    return model_def

def build_reduce_sum_axis_one(input_shape_dtype, output_shape_dtype):
    # Create inputs (ValueInfoProto)
    input_infos = [
        helper.make_tensor_value_info(name, NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(shape))
        for name, shape, dtype in input_shape_dtype if shape is not None and dtype is not None
    ]
    input_names = [
        name for name, _, _ in input_shape_dtype
    ]
    # Create outputs (ValueInfoProto)
    output_infos = [
        helper.make_tensor_value_info(name, NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(shape))
        for name, shape, dtype in output_shape_dtype
    ]
    output_names = [
        name for name, _, _ in output_shape_dtype
    ]
    axis_one = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["axis"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[1],
        )
    )
    input_names.append("axis")
    reduce_sum_node = helper.make_node(
        "ReduceSum",
        inputs=input_names,
        outputs=output_names,
    )
    graph_def = helper.make_graph(
        nodes=[axis_one, reduce_sum_node],
        name="test-model",
        inputs=input_infos,
        outputs=output_infos,
        initializer=[],
    )
    op = onnx.OperatorSetIdProto()
    op.version = 17

    # Create the model (ModelProto)
    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", opset_imports=[op])

    onnx.checker.check_model(model_def)
    return model_def