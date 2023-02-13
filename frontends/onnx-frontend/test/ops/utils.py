import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


def build_onnx(node_name, input_shape_dtype, output_shape_dtype, initializer=[], **kwargs):
    # Create inputs (ValueInfoProto)
    input_infos = [
        helper.make_tensor_value_info(name, NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(shape))
        for name, shape, dtype in input_shape_dtype
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
    op.version = 13

    # Create the model (ModelProto)
    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", opset_imports=[op])

    onnx.checker.check_model(model_def)
    return model_def
