import numpy as np
import pytest
import onnx
from test.base import TestBase
from test.ops.utils import build_onnx


class TestOpsTensor(TestBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        self.setup_base(tmpdir_factory, "test/ops/data/tensor")

    def test_concat(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
            ["input_1", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="concat.onnx",
                 input_shape_dtype=input_shape_dtype)

    def test_depth_to_space(self):
        input_shape_dtype = [
            ["X", (10, 16, 20, 20), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (10, 4, 40, 40), "float32"],
        ]
        proto = build_onnx("DepthToSpace", input_shape_dtype, output_shape_dtype, blocksize=2)
        self.run(model_filename="depth_to_space.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_shape(self):
        input_shape_dtype = [
            ["X", (3, 2, 4, 5), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (4,), "int64"],
        ]
        proto = build_onnx("Shape", input_shape_dtype, output_shape_dtype)
        self.run(model_filename="shape.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_gather(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
            ["Y", (2, 2), "int64"],
        ]
        output_shape_dtype = [
            ["Z", (2, 2, 2), "float32"],
        ]
        proto = build_onnx("Gather", input_shape_dtype, output_shape_dtype, axis=0)

        np.random.seed(0)
        input_data = {
            "X": np.random.rand(3, 2).astype(np.float32),
            "Y": np.array([[0, 1], [1, 2]], dtype=np.int64),
        }
        self.run(model_filename="gather.onnx", model_onnx_pb=proto, input_data=input_data)

    def test_gather_elements(self):
        input_shape_dtype = [
            ["X", (3, 2), "float32"],
            ["Y", (2, 2), "int64"],
        ]
        output_shape_dtype = [
            ["Z", (2, 2), "float32"],
        ]
        proto = build_onnx("GatherElements", input_shape_dtype, output_shape_dtype, axis=0)

        np.random.seed(0)
        input_data = {
            "X": np.random.rand(3, 2).astype(np.float32),
            "Y": np.array([[0, 1], [1, 2]], dtype=np.int64),
        }
        self.run(model_filename="gather_elments.onnx", model_onnx_pb=proto, input_data=input_data)

    def test_scatternd(self):
        input_shape_dtype = [
            ["data", (4, 4, 4), "float32"],
            ["indices", (2, 1), "int64"],
            ["updates", (2, 4, 4), "float32"],
        ]
        output_shape_dtype = [
            ["output", (4, 4, 4), "float32"],
        ]
        indices_tensor = onnx.helper.make_tensor(
            "indices", onnx.TensorProto.INT64, [2, 1], np.array([[0], [2]]))
        proto = build_onnx(
            "ScatterND", input_shape_dtype, output_shape_dtype,
            initializer=[indices_tensor]
        )
        input_shape_dtype = [input_shape_dtype[0], input_shape_dtype[2]]
        self.run(model_filename="scatternd.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_split(self):
        input_shape_dtype = [
            ["X", (12, 2), "float32"],
        ]
        output_shape_dtype = [
            ["output_0", (4, 2), "float32"],
            ["output_1", (4, 2), "float32"],
            ["output_2", (4, 2), "float32"],
        ]
        proto = build_onnx("Split", input_shape_dtype, output_shape_dtype, axis=0)
        self.run(model_filename="split.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_transpose(self):
        input_shape_dtype = [
            ["X", (5, 5, 1, 32), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (5, 1, 32, 5), "float32"],
        ]
        perm = [0, 2, 3, 1]
        proto = build_onnx("Transpose", input_shape_dtype, output_shape_dtype, perm=perm)
        self.run(model_filename="transpose.onnx",
                 model_onnx_pb=proto,
                 input_shape_dtype=input_shape_dtype)

    def test_unsqueeze(self):
        input_shape_dtype = [
            ["X", (20, 10, 5), "float32"],
            ["axes", None, None],
        ]
        output_shape_dtype = [
            ["Y", (20, 1, 10, 5, 1), "float32"],
        ]
        axes_tensor = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [2], np.array([1, 4]))
        proto = build_onnx("Unsqueeze", input_shape_dtype, output_shape_dtype, initializer=[axes_tensor])

        input_shape_dtype.pop()
        self.run(model_filename="unsqueeze.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_concat_dynamic_shape(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
            ["input_1", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="concat_dynamic_shape.onnx", input_shape_dtype=input_shape_dtype)

    def test_arg_max(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="arg_max.onnx", input_shape_dtype=input_shape_dtype)

    def test_arg_min(self):
        input_shape_dtype = [
            ["input_0", (1, 5, 5, 3), "float32"],
        ]
        self.run(model_filename="arg_min.onnx", input_shape_dtype=input_shape_dtype)

    def test_onehot(self):
        input_shape_dtype = [
            ["X", (2, 3, 4), "int64"],
            ["depth", (1,), "int64"],
            ["values", (2,), "float32"],
        ]
        output_shape_dtype = [
            ["Y", (2, 3, 4, 5), "float32"],
        ]
        depth_tensor = onnx.helper.make_tensor(
            "depth", onnx.TensorProto.INT64, [1], np.array([5]))
        values_tensor = onnx.helper.make_tensor(
            "values", onnx.TensorProto.FLOAT, [2], np.array([0.0, 1.0]))
        proto = build_onnx(
            "OneHot", input_shape_dtype, output_shape_dtype,
            initializer=[depth_tensor, values_tensor], axis=-1
        )
        input_shape_dtype = [input_shape_dtype[0]]
        self.run(model_filename="onehot.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_pad(self):
        input_shape_dtype = [
            ["X", (1, 3, 5, 5), "float32"],
            ["pads", None, None],
            ["constant_value", None, None],
        ]
        output_shape_dtype = [
            ["Y", (1, 3, 7, 7), "float32"],
        ]
        pads_tensor = onnx.helper.make_tensor(
            "pads", onnx.TensorProto.INT64, [8], np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        constant_value_tensor = onnx.helper.make_tensor(
            "constant_value", onnx.TensorProto.FLOAT, [], np.array([2.0]))
        proto = build_onnx(
            "Pad", input_shape_dtype, output_shape_dtype,
            initializer=[pads_tensor, constant_value_tensor], mode="constant"
        )
        input_shape_dtype = [input_shape_dtype[0]]
        self.run(model_filename="pad.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_resize_nearest_v10(self):
        input_shape_dtype = [
            ["725", (1, 384, 20, 20), "float32"],
        ]
        self.run(model_filename="resize_nearest_v10.onnx", input_shape_dtype=input_shape_dtype)

    def test_resize_nearest(self):
        input_shape_dtype = [
            ["X", (1, 3, 5, 5), "float32"],
            ["", None, None],
            ["", None, None],
            ["sizes", None, None],
        ]
        output_shape_dtype = [
            ["Y", (1, 3, 7, 7), "float32"],
        ]
        sizes_tensor = onnx.helper.make_tensor(
            "sizes", onnx.TensorProto.INT64, [4], np.array([1, 3, 7, 7]))
        proto = build_onnx(
            "Resize", input_shape_dtype, output_shape_dtype,
            initializer=[sizes_tensor],
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
        )
        input_shape_dtype = [input_shape_dtype[0]]
        self.run(model_filename="resize_nearest.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_resize_linear_align_corners(self):
        input_shape_dtype = [
            ["X", (1, 3, 5, 5), "float32"],
            ["", None, None],
            ["", None, None],
            ["sizes", None, None],
        ]
        output_shape_dtype = [
            ["Y", (1, 3, 10, 10), "float32"],
        ]
        sizes_tensor = onnx.helper.make_tensor(
            "sizes", onnx.TensorProto.INT64, [4], np.array([1, 3, 10, 10]))
        proto = build_onnx(
            "Resize", input_shape_dtype, output_shape_dtype,
            initializer=[sizes_tensor],
            coordinate_transformation_mode="align_corners",
            mode="linear",
        )
        input_shape_dtype = [input_shape_dtype[0]]
        self.run(model_filename="resize_linear_align_corners.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_constant_of_shape_1(self):
        input_shape_dtype = [
            ["shape", None, None],
        ]
        output_shape_dtype = [
            ["output", (1, 3, 5, 5), "float32"],
        ]
        shape_tensor = onnx.helper.make_tensor(
            "shape", onnx.TensorProto.INT64, [4], np.array([1, 3, 5, 5]))
        proto = build_onnx(
            "ConstantOfShape", input_shape_dtype, output_shape_dtype,
            initializer=[shape_tensor],
        )
        input_shape_dtype = []
        self.run(model_filename="constant_of_shape_1.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_constant_of_shape_2(self):
        input_shape_dtype = [
            ["shape", None, None],
        ]
        output_shape_dtype = [
            ["output", (1, 3, 5, 5), "int32"],
        ]
        shape_tensor = onnx.helper.make_tensor(
            "shape", onnx.TensorProto.INT64, [4], np.array([1, 3, 5, 5]))
        value_tensor = onnx.helper.make_tensor(
            "value", onnx.TensorProto.INT32, [1], [10]
        )
        proto = build_onnx(
            "ConstantOfShape", input_shape_dtype, output_shape_dtype,
            initializer=[shape_tensor],
            value=value_tensor
        )
        input_shape_dtype = []
        self.run(model_filename="constant_of_shape_2.onnx", model_onnx_pb=proto, input_shape_dtype=input_shape_dtype)

    def test_scatter_elements_1d_basic(self):
        """Test basic ScatterElements operation with 1D tensors."""
        input_shape_dtype = [
            ["data", (8,), "float32"],
            ["indices", (4,), "int64"],
            ["updates", (4,), "float32"],
        ]
        output_shape_dtype = [
            ["output", (8,), "float32"],
        ]
        proto = build_onnx("ScatterElements", input_shape_dtype, output_shape_dtype, axis=0)

        np.random.seed(0)
        input_data = {
            "data": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            "indices": np.array([4, 3, 1, 7], dtype=np.int64),
            "updates": np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
        }
        self.run(model_filename="scatter_elements_1d_basic.onnx",
                 model_onnx_pb=proto,
                 input_data=input_data)

    def test_scatter_elements_2d_axis0(self):
        """Test ScatterElements with 2D tensors along axis 0."""
        input_shape_dtype = [
            ["data", (3, 3), "float32"],
            ["indices", (2, 3), "int64"],
            ["updates", (2, 3), "float32"],
        ]
        output_shape_dtype = [
            ["output", (3, 3), "float32"],
        ]
        proto = build_onnx("ScatterElements", input_shape_dtype, output_shape_dtype, axis=0)

        np.random.seed(0)
        input_data = {
            "data": np.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]], dtype=np.float32),
            "indices": np.array([[1, 0, 2],
                                [0, 2, 1]], dtype=np.int64),
            "updates": np.array([[1.1, 2.1, 3.1],
                                [4.1, 5.1, 6.1]], dtype=np.float32),
        }
        self.run(model_filename="scatter_elements_2d_axis0.onnx",
                 model_onnx_pb=proto,
                 input_data=input_data)

    def test_scatter_elements_2d_axis1(self):
        """Test ScatterElements with 2D tensors along axis 1."""
        input_shape_dtype = [
            ["data", (3, 3), "float32"],
            ["indices", (3, 2), "int64"],
            ["updates", (3, 2), "float32"],
        ]
        output_shape_dtype = [
            ["output", (3, 3), "float32"],
        ]
        proto = build_onnx("ScatterElements", input_shape_dtype, output_shape_dtype, axis=1)

        np.random.seed(0)
        input_data = {
            "data": np.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]], dtype=np.float32),
            "indices": np.array([[1, 2],
                                [0, 1],
                                [2, 0]], dtype=np.int64),
            "updates": np.array([[1.1, 2.1],
                                [3.1, 4.1],
                                [5.1, 6.1]], dtype=np.float32),
        }
        self.run(model_filename="scatter_elements_2d_axis1.onnx",
                 model_onnx_pb=proto,
                 input_data=input_data)

    def test_scatter_elements_3d(self):
        """Test ScatterElements with 3D tensors."""
        input_shape_dtype = [
            ["data", (2, 3, 3), "float32"],
            ["indices", (2, 1, 3), "int64"],
            ["updates", (2, 1, 3), "float32"],
        ]
        output_shape_dtype = [
            ["output", (2, 3, 3), "float32"],
        ]
        proto = build_onnx("ScatterElements", input_shape_dtype, output_shape_dtype, axis=1)

        np.random.seed(0)
        input_data = {
            "data": np.random.rand(2, 3, 3).astype(np.float32),
            "indices": np.array([[[1, 0, 2]], [[2, 1, 0]]], dtype=np.int64),
            "updates": np.random.rand(2, 1, 3).astype(np.float32),
        }
        self.run(model_filename="scatter_elements_3d.onnx",
                 model_onnx_pb=proto,
                 input_data=input_data)

    def test_scatter_elements_negative_indices(self):
        """Test ScatterElements with negative indices."""
        input_shape_dtype = [
            ["data", (5,), "float32"],
            ["indices", (3,), "int64"],
            ["updates", (3,), "float32"],
        ]
        output_shape_dtype = [
            ["output", (5,), "float32"],
        ]
        proto = build_onnx("ScatterElements", input_shape_dtype, output_shape_dtype, axis=0)

        np.random.seed(0)
        input_data = {
            "data": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            "indices": np.array([-1, -3, 1], dtype=np.int64),  # -1 -> 4, -3 -> 2
            "updates": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        }
        self.run(model_filename="scatter_elements_negative_indices.onnx",
                 model_onnx_pb=proto,
                 input_data=input_data)

    def test_scatter_elements_reduction_add(self):
        """Test ScatterElements with reduction='add'."""
        input_shape_dtype = [
            ["data", (5,), "float32"],
            ["indices", (5,), "int64"],
            ["updates", (5,), "float32"],
        ]
        output_shape_dtype = [
            ["output", (5,), "float32"],
        ]
        proto = build_onnx("ScatterElements", input_shape_dtype, output_shape_dtype,
                          axis=0, reduction="add")

        np.random.seed(0)
        input_data = {
            "data": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            "indices": np.array([1, 1, 3, 2, 3], dtype=np.int64),  # Duplicates at 1 and 3
            "updates": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        }
        self.run(model_filename="scatter_elements_reduction_add.onnx",
                 model_onnx_pb=proto,
                 input_data=input_data)

    def test_scatter_elements_reduction_mul(self):
        """Test ScatterElements with reduction='mul'."""
        input_shape_dtype = [
            ["data", (5,), "float32"],
            ["indices", (3,), "int64"],
            ["updates", (3,), "float32"],
        ]
        output_shape_dtype = [
            ["output", (5,), "float32"],
        ]
        proto = build_onnx("ScatterElements", input_shape_dtype, output_shape_dtype,
                          axis=0, reduction="mul")

        np.random.seed(0)
        input_data = {
            "data": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            "indices": np.array([0, 2, 2], dtype=np.int64),  # Duplicate at index 2
            "updates": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        }
        self.run(model_filename="scatter_elements_reduction_mul.onnx",
                 model_onnx_pb=proto,
                 input_data=input_data)
