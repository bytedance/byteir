import argparse
import sys
import torch_frontend
import torch
from typing import List

def dtype_str_to_torch_dtype(dtype: str):
  _map = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float,
    "f64": torch.float64,
    "i1": torch.bool,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "ui8": torch.uint8,
  }
  ret = _map.get(dtype, None)
  assert ret != None, f"unspported dtype str: {dtype}"
  return ret


# Ex1: --input_name_and_shapes name0:2,3 name1:-1,4
#   return (["name0", "name1"], [(2, 3), (-1, 4)])
# Ex2: --input_name_and_shapes 2,3 3,4
#   return ([], [(2, 3), (3, 4)])
# Ex3: --input_name_and_shapes s st (`s` means scalar, `st` means scalar tensor)
#   return ([], [None, ()])
def parse_input_name_and_shapes(input_name_and_shapes: List[str]):
  names = []
  shapes = []
  for name_and_shape in input_name_and_shapes:
    if ':' in name_and_shape:
      name, shape = name_and_shape.split(':')
      names.append(name)
    else:
      shape = name_and_shape

    if shape == "s":
      shape = None
    elif shape == "st":
      shape = ()
    else:
      shape = tuple([int(i) for i in shape.split(',')])
    shapes.append(shape)
  assert len(names) == 0 or len(names) == len(shapes)
  return (names, shapes)


# Ex1: --input_name_and_dtypes name0:f32 name1:f16
#   return (["name0", "name1"], ["f32", "f16"])
# Ex2: --input_name_and_dtypes f32 f16
#   return ([], ["f32", "f16"])
def parse_input_name_and_dtypes(input_name_and_dtypes: List[str]):
  names = []
  dtypes = []
  for name_and_dtype in input_name_and_dtypes:
    if ':' in name_and_dtype:
      name, dtype = name_and_dtype.split(':')
      names.append(name)
    else:
      dtype = name_and_dtype
    dtypes.append(dtype)
  assert len(names) == 0 or len(names) == len(dtypes)
  return (names, dtypes)


def compile_torchscript(args):
    from torch_mlir.torchscript import TensorPlaceholder
    assert args.input_name_and_shapes is not None and args.input_name_and_dtypes is not None
    assert len(args.input_name_and_shapes) == len(args.input_name_and_dtypes)
    _, shapes = parse_input_name_and_shapes(args.input_name_and_shapes)
    _, dtypes = parse_input_name_and_dtypes(args.input_name_and_dtypes)
    assert len(shapes) == len(dtypes)
    
    sample_inputs_placeholder = []
    for shape, dtype in zip(shapes, dtypes):
        sample_inputs_placeholder.append(TensorPlaceholder(shape, dtype_str_to_torch_dtype(dtype)))

    ts_model = torch.jit.load(args.model_path, map_location="cpu")
    if args.enable_jit_rewrite:
      torch_frontend.utils.replace_copy_fill_with_slice_scatter(ts_model.graph)

    module = torch_frontend.compile(ts_model, sample_inputs_placeholder, args.output_type, verbose=args.verbose, debug=torch_frontend.DebugType(1))
    if len(args.output_file_path) != 0:
      with open(args.output_file_path, "w") as f:
          if args.elide:
              print(module.operation.get_asm(large_elements_limit=10), file=f)
          else:
              print(module.operation.get_asm(), file=f)
    else:
      if args.elide:
        print(module.operation.get_asm(large_elements_limit=10), file=sys.stdout)
      else:
        print(module.operation.get_asm(), file=f)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("model_path")
    parser.add_argument("-o", "--output_file_path", type=str, default="", help="output file path")
    parser.add_argument("--model_type", type=str, default="torchscript", choices=["torchscript"])
    parser.add_argument("--output_type", type=str, default="stablehlo", choices=["raw", "torch", "stablehlo"])
    parser.add_argument("--elide", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--enable_jit_rewrite", default=False, action="store_true")
    parser.add_argument(
        "--input_name_and_shapes",
        nargs="+",
        help="""Specify some input's shapes, -1 means ?, s means scalar (like i64), st means scalar tensor (like tensor<i64>).
                Ex1: --input_name_and_shapes name0:2,3 name1:-1,4.
                Ex2: --input_name_and_shapes 2,3 -1,4 s st.""",
    )
    parser.add_argument(
        "--input_name_and_dtypes",
        nargs="+",
        choices=["bf16", "f16", "f32", "f64", "i1", "i8", "i16", "i32", "i64", "ui8", "ui16", "ui32", "ui64", "str"],
        help="""Specify some input's dtypes. Only available for TorchScript models.
                Ex1: --input_name_and_dtypes name0:f32 name1:f16.
                Ex2: --input_name_and_dtypes f32 f16.""",
    )

    args = parser.parse_args()
    if args.model_type == "torchscript":
        compile_torchscript(args)


if __name__=="__main__":
    main()
