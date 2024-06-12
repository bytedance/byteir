import argparse
import numpy as np

from byteir import ir
from byteir.dialects.func import FuncOp
from byteir.dialects.cat import IRProcessor
from byteir.utils import mlir_type_to_np_dtype

parser = argparse.ArgumentParser()
parser.add_argument("input_mlir_path", type=str, help="path to mhlo ir")
parser.add_argument(
    "--preprocess", action="store_true", help="whether preprocess mhlo ir"
)
parser.add_argument(
    "--mode", type=str, default="numerical", choices=["numerical", "profile"], help="execution mode"
)
parser.add_argument(
    "--workdir", type=str, default="./workspace", help="workspace directory"
)
parser.add_argument(
    "--backend", type=str, default="ait", choices=["ait"], help="runtime backend"
)
parser.add_argument(
    "-v", "--verbose", action="store_true"
)
args = parser.parse_args()

def generate_inputs(entry_func: FuncOp):
    ret = []
    for arg in entry_func.arguments:
        shaped_type = ir.ShapedType(arg.type)
        shape = shaped_type.shape
        dtype = mlir_type_to_np_dtype(shaped_type.element_type)
        if dtype == np.bool_:
            ret.append(np.random.randint(2, size=shape).astype(dtype))
        elif dtype in [np.uint8, np.int8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            ret.append(np.random.randint(50, size=shape).astype(dtype))
        else:
            ret.append(np.random.random(size=shape).astype(dtype))
        # ret.append(np.ones(shape=shape, dtype=dtype))
    return ret


if __name__ == "__main__":
    processor = IRProcessor("model", args.workdir, verbose=args.verbose)
    with ir.Context() as context:
        processor.load_from_file(args.input_mlir_path)
    func = processor.module.body.operations[0]
    func_name = func.name.value
    inputs = generate_inputs(func)

    # fuse mhlo to cat
    if args.preprocess:
        processor.preprocess_pass()
    processor.cat_opt_pass(anchor_only=True)

    if args.mode == "numerical":
        # run cat on arg.backend
        outputs = processor.execute(inputs, backend=args.backend)

        # run golden
        from mhlo_tools.ir_executor import Interpreter
        interp = Interpreter.load_from_file(args.input_mlir_path)
        golden_outputs = interp.call_function(func_name, inputs)

        # compare outputs
        for golden_output, output in zip(golden_outputs, outputs):
            # np.testing.assert_almost_equal(golden_output, output.detach().cpu().numpy(), decimal=4)
            assert(np.allclose(golden_output, output.detach().cpu().numpy(), rtol=0.05, atol=0.05))
        print(f"cat {args.backend} numerical test pass")
    elif args.mode == "profile":
        processor.benchmark(backend=args.backend, num_trials=10)
        print(f"cat {args.backend} profile finish")
    else:
        raise NotImplemented(f"unimplemented mode {args.mode}")
