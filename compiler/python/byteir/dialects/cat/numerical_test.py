import argparse
from mhlo_tools.ir_executor import Interpreter
from mhlo_tools.mlir import ir
from mhlo_tools.ir_executor.helper import (
    mlir_type_to_dtype,
    mlir_attr_to_pyobj,
)
import numpy as np

from byteir.dialects.cat import register_cat_dialect, IRProcessor
from byteir.dialects.mhlo import register_mhlo_dialect

parser = argparse.ArgumentParser()
parser.add_argument("--before-pass-file", type=str, required=True, help="path to mhlo ir")
parser.add_argument("--after-pass-file", type=str, default=None, required=False, help="path to cat ir after conversion")
parser.add_argument(
    "--preprocess", action="store_true", help="whether preprocess mhlo ir"
)
parser.add_argument(
    "--backend", type=str, choices=["ait"], default="ait", help="runtime backend"
)

args = parser.parse_args()

def generate_inputs(interp):
    module = interp._mod
    entry_func = module.body.operations[0]
    ret = []
    for arg in entry_func.arguments:
        shaped_type = ir.ShapedType(arg.type)
        shape = shaped_type.shape
        dtype = mlir_type_to_dtype(shaped_type.element_type)
        #ret.append(np.random.random(size=shape).astype(dtype))
        ret.append(np.ones(shape=shape, dtype=dtype))
    return ret


if __name__ == "__main__":
    interp = Interpreter.load_from_file(args.before_pass_file)
    inputs = generate_inputs(interp)
    func_name = "main"

    # run ait
    from byteir import ir
    with ir.Context() as context:
        register_cat_dialect(context)
        register_mhlo_dialect(context)
        context.allow_unregistered_dialects = True

        if args.after_pass_file == None:
            processor = IRProcessor("model", "./workspace")
            processor.load_from_file(args.before_pass_file)
            if args.preprocess:
                processor.preprocess_pass()
            processor.cat_opt_pass(anchor_only=True)
            func_name = processor.module.body.operations[0].name.value
        else:
            processor = IRProcessor("model", "./workspace")
            processor.load_from_file(args.after_pass_file)
            func_name = processor.module.body.operations[0].name.value
        outputs = processor.execute(inputs, backend="ait")

    # run golden
    golden_outputs = interp.call_function(func_name, inputs)

    # compare outputs
    for golden_output, output in zip(golden_outputs, outputs):
        np.testing.assert_almost_equal(golden_output, output.cpu().numpy(), decimal=4)
    print("numerical test pass")
