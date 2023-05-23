import byteir
from byteir import ir
from byteir.passmanager import PassManager
from byteir.dialects.mhlo import register_mhlo_dialect
import os

def _print_verbose(module, pipeline_msg: str):
    print(pipeline_msg)
    print(module.operation.get_asm(large_elements_limit=10))
    print()

def compile_cuda(
    input: str,
    output: str,
    entry_func: str = "main",
    verbose: bool = False,
    **kwargs,
):
    target = "cuda"
    output_file_dir = os.path.dirname(output)
    output_file_name = os.path.basename(output)

    context = ir.Context()
    register_mhlo_dialect(context)
    context.allow_unregistered_dialects = True
    with open(input, "r") as f:
        module = ir.Module.parse(f.read(), context)
        if verbose:
            _print_verbose(module, "// Input mlir:")

    entry_func_str = "entry-func={}".format(entry_func)
    target_str = "target={}".format(target)
    with context:
        PassManager().parse("builtin.module(hlo-opt{outline-single-elemwise-op})").run(module.operation)
    if verbose:
        _print_verbose(module, "// After Hlo Opt:")
    with context:
        PassManager.parse("builtin.module(linalg-tensor-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// After Linalg Tensor Opt:")
    with context:
        PassManager.parse("builtin.module(byteir-bufferize-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// After ByteIR Bufferize Opt:")
    with context:
        PassManager.parse("builtin.module(affine-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// After Affine Opt:")
    with context:
        PassManager.parse("builtin.module(gpu-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// After GPU Opt:")
    with context:
        PassManager.parse("builtin.module(func.func(remove-func-body{anchor-attr=__byteir_elementwise_fusion__}))").run(module.operation)
        PassManager.parse("builtin.module(func.func(set-op-space{" + entry_func_str + " space={}".format(target) +  "}))").run(module.operation)
        PassManager.parse("builtin.module(set-arg-space{" + entry_func_str + " all-space={}".format(target) + "})").run(module.operation)
    if verbose:
        _print_verbose(module, "// After Set Space Opt:")
    with context:
        PassManager.parse("builtin.module(byre-opt{append-arg-types " + entry_func_str + "})").run(module.operation)
    if verbose:
        _print_verbose(module, "// After Byre Opt:")

    # create device context and module
    module_str = module.operation.get_asm(print_generic_op_form=True)
    with ir.Context() as new_context:
        register_mhlo_dialect(new_context)
        new_context.allow_unregistered_dialects = True
        device_module = ir.Module.parse(module_str, new_context)
        PassManager.parse("builtin.module(nvvm-codegen)").run(device_module.operation)
        if verbose:
            _print_verbose(device_module, "// After NVVM Codegen:")
        # write to output device ptx
        byteir.register_translation_dialects(new_context)
        byteir.translate_to_ptx(device_module.operation, output_file_dir + "/" + output_file_name)

    with context:
        PassManager.parse("builtin.module(byre-host{device-file-name=" + output_file_name + ".ptx" + " " + target_str + " " + entry_func_str + "})").run(module.operation)
    if verbose:
        _print_verbose(module, "// After Byre Host:")
    # write to output host mlir
    with open(output, "w") as f:
        f.write(module.operation.get_asm())


def compile(
    input: str,
    output: str,
    entry_func: str = "main",
    target: str = "cuda",
    verbose: bool = False,
    **kwargs,
):
    if target == "cuda":
        compile_cuda(input, output, entry_func, verbose)
    else:
        raise NotImplemented("not implemented target: {}".format(target))
