import byteir
from byteir import ir
from byteir.passmanager import PassManager
from byteir.dialects.cat import IRProcessor
from pathlib import Path
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

    with open(input, "r") as f:
        module = ir.Module.parse(f.read(), context)
        if verbose:
            _print_verbose(module, "// IR Dump Input MLIR:")

    entry_func_str = "entry-func={}".format(entry_func)
    target_str = "target={}".format(target)
    with context:
        PassManager().parse("builtin.module(hlo-opt{outline-single-elemwise-op})").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After Hlo Opt:")
    with context:
        PassManager.parse("builtin.module(linalg-tensor-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After Linalg Tensor Opt:")
    with context:
        PassManager.parse("builtin.module(byre-tensor-opt{{append-arg-types {}}})".format(entry_func_str)).run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After Byre Tensor Opt:")
    with context:
        PassManager.parse("builtin.module(byteir-bufferize-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After ByteIR Bufferize Opt:")
    with context:
        PassManager.parse("builtin.module(linalg-memref-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After Linalg Memref Opt:")
    with context:
        PassManager.parse("builtin.module(affine-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After Affine Opt:")
    with context:
        PassManager.parse("builtin.module(gpu-opt)").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After GPU Opt:")
    with context:
        PassManager.parse("builtin.module(func.func(remove-func-body{anchor-attr=__byteir_elementwise_fusion__}))").run(module.operation)
        PassManager.parse("builtin.module(func.func(set-op-space{" + entry_func_str + " space={}".format(target) +  "}))").run(module.operation)
        PassManager.parse("builtin.module(set-arg-space{" + entry_func_str + " all-space={}".format(target) + "})").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After Set Space Opt:")
    with context:
        PassManager.parse("builtin.module(byre-opt{append-arg-types " + entry_func_str + "})").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After Byre Opt:")

    # create device context and module
    module_str = module.operation.get_asm(print_generic_op_form=True)
    with ir.Context() as new_context:
        device_module = ir.Module.parse(module_str, new_context)
        PassManager.parse("builtin.module(nvvm-codegen)").run(device_module.operation)
        if verbose:
            _print_verbose(device_module, "// IR Dump After NVVM Codegen:")
        # write to output device ptx
        byteir.translate_to_ptx(device_module.operation, output_file_dir + "/" + output_file_name)

    with context:
        PassManager.parse("builtin.module(byre-host{device-file-name=" + output_file_name + ".ptx" + " " + target_str + " " + entry_func_str + "})").run(module.operation)
    if verbose:
        _print_verbose(module, "// IR Dump After Byre Host:")
    # write to output host mlir
    with open(output, "w") as f:
        f.write(module.operation.get_asm())


def compile_cuda_with_ait(
    input: str,
    output: str,
    entry_func: str = "main",
    verbose: bool = False,
    name: str = "model",
    **kwargs,
):
    target = "cuda"
    output_file_dir = os.path.dirname(output)
    output_file_name = os.path.basename(output)

    context = ir.Context()

    entry_func_str = "entry-func={}".format(entry_func)
    target_str = "target={}".format(target)

    processor = IRProcessor(name, "./workspace")
    with context:
        processor.load_from_file(input)
    if verbose:
        _print_verbose(processor.module, "// IR Dump Input MLIR:")
    processor.preprocess_pass()
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Cat Preprocess:")
    with context:
        processor.cat_opt_pass(anchor_only=False)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Cat Opt:")
    # clustering
    with context:
        processor.hlo_opt_pass()
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Hlo Opt:")
    # generate ait .so for subgraphs
    dll_paths = []
    with context:
        _, dll_paths = processor.ait_opt_pass(anchor_only=True)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After AIT Opt:")
    # move .so to target
    target_dir = Path(output)
    for dll_path in dll_paths:
        print("cp {} {}".format(dll_path, target_dir.parent.absolute()))
        os.system("cp {} {}".format(dll_path, target_dir.parent.absolute()))

    with context:
        PassManager.parse("builtin.module(linalg-tensor-opt)").run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Linalg Tensor Opt:")
    with context:
        PassManager.parse("builtin.module(byre-tensor-opt{{append-arg-types {}}})".format(entry_func_str)).run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Linalg Tensor Opt:")
    with context:
        PassManager.parse("builtin.module(byteir-bufferize-opt)").run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After ByteIR Bufferize Opt:")
    with context:
        PassManager.parse("builtin.module(linalg-memref-opt)").run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Linalg Memref Opt:")
    with context:
        PassManager.parse("builtin.module(affine-opt)").run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Affine Opt:")
    with context:
        PassManager.parse("builtin.module(gpu-opt)").run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After GPU Opt:")
    with context:
        PassManager.parse("builtin.module(func.func(remove-func-body{anchor-attr=__byteir_elementwise_fusion__}))").run(processor.module.operation)
        PassManager.parse("builtin.module(func.func(set-op-space{" + entry_func_str + " space={}".format(target) +  "}))").run(processor.module.operation)
        PassManager.parse("builtin.module(set-arg-space{" + entry_func_str + " all-space={}".format(target) + "})").run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Set Space Opt:")
    with context:
        PassManager.parse("builtin.module(byre-opt{append-arg-types " + entry_func_str + "})").run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Byre Opt:")

    # create device context and module
    module_str = processor.module.operation.get_asm(print_generic_op_form=True)
    with ir.Context() as new_context:
        device_module = ir.Module.parse(module_str, new_context)
        PassManager.parse("builtin.module(nvvm-codegen)").run(device_module.operation)
        if verbose:
            _print_verbose(device_module, "// IR Dump After NVVM Codegen:")
        # write to output device ptx
        byteir.translate_to_ptx(device_module.operation, output_file_dir + "/" + output_file_name)

    with context:
        PassManager.parse("builtin.module(byre-host{device-file-name=" + output_file_name + ".ptx" + " " + target_str + " " + entry_func_str + "})").run(processor.module.operation)
    if verbose:
        _print_verbose(processor.module, "// IR Dump After Byre Host:")
    # write to output host mlir
    with open(output, "w") as f:
        f.write(processor.module.operation.get_asm())


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
    elif target == "cuda_with_ait":
        compile_cuda_with_ait(input, output, entry_func, verbose)
    else:
        raise NotImplemented("not implemented target: {}".format(target))
