import byteir
from byteir import ir
from byteir.passmanager import PassManager
from byteir.dialects.cat import IRProcessor
from byteir.dialects.builtin import ModuleOp
from byteir.utils import detect_cuda_with_nvidia_smi
from pathlib import Path
import os
from shutil import copymode

def _print_verbose(module: ModuleOp, pipeline_msg: str):
    print(pipeline_msg)
    print(module.operation.get_asm(large_elements_limit=10))
    print()

def _compile_cuda(
    module: ModuleOp,
    output_file_path: str,
    entry_func: str,
    gpu_type: str,
    verbose: bool = False,
    **kwargs,
) -> None:
    target = "cuda"
    output_file_dir = os.path.dirname(output_file_path)
    output_file_dir = "./" if len(output_file_dir) == 0 else output_file_dir
    output_file_name = os.path.basename(output_file_path)
    useBarePtrCallConv = True # all tensor must have static shapes if True

    context = module.context

    entry_func_str = "entry-func={}".format(entry_func)
    target_str = "target={}".format(target)
    with context:
        PassManager().parse("builtin.module(hlo-opt{outline-single-elemwise-op})").run(module.operation)
        _print_verbose(module, "// IR Dump After Hlo Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(linalg-tensor-opt)").run(module.operation)
        _print_verbose(module, "// IR Dump After Linalg Tensor Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(byre-tensor-opt{{append-arg-types {}}})".format(entry_func_str)).run(module.operation)
        _print_verbose(module, "// IR Dump After Byre Tensor Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(byteir-bufferize-opt)").run(module.operation)
        _print_verbose(module, "// IR Dump After ByteIR Bufferize Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(linalg-memref-opt)").run(module.operation)
        _print_verbose(module, "// IR Dump After Linalg Memref Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(scf-opt)").run(module.operation)
        _print_verbose(module, "// IR Dump After SCF Opt:") if verbose else ...
    with context:
        if useBarePtrCallConv:
            PassManager.parse("builtin.module(gpu-opt{use-bare-ptr-memref-call-conv=true})").run(module.operation)
        else:
            PassManager.parse("builtin.module(gpu-opt)").run(module.operation)
        _print_verbose(module, "// IR Dump After GPU Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(func.func(remove-func-body{anchor-attr=__byteir_elementwise_fusion__}))").run(module.operation)
        PassManager.parse("builtin.module(inline)").run(module.operation)
        if useBarePtrCallConv:
            PassManager.parse("builtin.module(func.func(gpu-launch-func-to-byre{use-bare-ptr-memref-call-conv=true}))").run(module.operation)
        else:
            PassManager.parse("builtin.module(func.func(gpu-launch-func-to-byre))").run(module.operation)
        PassManager.parse("builtin.module(func.func(set-op-space{" + entry_func_str + " space={}".format(target) +  "}))").run(module.operation)
        PassManager.parse("builtin.module(set-arg-space{" + entry_func_str + " all-space={}".format(target) + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Set Space Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(byre-opt{append-arg-types " + entry_func_str + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Byre Opt:") if verbose else ...

    # create device module
    module_str = module.operation.get_asm(print_generic_op_form=True)
    device_module = ir.Module.parse(module_str, context)
    with context:
        if useBarePtrCallConv:
            PassManager.parse("builtin.module(nvvm-codegen{use-bare-ptr-memref-call-conv=true})").run(device_module.operation)
        else:
            PassManager.parse("builtin.module(nvvm-codegen)").run(device_module.operation)
        _print_verbose(device_module, "// IR Dump After NVVM Codegen:") if verbose else ...
    # write to output device ptx file
    byteir.translate_to_ptx(device_module, output_file_dir + "/" + output_file_name, gpu_type)

    # create host mlir
    with context:
        PassManager.parse("builtin.module(byre-host{device-file-name=" + output_file_name + ".ptx" + " " + target_str + " " + entry_func_str + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Byre Host:") if verbose else ...
    # write to output host mlir file
    with open(output_file_path, "w") as f:
        f.write(module.operation.get_asm())


def _compile_cuda_with_ait(
    module: ModuleOp,
    output_file_path: str,
    entry_func: str,
    gpu_type: str,
    verbose: bool = False,
    name: str = "model",
    aggressive_mode: bool = False,
    parallelism: int = 1,
    disable_byteir_ait_cache: bool = False,
    **kwargs,
) -> None:
    target = "cuda"
    output_file_dir = os.path.dirname(output_file_path)
    output_file_dir = "./" if len(output_file_dir) == 0 else output_file_dir
    output_file_name = os.path.basename(output_file_path)
    useBarePtrCallConv = True # all tensor must have static shapes if True

    context = module.context

    entry_func_str = "entry-func={}".format(entry_func)
    target_str = "target={}".format(target)

    processor = IRProcessor(name, 
                            "./workspace", 
                            compile_parallelism=parallelism,
                            disable_byteir_ait_cache=disable_byteir_ait_cache,
                            verbose=verbose)
    processor.module = module

    processor.preprocess_pass()
    _print_verbose(processor.module, "// IR Dump After Cat Preprocess:") if verbose else ...
    with context:
        processor.cat_opt_pass(anchor_only=False, aggressive_mode=aggressive_mode)
        _print_verbose(processor.module, "// IR Dump After Cat Opt:") if verbose else ...
    # clustering
    with context:
        processor.hlo_opt_pass(outline_single_elemwise_op=True, aggressive_mode=aggressive_mode)
        _print_verbose(processor.module, "// IR Dump After Hlo Opt:") if verbose else ...
    # generate ait .so for subgraphs
    dll_paths = []
    with context:
        _, dll_paths = processor.ait_opt_pass(anchor_only=True)
        _print_verbose(processor.module, "// IR Dump After AIT Opt:") if verbose else ...
    # move .so to target
    target_dir = Path(output_file_path)
    for dll_path in dll_paths:
        print("cp -p {} {}".format(dll_path, target_dir.parent.absolute()))
        os.system("cp -p {} {}".format(dll_path, target_dir.parent.absolute()))

    with context:
        PassManager.parse("builtin.module(linalg-tensor-opt)").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After Linalg Tensor Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(byre-tensor-opt{{append-arg-types {}}})".format(entry_func_str)).run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After Byre Tensor Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(byteir-bufferize-opt)").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After ByteIR Bufferize Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(linalg-memref-opt)").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After Linalg Memref Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(scf-opt)").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After SCF Opt:") if verbose else ...
    with context:
        if useBarePtrCallConv:
            PassManager.parse("builtin.module(gpu-opt{use-bare-ptr-memref-call-conv=true})").run(processor.module.operation)
        else:
            PassManager.parse("builtin.module(gpu-opt)").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After GPU Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(func.func(remove-func-body{anchor-attr=__byteir_elementwise_fusion__}))").run(processor.module.operation)
        PassManager.parse("builtin.module(inline)").run(processor.module.operation)
        if useBarePtrCallConv:
            PassManager.parse("builtin.module(func.func(gpu-launch-func-to-byre{use-bare-ptr-memref-call-conv=true}))").run(processor.module.operation)
        else:
            PassManager.parse("builtin.module(func.func(gpu-launch-func-to-byre))").run(processor.module.operation)
        PassManager.parse("builtin.module(func.func(set-op-space{" + entry_func_str + " space={}".format(target) +  "}))").run(processor.module.operation)
        PassManager.parse("builtin.module(set-arg-space{" + entry_func_str + " all-space={}".format(target) + "})").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After Set Space Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(byre-opt{append-arg-types " + entry_func_str + "})").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After Byre Opt:") if verbose else ...

    # create device module
    module_str = processor.module.operation.get_asm(print_generic_op_form=True)
    device_module = ir.Module.parse(module_str, context)
    with context:
        if useBarePtrCallConv:
            PassManager.parse("builtin.module(nvvm-codegen{use-bare-ptr-memref-call-conv=true})").run(device_module.operation)
        else:
            PassManager.parse("builtin.module(nvvm-codegen)").run(device_module.operation)
        _print_verbose(device_module, "// IR Dump After NVVM Codegen:") if verbose else ...
    # write to output device ptx
    byteir.translate_to_ptx(device_module, output_file_dir + "/" + output_file_name, gpu_type)

    with context:
        PassManager.parse("builtin.module(byre-host{device-file-name=" + output_file_name + ".ptx" + " " + target_str + " " + entry_func_str + "})").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After Byre Host:") if verbose else ...
    # write to output host mlir
    with open(output_file_path, "w") as f:
        f.write(processor.module.operation.get_asm())


def compile(
    input_file_path: str,
    output_file_path: str,
    entry_func: str = "main",
    target: str = "cuda",
    gpu_type: str = "local",
    verbose: bool = False,
    parallelism: int = 1,
    disable_byteir_ait_cache: bool = False,
    **kwargs,
) -> None:
    ### optional detecting gpu type from nvidia-smi
    if gpu_type == "local":
        local_gpu = detect_cuda_with_nvidia_smi()
        assert local_gpu is not None
        gpu_type = local_gpu
    print(f"Compiling PTX to {gpu_type}")

    ### load from .mlir or .mlirbc
    from byteir._mlir_libs._stablehlo import deserialize_portable_artifact
    context = ir.Context()
    if input_file_path.endswith(".mlirbc"):
        module_bytes = deserialize_portable_artifact(open(input_file_path, "rb").read())
        module = ir.Module.parse(module_bytes, context)
    else:
        module = ir.Module.parse(open(input_file_path, "r").read(), context)
    _print_verbose(module, "// IR Dump Input MLIR:") if verbose else ...

    ### legalize stablehlo to mhlo
    with context:
        PassManager.parse("builtin.module(canonicalize,stablehlo-legalize-to-hlo,canonicalize)").run(module.operation)
        _print_verbose(module, "// IR Dump After Legalize to HLO:") if verbose else ...

    ### compiling
    if target == "cuda":
        _compile_cuda(module, output_file_path, entry_func, gpu_type, verbose)
    elif target == "cuda_with_ait":
        _compile_cuda_with_ait(module,
                              output_file_path,
                              entry_func,
                              gpu_type,
                              verbose,
                              parallelism=parallelism,
                              disable_byteir_ait_cache=disable_byteir_ait_cache)
    elif target == "cuda_with_ait_aggressive":
        _compile_cuda_with_ait(module,
                              output_file_path,
                              entry_func,
                              gpu_type,
                              verbose,
                              aggressive_mode=True,
                              parallelism=parallelism,
                              disable_byteir_ait_cache=disable_byteir_ait_cache)
    else:
        raise NotImplemented("not implemented target: {}".format(target))
