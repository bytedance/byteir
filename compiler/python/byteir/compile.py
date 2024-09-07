from enum import Enum
from pathlib import Path
import os
from typing import Union

from . import ir
from .passmanager import PassManager
from ._backend_registry import register_byteir_compiler_backend, get_target_device, look_up_backend
from .utils import detect_gpu_arch_with_nvidia_smi

import byteir

class OutputType(Enum):
    MLIR = "mlir"
    MLIRBC = "mlirbc"

class CompileOptions:
    def __init__(self,
                 target: str,
                 module: ir.Module,
                 output_dir: str,
                 output_file_prefix: str,
                 output_type: OutputType = OutputType.MLIR,
                 entry_func: str = "main",
                 gpu_arch: str = "local",
                 cpu_arch: str = '', # cpu arch ?
                 byre_serial_version: str = "1.0.0",
                 verbose: bool = False,
                 name: str = "model",
                 enable_tf32: bool = False,
                 parallelism: int = 1,
                 disable_byteir_ait_cache: bool = False,
                 **kwargs):
        self.target = target
        self.module = module
        self.output_dir = output_dir
        self.output_file_prefix = output_file_prefix
        self.output_type = output_type
        self.entry_func = entry_func
        self.gpu_arch = gpu_arch
        self.cpu_arch = cpu_arch
        self.byre_serial_version = byre_serial_version
        self.verbose = verbose
        self.name = name
        self.enable_tf32 = enable_tf32
        self.parallelism = parallelism
        self.disable_byteir_ait_cache = disable_byteir_ait_cache
        self.kwargs = kwargs

class DebugType(Enum):
    NO_DEBUG = 0
    PRINT_AFTER_FAILURE = 1
    PRINT_AFTER_ONLY_CHANGE = 2


def _get_debug_parameters(debug: DebugType):
    assert isinstance(debug, DebugType), "unknown debug type"
    # note: if you want to set `print_module_scope = True``,
    # you should set `module.context.enable_multithreading(False)`
    debug_parameters = {}
    if debug == DebugType.PRINT_AFTER_FAILURE:
        debug_parameters = {"print_before_pass":False,
                            "print_after_pass":True,
                            "print_after_only_on_change":False,
                            "print_after_only_on_failure":True,
                            "print_module_scope":False,
                            "large_elements_limit":10}
    elif debug == DebugType.PRINT_AFTER_ONLY_CHANGE:
        debug_parameters = {"print_before_pass":False,
                            "print_after_pass":True,
                            "print_after_only_on_change":True,
                            "print_after_only_on_failure":False,
                            "print_module_scope":False,
                            "large_elements_limit":10}
    return debug_parameters

def _print_verbose(module: ir.Module, pipeline_msg: str):
    print(pipeline_msg)
    print(module.operation.get_asm(large_elements_limit=10))
    print()


@register_byteir_compiler_backend(target="cuda", device="cuda")
def _compile_cuda(
    compile_options: CompileOptions,
) -> None:
    target = compile_options.target
    module = compile_options.module
    entry_func = compile_options.entry_func
    gpu_arch = compile_options.gpu_arch
    verbose = compile_options.verbose
    enable_tf32 = compile_options.enable_tf32

    output_file_dir = compile_options.output_dir
    output_file_prefix = compile_options.output_file_prefix
    output_type = compile_options.output_type
    useBarePtrCallConv = True # all tensor must have static shapes if True

    context = module.context

    entry_func_str = "entry-func={}".format(entry_func)
    target_str = "target={}".format(target)
    with context:
        PassManager().parse("builtin.module(hlo-graph-opt{" + entry_func_str + " " + target_str + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Hlo Graph Opt:") if verbose else ...
    with context:
        PassManager().parse("builtin.module(hlo-fusion-opt{outline-single-elemwise-op})").run(module.operation)
        _print_verbose(module, "// IR Dump After Hlo Fusion Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(linalg-tensor-opt)").run(module.operation)
        _print_verbose(module, "// IR Dump After Linalg Tensor Opt:") if verbose else ...
    with context:
        if enable_tf32:
            PassManager.parse("builtin.module(byre-tensor-opt{{append-arg-types enable-tf32 {}}})".format(entry_func_str)).run(module.operation)
        else:
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
            PassManager.parse("builtin.module(gpu-opt{use-bare-ptr-memref-call-conv=true  device-file-name="+ output_file_prefix + ".ptx" + "})").run(module.operation)
        else:
            PassManager.parse("builtin.module(gpu-opt{device-file-name=" + output_file_prefix + ".ptx" + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After GPU Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(inline)").run(module.operation)
        PassManager.parse("builtin.module(func.func(lccl-to-byre))").run(module.operation)
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
            PassManager.parse("builtin.module(nvvm-codegen{use-bare-ptr-memref-call-conv=true " + f" gpu-arch={gpu_arch}" + "})").run(device_module.operation)
        else:
            PassManager.parse("builtin.module(nvvm-codegen{" + f" gpu-arch= {gpu_arch}"  + "})").run(device_module.operation)
        _print_verbose(device_module, "// IR Dump After NVVM Codegen:") if verbose else ...
    # write to output device ptx file
    byteir.translate_to_ptx(device_module, output_file_dir + "/" + output_file_prefix, gpu_arch)

    # create host module
    with context:
        PassManager.parse("builtin.module(byre-host)").run(module.operation)
        PassManager.parse("builtin.module(remove-module-tag{attr-name=gpu.container_module})").run(module.operation)
        PassManager.parse("builtin.module(remove-module-tag{attr-name=torch.debug_module_name})").run(module.operation)
        _print_verbose(module, "// IR Dump After Byre Host:") if verbose else ...
    
    output_host_mlir_path = os.path.join(output_file_dir, output_file_prefix + "." + OutputType.MLIR.value)
    output_host_mlirbc_path = os.path.join(output_file_dir, output_file_prefix + "." + OutputType.MLIRBC.value)
    # write to output host mlir file
    with open(output_host_mlir_path, "w") as f:
        f.write(module.operation.get_asm())
    if output_type is OutputType.MLIRBC:
        byteir.serialize_byre(module, compile_options.byre_serial_version, output_host_mlirbc_path)
        deserialized_module = byteir.deserialize_byre(open(output_host_mlirbc_path, "rb").read(), context)
        if (module.operation.get_asm() != deserialized_module.operation.get_asm()):
            raise ValueError("module asm has be changed after byre serialization")


@register_byteir_compiler_backend(target="cuda_with_ait", device="cuda")
def _compile_cuda_with_ait(
    compile_options: CompileOptions,
) -> None:
    from .dialects.cat import IRProcessor

    target = "cuda"
    module = compile_options.module
    entry_func = compile_options.entry_func
    gpu_arch = compile_options.gpu_arch
    verbose = compile_options.verbose
    name = compile_options.name
    enable_tf32 = compile_options.enable_tf32
    parallelism = compile_options.parallelism
    disable_byteir_ait_cache = compile_options.disable_byteir_ait_cache

    output_file_dir = compile_options.output_dir
    output_file_prefix = compile_options.output_file_prefix
    output_type = compile_options.output_type
    useBarePtrCallConv = True # all tensor must have static shapes if True

    context = module.context

    entry_func_str = "entry-func={}".format(entry_func)
    target_str = "target={}".format(target)

    with context:
        PassManager().parse("builtin.module(hlo-graph-opt{" + entry_func_str + " " + target_str + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Hlo Graph Opt:") if verbose else ...

    processor = IRProcessor(name, 
                            "./workspace",
                            enable_tf32=enable_tf32,
                            compile_parallelism=parallelism,
                            disable_byteir_ait_cache=disable_byteir_ait_cache,
                            verbose=verbose)
    processor.module = module

    processor.preprocess_pass()
    processor.cat_opt_pass(anchor_only=False)

    with context:
        pm = PassManager().parse("builtin.module(hlo-fusion-opt{outline-single-elemwise-op outline-cat-op})")
        pm.run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After Hlo Fusion Opt (with Cat):") if verbose else ...

    # generate ait lib .so for cat functions
    module = processor.ait_opt_pass(output_file_dir)

    with context:
        PassManager.parse("builtin.module(linalg-tensor-opt)").run(processor.module.operation)
        _print_verbose(processor.module, "// IR Dump After Linalg Tensor Opt:") if verbose else ...
    with context:
        if enable_tf32:
            PassManager.parse("builtin.module(byre-tensor-opt{{append-arg-types enable-tf32 {}}})".format(entry_func_str)).run(processor.module.operation)
        else:
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
            PassManager.parse("builtin.module(gpu-opt{use-bare-ptr-memref-call-conv=true  device-file-name="+ output_file_prefix + ".ptx" + "})").run(module.operation)
        else:
            PassManager.parse("builtin.module(gpu-opt{device-file-name=" + output_file_prefix + ".ptx" + "})").run(module.operation)
        _print_verbose(processor.module, "// IR Dump After GPU Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(inline)").run(processor.module.operation)
        PassManager.parse("builtin.module(func.func(lccl-to-byre))").run(module.operation)
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
            PassManager.parse("builtin.module(nvvm-codegen{use-bare-ptr-memref-call-conv=true " + f" gpu-arch={gpu_arch}" + "})").run(device_module.operation)
        else:
            PassManager.parse("builtin.module(nvvm-codegen{" + f" gpu-arch= {gpu_arch}" + "})").run(device_module.operation)
        _print_verbose(device_module, "// IR Dump After NVVM Codegen:") if verbose else ...
    # write to output device ptx
    byteir.translate_to_ptx(device_module, output_file_dir + "/" + output_file_prefix, gpu_arch)

    # create host module
    with context:
        PassManager.parse("builtin.module(byre-host)").run(processor.module.operation)
        PassManager.parse("builtin.module(remove-module-tag{attr-name=gpu.container_module})").run(module.operation)
        PassManager.parse("builtin.module(remove-module-tag{attr-name=torch.debug_module_name})").run(module.operation)
        _print_verbose(processor.module, "// IR Dump After Byre Host:") if verbose else ...
    
    output_host_mlir_path = os.path.join(output_file_dir, output_file_prefix + "." + OutputType.MLIR.value)
    output_host_mlirbc_path = os.path.join(output_file_dir, output_file_prefix + "." + OutputType.MLIRBC.value)
    # write to output host mlir file
    with open(output_host_mlir_path, "w") as f:
        f.write(module.operation.get_asm())
    if output_type is OutputType.MLIRBC:
        byteir.serialize_byre(module, compile_options.byre_serial_version, output_host_mlirbc_path)
        deserialized_module = byteir.deserialize_byre(open(output_host_mlirbc_path, "rb").read(), context)
        if (module.operation.get_asm() != deserialized_module.operation.get_asm()):
            raise ValueError("module asm has be changed after byre serialization")


@register_byteir_compiler_backend(target="cpu", device="cpu")
def _compile_cpu(
    compile_options: CompileOptions,
) -> None:
    target = compile_options.target
    module = compile_options.module
    entry_func = compile_options.entry_func
    cpu_arch = compile_options.cpu_arch
    verbose = compile_options.verbose

    output_file_dir = compile_options.output_dir
    output_file_prefix = compile_options.output_file_prefix
    output_type = compile_options.output_type
    bc_file_name = output_file_prefix + ".kernel.ll.bc"
    llir_file_name = output_file_prefix + ".kernel.ll"
    useBarePtrCallConv = True # all tensor must have static shapes if True

    context = module.context

    entry_func_str = "entry-func={}".format(entry_func)
    target_str = "target={}".format(target)
    arch_str="arch={}".format(cpu_arch)
    with context:
        PassManager().parse("builtin.module(hlo-graph-opt{" + entry_func_str + " " + target_str + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Hlo Graph Opt:") if verbose else ...
    with context:
        PassManager().parse("builtin.module(hlo-fusion-opt{" + entry_func_str + " " + target_str + " outline-single-elemwise-op})").run(module.operation)
        _print_verbose(module, "// IR Dump After Hlo Fusion Opt:") if verbose else ...
    with context:
        PassManager.parse("builtin.module(linalg-tensor-opt{" + target_str + " " + arch_str + "})").run(module.operation)
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
        PassManager.parse("builtin.module(host-opt{" + "file-name={}".format(bc_file_name) + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Host Opt:") if verbose else ...

        PassManager.parse("builtin.module(func.func(set-op-space{" + entry_func_str + " space={}".format(target) +  "}))").run(module.operation)
        _print_verbose(module, "// IR Dump After Set Op Space Opt:") if verbose else ...
        PassManager.parse("builtin.module(set-arg-space{" + entry_func_str + " all-space={}".format(target) + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Set Space Opt:") if verbose else ...

    with context:
        PassManager.parse("builtin.module(byre-opt{append-arg-types " + entry_func_str + "})").run(module.operation)
        _print_verbose(module, "// IR Dump After Byre Opt:") if verbose else ...

    module_str = module.operation.get_asm(print_generic_op_form=True)
    llvm_module = ir.Module.parse(module_str, context)
    with context:
        PassManager.parse("builtin.module(to-llvm)").run(llvm_module.operation)
        _print_verbose(llvm_module, "// IR Dump After To LLVM:") if verbose else ...

    output_bc_path = output_file_dir + "/" + bc_file_name
    output_llir_path = output_file_dir + "/" + llir_file_name
    # write to output llvmbc file
    byteir.translate_to_llvmbc(llvm_module, output_bc_path)
    # write to llvm ir file for debug
    byteir.translate_to_llvmir(llvm_module, output_llir_path)

    # create host module
    with context:
        PassManager.parse("builtin.module(byre-host)").run(module.operation)
        _print_verbose(module, "// IR Dump After Byre Host:") if verbose else ...

    output_host_mlir_path = os.path.join(output_file_dir, output_file_prefix + "." + OutputType.MLIR.value)
    output_host_mlirbc_path = os.path.join(output_file_dir, output_file_prefix + "." + OutputType.MLIRBC.value)
    # write to output host mlir file
    with open(output_host_mlir_path, "w") as f:
        f.write(module.operation.get_asm())
    if output_type is OutputType.MLIRBC:
        byteir.serialize_byre(module, compile_options.byre_serial_version, output_host_mlirbc_path)
        deserialized_module = byteir.deserialize_byre(open(output_host_mlirbc_path, "rb").read(), context)
        if (module.operation.get_asm() != deserialized_module.operation.get_asm()):
            raise ValueError("module asm has be changed after byre serialization")

def compile_from_string(
    input_string_or_bytes: Union[str, bytes],
    output_file_path: str,
    entry_func: str = "main",
    target: str = "cuda",
    gpu_arch: str = "local",
    cpu_arch: str = "x86_64",
    byre_serial_version: str = "1.0.0",
    verbose: bool = False,
    enable_tf32: bool = False,
    parallelism: int = 1,
    disable_byteir_ait_cache: bool = False,
    **kwargs,
) -> None:
    _device = get_target_device(target)
    ### optional detecting gpu type from nvidia-smi
    if _device == "cuda" and gpu_arch == "local":
        local_gpu_arch = detect_gpu_arch_with_nvidia_smi()
        assert local_gpu_arch is not None, "seems it doesn't have gpu on local"
        gpu_arch = local_gpu_arch
    if _device == "cuda":
        gpu_arch_num = int(gpu_arch[3:])
        if enable_tf32:
            assert gpu_arch_num >= 80, "1xtf32 only support on gpu >= sm_80"
        print(f"[ByteIR] Compiling to {gpu_arch}")
    elif _device  == "cpu":
        print(f"[ByteIR] Compiling to {cpu_arch}")

    context = ir.Context()
    module = ir.Module.parse(input_string_or_bytes, context)
    _print_verbose(module, "// IR Dump Input MLIR:") if verbose else ...

    ### legalize stablehlo to mhlo
    with context:
        PassManager.parse("builtin.module(canonicalize,stablehlo-legalize-to-hlo,canonicalize)").run(module.operation)
        _print_verbose(module, "// IR Dump After Legalize to HLO:") if verbose else ...

    ### parse output options from output_file_path
    output_dir = os.path.dirname(os.path.abspath(output_file_path))
    os.makedirs(output_dir, exist_ok=True)
    output_file_basename = os.path.basename(output_file_path)
    output_type = OutputType.MLIR
    if output_file_basename.endswith(".mlirbc"):
        output_type = OutputType.MLIRBC
    else:
        assert output_file_basename.endswith(".mlir")
    output_file_prefix = os.path.splitext(output_file_basename)[0]

    ### compile options
    compile_options = CompileOptions(
        target,
        module,
        output_dir,
        output_file_prefix,
        output_type=output_type,
        entry_func=entry_func,
        gpu_arch=gpu_arch,
        cpu_arch=cpu_arch,
        byre_serial_version=byre_serial_version,
        verbose=verbose,
        enable_tf32=enable_tf32,
        parallelism=parallelism,
        disable_byteir_ait_cache=disable_byteir_ait_cache,
        kwargs=kwargs)

    ### compiling
    _compile_fn = look_up_backend(compile_options.target)
    if _compile_fn is not None:
        _compile_fn(compile_options)
    else:
        raise NotImplementedError("not implemented target: {}".format(target))

def compile(
    input_file_path: str,
    output_file_path: str,
    entry_func: str = "main",
    target: str = "cuda",
    gpu_arch: str = "local",
    cpu_arch: str = "x86_64",
    byre_serial_version: str = "1.0.0",
    verbose: bool = False,
    enable_tf32: bool = False,
    parallelism: int = 1,
    disable_byteir_ait_cache: bool = False,
    **kwargs,
) -> None:
    ### load from .mlir or .mlirbc
    from byteir._mlir_libs._stablehlo import deserialize_portable_artifact
    if input_file_path.endswith(".mlirbc"):
        module_bytes = deserialize_portable_artifact(open(input_file_path, "rb").read())
    else:
        module_bytes = open(input_file_path, "r").read()
    
    compile_from_string(module_bytes,
                        output_file_path=output_file_path,
                        entry_func=entry_func,
                        target=target,
                        gpu_arch=gpu_arch,
                        cpu_arch=cpu_arch,
                        byre_serial_version=byre_serial_version,
                        verbose=verbose,
                        enable_tf32=enable_tf32,
                        parallelism=parallelism,
                        disable_byteir_ait_cache=disable_byteir_ait_cache,
                        kwargs=kwargs)    
