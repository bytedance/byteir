from byteir import ir
from byteir.dialects.cat.ait_cache import AITCache
from byteir.dialects.builtin import ModuleOp
from byteir.passmanager import PassManager

from pathlib import Path
from shutil import copyfile, copymode
import os
import time
import multiprocessing
import torch
from byteir.utils import get_gpu_type
import hashlib

BYTEIR_CAT_ATTR = "__byteir_cat_fusion__"

available_cuda_device_num = torch.cuda.device_count()
MAX_COMPILATION_PARALLELISM = available_cuda_device_num

def _print_verbose(module: ModuleOp, pipeline_msg: str):
    print(pipeline_msg)
    print(module.operation.get_asm(large_elements_limit=10))
    print()

def func_hash_str(func, gpu_type):
    hash_str = gpu_type + "_"
    ops = func.entry_block.operations
    for op in ops:
        hash_str += f"{op.get_asm(large_elements_limit=None)};"
    return hash_str

class IRProcessor:
    def __init__(self, 
                 job_name, 
                 workdir, 
                 compile_parallelism = MAX_COMPILATION_PARALLELISM,
                 disable_byteir_ait_cache = False,
                 verbose = False):
        self.job_name = job_name
        self.workdir = workdir
        self.module = None
        self.ait_reuse_recorder = {} # key: hash str, value: Tuple(dll_name, ait_module_path)
        self.compile_parallelism = min(compile_parallelism, MAX_COMPILATION_PARALLELISM)
        if self.compile_parallelism > 1:
            self.pool = multiprocessing.Pool(compile_parallelism)
        else:
            self.pool = None
        self.byteir_cache = AITCache()
        self.verbose = verbose
        self.disable_byteir_ait_cache = disable_byteir_ait_cache
        if not disable_byteir_ait_cache:
            self.byteir_cache.load_or_create_cache()

    def _get_builder(self, module, subgraph_name, backend="ait"):
        assert module != None
        if backend == "ait":
            from byteir.dialects.cat.ir_translator.ait_builder import AITBuilder
            return AITBuilder(module, workdir=self.workdir, subgraph_name=subgraph_name)
        else:
            raise RuntimeError(f"Unsupported runtime backend {backend}")

    def load_from_file(self, module_file):
        self.module = ir.Module.parse(Path(module_file).read_text())

    def preprocess_pass(self):
        with self.module.context:
            pass_arg = "builtin.module(cat-preprocess)"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            _print_verbose(self.module, "// IR Dump After Cat Preprocess:") if self.verbose else ...
        return self.module

    def hlo_opt_pass(self, outline_single_elemwise_op=True, aggressive_mode=False):
        with self.module.context:
            if outline_single_elemwise_op:
                if aggressive_mode:
                    pass_arg = "builtin.module(hlo-opt{outline-single-elemwise-op outline-cat-op aggressive-cat-fusion})"
                else:
                    pass_arg = "builtin.module(hlo-opt{outline-single-elemwise-op outline-cat-op})"
            else:
                pass_arg = "builtin.module(hlo-opt{outline-cat-op})"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            _print_verbose(self.module, "// IR Dump After Hlo Opt (with Cat):") if self.verbose else ...
        return self.module

    def cat_opt_pass(self, anchor_only=False, aggressive_mode=False):
        with self.module.context:
            if anchor_only:
                pass_arg = "builtin.module(cat-opt{anchor-only})"
            else:
                if aggressive_mode:
                    pass_arg = "builtin.module(cat-opt{aggressive-mode})"
                else:
                    pass_arg = "builtin.module(cat-opt)"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            _print_verbose(self.module, "// IR Dump After Cat Opt:") if self.verbose else ...
        return self.module

    def ait_opt_pass(self, anchor_only=False):
        if not anchor_only:
            builder = self._get_builder()
            builder.compile()
            return self.module
        funcNameArg = ""
        aitLibPathArg = ""
        dllPaths = []
        
        gpu_type = get_gpu_type()
        if gpu_type == None:
            raise RuntimeError("No gpu found in this machine! cannot perform ait-opt-pass")
        dedup_work_items = [] # deduplicated work items
        libs_to_add_to_cache = {} # key: hash_str, value: lib path 
        for func in self.module.body.operations:
            if BYTEIR_CAT_ATTR not in func.attributes:
                continue
            func_ir_str = func.get_asm(large_elements_limit=None)
            hash_str = func_hash_str(func, gpu_type)
            if self.verbose:
                print("ait op:")
                print(func_ir_str)
                print("hash str for this ait op:")
                print(hash_str)
            # perform ait reuse to remove duplicated work items
            if hash_str in self.ait_reuse_recorder:
                funcNameArg += func.name.value + ","
                aitLibPathArg += self.ait_reuse_recorder[hash_str][0] + ","
                dllPaths.append(self.ait_reuse_recorder[hash_str][1])
            else:
                builder = self._get_builder(module=func, subgraph_name=func.name.value, backend="ait")
                # builder.benchmark()
                funcNameArg += func.name.value + ","
                aitLibPathArg += builder.dll_name + ","
                dllPaths.append(builder.ait_module_path)
                self.ait_reuse_recorder[hash_str] = (builder.dll_name, builder.ait_module_path)
                libs_to_add_to_cache[hash_str] = builder.ait_module_path
                dedup_work_items.append((hash_str, func_ir_str))

        # search in byteir cache
        work_items_not_in_cache = []
        for hash_str, func_ir_str in dedup_work_items:
            cached_lib = self.byteir_cache.find(gpu_type, hash_str)
            if cached_lib != None:
                # hit, copy cached lib
                context = ir.Context()
                _module = ir.Module.parse(func_ir_str, context)
                assert len(_module.body.operations) == 1
                _func = _module.body.operations[0]
                builder = self._get_builder(module=_func, subgraph_name=_func.name.value, backend="ait")
                os.makedirs(os.path.dirname(builder.ait_module_path), exist_ok=True)
                copyfile(cached_lib, builder.ait_module_path)
                copymode(cached_lib, builder.ait_module_path)
                continue
            else:
                # miss, add to work_items
                work_items_not_in_cache.append(func_ir_str)

        # compile and benchmark
        print("compile ait module using {} processes".format(min(len(work_items_not_in_cache), self.compile_parallelism)))
        t_st = time.time()
        for func_ir_str in work_items_not_in_cache:
            if self.pool:
                self.pool.apply_async(_parallel_ait_compile, (self.workdir, func_ir_str))
            else:
                _parallel_ait_compile(self.workdir, func_ir_str)

        if self.pool:
            self.pool.close()
            self.pool.join()
        t_ed = time.time()
        print("compilation finished in {}s".format(t_ed-t_st))

        # update byteir cache
        if not self.disable_byteir_ait_cache:
            for key, lib_path in libs_to_add_to_cache.items():
                self.byteir_cache.add(gpu_type, key, lib_path, override=False)
            self.byteir_cache._save()
            self.byteir_cache.close_cache()

        with self.module.context:
            pm = PassManager.parse("builtin.module(func.func(gen-ait-config{{func-names={} ait-lib-paths={}}}))".format(funcNameArg, aitLibPathArg))
            pm.run(self.module.operation)
            _print_verbose(self.module, "// IR Dump After Gen AIT Config:") if self.verbose else ...

        return self.module, dllPaths

    def bufferize_opt_pass(self, dump_ir=False):
        with self.module.context:
            pass_arg = "builtin.module(byteir-bufferize-opt)"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            _print_verbose(self.module, "// IR Dump After ByteIR Bufferize Opt:") if self.verbose else ...
        return self.module

    def execute(self, inputs, backend="ait"):
        module = self.module.body.operations[0]
        subgraph_name = module.name.value
        builder = self._get_builder(module=module, subgraph_name=subgraph_name, backend=backend)
        builder.compile()
        return builder.execute(inputs)

    def benchmark(self, backend="ait", num_trials=5):
        module = self.module.body.operations[0]
        subgraph_name = module.name.value
        builder = self._get_builder(module=module, subgraph_name=subgraph_name, backend=backend)
        builder.compile()
        builder.benchmark(num_trials)

    def profile(self, backend="ait"):
        pass


def _parallel_ait_compile(workdir: str, ir_str: str):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(os.getpid() % available_cuda_device_num)
    context = ir.Context()
    module = ir.Module.parse(ir_str, context)
    assert len(module.body.operations) == 1
    func = module.body.operations[0]
    from byteir.dialects.cat.ir_translator.ait_builder import AITBuilder
    builder = AITBuilder(func, workdir=workdir, subgraph_name=func.name.value)
    builder.compile()
    builder.benchmark()
