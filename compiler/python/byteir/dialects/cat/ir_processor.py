from byteir import ir
from byteir.dialects.func import FuncOp
from byteir.passmanager import PassManager
from byteir.utils import get_gpu_type

from .ait_cache import AITCache
from .tit_cache import TITCache

from pathlib import Path
from shutil import copyfile, copymode
import os
import time
import multiprocessing
import torch
import hashlib

BYTEIR_CAT_ATTR = "__byteir_cat_fusion__"

# FIXME(lyq): using cat op's python binding
SUPPORTED_CAT_OPS = [
    "cat.gemm_rrr",
    "cat.gemm_rcr",
    "cat.gemm_rrr_bias",
    "cat.gemm_rcr_bias",
    "cat.gemm_rcr_bias_relu",
    "cat.bmm_rrr",
    "cat.bmm_rcr",
    "cat.bmm_crr",
    "cat.bmm_ccr",
    "cat.bmm_rrr_add",
    "cat.bmm_rcr_add",
    "cat.bmm_crr_add",
    "cat.bmm_ccr_add",
    "cat.softmax",
    "cat.layernorm",
]

MAX_COMPILATION_PARALLELISM = torch.cuda.device_count()

def _print_verbose(module: ir.Module, pipeline_msg: str):
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
                 enable_tf32 = False,
                 compile_parallelism = 1,
                 disable_byteir_ait_cache = False,
                 verbose = False,
                 device = "cuda"):
        self.job_name = job_name
        self.workdir = workdir
        self.module = None
        self.enable_tf32 = enable_tf32
        self.compile_parallelism = min(compile_parallelism, MAX_COMPILATION_PARALLELISM)
        if self.compile_parallelism > 1:
            self.pool = multiprocessing.Pool(compile_parallelism)
        else:
            self.pool = None
        self.verbose = verbose
        #TODO: signature rename
        self.disable_byteir_cache = disable_byteir_ait_cache
        self.device = device

    def _get_builder(self, func, subgraph_name, backend="ait"):
        assert func != None
        assert isinstance(func, FuncOp)
        if backend == "ait":
            from byteir.dialects.cat.ir_translator.ait_builder import AITBuilder
            return AITBuilder(func, workdir=self.workdir, subgraph_name=subgraph_name, enable_tf32=self.enable_tf32)
        elif backend == "triton":
            from byteir.dialects.cat.ir_translator.tit_builder import TRITONTBuilder
            return TRITONTBuilder(func, workdir=self.workdir, subgraph_name=subgraph_name, enable_tf32=self.enable_tf32, device=self.device)
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

    def cat_opt_pass(self, anchor_only=False):
        with self.module.context:
            pass_arg = "builtin.module(func.func(convert-hlo-to-cat{valid-cat-ops=" + ",".join(SUPPORTED_CAT_OPS) + "}))"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            _print_verbose(self.module, "// IR Dump After Cat Fusion Opt:") if self.verbose else ...
        return self.module

    def ait_opt_pass(self, output_dir):
        self.byteir_cache = AITCache()
        if not self.disable_byteir_cache:
            self.byteir_cache.load_or_create_cache()
        funcNameArg = []
        aitLibPathArg = []

        gpu_type = get_gpu_type()
        if gpu_type == None:
            raise RuntimeError("No gpu found in this machine! cannot perform ait-opt-pass")
        work_items = [] # work items of FuncOp
        libs_to_add_to_cache = {} # key: hash_str, value: lib path
        for func in self.module.body.operations:
            if BYTEIR_CAT_ATTR not in func.attributes:
                continue
            output_lib_path = os.path.join(output_dir, func.name.value + ".so")
            hash_str = func_hash_str(func, gpu_type)
            cached_lib = self.byteir_cache.find(gpu_type, hash_str)
            if cached_lib:
                print(f"func {func.name.value} cache hit")
                copyfile(cached_lib, output_lib_path)
                copymode(cached_lib, output_lib_path)
            else:
                work_items.append(func)
                libs_to_add_to_cache[hash_str] = output_lib_path
            funcNameArg.append(func.name.value)
            aitLibPathArg.append(func.name.value + ".so")

        # compile and benchmark
        print("compile ait module using {} processes".format(min(len(work_items), self.compile_parallelism)))
        print("\n".join([str(func) for func in work_items]))
        t_st = time.time()
        for func in work_items:
            output_lib_path = os.path.join(output_dir, func.name.value + ".so")
            if self.pool:
                self.pool.apply_async(_parallel_ait_compile, (self.workdir, func, output_lib_path, self.enable_tf32))
            else:
                _parallel_ait_compile(self.workdir, func, output_lib_path, self.enable_tf32)
        if self.pool:
            self.pool.close()
            self.pool.join()
        t_ed = time.time()
        print("compilation finished in {}s".format(t_ed-t_st))

        # update byteir cache
        if not self.disable_byteir_cache:
            for key, lib_path in libs_to_add_to_cache.items():
                self.byteir_cache.add(gpu_type, key, lib_path, override=False)
            self.byteir_cache._save()
            self.byteir_cache.close_cache()

        with self.module.context:
            pm = PassManager.parse("builtin.module(func.func(gen-ait-config{{func-names={} ait-lib-paths={}}}))".format(",".join(funcNameArg), ",".join(aitLibPathArg)))
            pm.run(self.module.operation)
            _print_verbose(self.module, "// IR Dump After Gen AIT Config:") if self.verbose else ...

        return self.module
    
    def triton_opt_pass(self, output_dir):

        def decouple_triton_args(triton_args):
            func_name_args = []
            ptx_path_args = []
            gridsize_x_args = []
            gridsize_y_args = []
            gridsize_z_args = []
            blocksize_x_args = []
            blocksize_y_args = []
            blocksize_z_args = []
            smemsize_args = []
            for func_name,ptx_path,gridsize,blocksize,smem_size in triton_args:
                func_name_args.append(func_name)
                ptx_path_args.append(ptx_path)
                gridsize_x_args.append(str(gridsize[0]))
                gridsize_y_args.append(str(gridsize[1]))
                gridsize_z_args.append(str(gridsize[2]))
                blocksize_x_args.append(str(blocksize))
                blocksize_y_args.append(str(1))
                blocksize_z_args.append(str(1))
                smemsize_args.append(str(smem_size))
            return func_name_args, ptx_path_args, gridsize_x_args, gridsize_y_args, gridsize_z_args, blocksize_x_args, blocksize_y_args, blocksize_z_args,smemsize_args

        self.pool=None

        self.byteir_cache = TITCache()
        if not self.disable_byteir_cache:
            self.byteir_cache.load_or_create_cache()
        triton_args = []

        gpu_type = get_gpu_type()
        if gpu_type == None:
            raise RuntimeError("No gpu found in this machine! cannot perform triton-opt-pass")
        work_items = [] # work items of FuncOp
        
        for func in self.module.body.operations:
            if BYTEIR_CAT_ATTR not in func.attributes:
                continue
            output_ptx_path = os.path.join(output_dir, func.name.value + ".ptx")
            hash_str = func_hash_str(func, gpu_type)
            # TODO: gridsize order need to be checked 
            # gridsize form (x,y,z), blocksize form (x,y,z)
            cached_argv = self.byteir_cache.find(gpu_type, hash_str)
            if cached_argv:
                cache_ptx,gridsize,blocksize,smemsize = cached_argv
                print(f"func {func.name.value} cache hit")
                copyfile(cache_ptx, output_ptx_path)
                copymode(cache_ptx, output_ptx_path)
                triton_args.append((func.name.value,output_ptx_path, gridsize, blocksize,smemsize))
            else:
                work_items.append(func)

        # compile and benchmark
        print("compile triton module using {} processes".format(min(len(work_items), self.compile_parallelism)))
        print("\n".join([str(func) for func in work_items]))
        t_st = time.time()
        
        new_args = []
        for func in work_items:
            output_ptx_path = os.path.join(output_dir, func.name.value + ".ptx")
            if self.pool:
                new_args.append(self.pool.apply_async(_parallel_tit_compile, 
                    (self.workdir, func, output_ptx_path, self.enable_tf32)))
            else:
                new_args.append(_parallel_tit_compile(self.workdir, func, output_ptx_path, self.enable_tf32))
                
        if self.pool:
            self.pool.close()
            self.pool.join()
        
        for func,output_ptx_path,gridsize,blocksize,smemsize in new_args:
            triton_args.append((func.name.value,output_ptx_path, gridsize, blocksize,smemsize))
            self.byteir_cache.load_or_create_cache()
            self.byteir_cache.add(gpu_type, func_hash_str(func, gpu_type), (output_ptx_path, gridsize, blocksize,smemsize), override=False)
            self.byteir_cache._save()
            self.byteir_cache.close_cache()
                
        t_ed = time.time()
        print("compilation finished in {}s".format(t_ed-t_st))

        func_name_args, ptx_path_args, gridsize_x_args, gridsize_y_args, gridsize_z_args, blocksize_x_args, blocksize_y_args, blocksize_z_args,smemsize_args = decouple_triton_args(triton_args)
        ptx_path_args= [os.path.split(path)[-1] for path in ptx_path_args]

        with self.module.context:
            pm_str="builtin.module(func.func(gen-tit-config{{func-names={} tit-ptx-paths={} smemsize-args={} gridsize-x-args={} gridsize-y-args={} gridsize-z-args={} blocksize-x-args={} blocksize-y-args={} blocksize-z-args={}}}))".format(",".join(func_name_args), ",".join(ptx_path_args),",".join(smemsize_args), ",".join(gridsize_x_args), ",".join(gridsize_y_args), ",".join(gridsize_z_args), ",".join(blocksize_x_args), ",".join(blocksize_y_args), ",".join(blocksize_z_args))
            pm = PassManager.parse(pm_str)
            pm.run(self.module.operation)
            _print_verbose(self.module, "// IR Dump After Gen TIT Config:") if self.verbose else ...


        return self.module

    def execute(self, inputs, func_name=None, backend="ait"):
        if func_name is None:
            func = self.module.body.operations[0]
        else:
            func = ir.SymbolTable(self.module.operation)[func_name]
        builder = self._get_builder(func=func, subgraph_name=func.name.value, backend=backend)
        builder.compile()
        return builder.execute(inputs)

    def benchmark(self, func_name=None, backend="ait", num_trials=5):
        if func_name is None:
            func = self.module.body.operations[0]
        else:
            func = ir.SymbolTable(self.module.operation)[func_name]
        builder = self._get_builder(func=func, subgraph_name=func.name.value, backend=backend)
        builder.compile()
        builder.benchmark(num_trials)

    def profile(self, backend="ait"):
        pass


def _parallel_ait_compile(workdir: str, func: FuncOp, output_lib_path, enable_tf32):
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(os.getpid() % available_cuda_device_num)
    from byteir.dialects.cat.ir_translator.ait_builder import AITBuilder
    builder = AITBuilder(func, workdir=workdir, subgraph_name=func.name.value, enable_tf32=enable_tf32)
    builder.compile()
    builder.benchmark()
    copyfile(builder.ait_module_path, output_lib_path)
    copymode(builder.ait_module_path, output_lib_path)

def _parallel_tit_compile(workdir: str, func: FuncOp, output_ptx_path, enable_tf32):

    # os.environ["CUDA_VISIBLE_DEVICES"]=str(os.getpid() % available_cuda_device_num)
    from byteir.dialects.cat.ir_translator.tit_builder import TITBuilder
    builder = TITBuilder(func, workdir=workdir, subgraph_name=func.name.value, enable_tf32=enable_tf32)
    builder.compile()
    blockSize,gridsize,smemsize=builder.blocksize,builder.gridsize,builder.smemsize
    copyfile(builder.tit_module_path, output_ptx_path)
    copymode(builder.tit_module_path, output_ptx_path)
    return func,output_ptx_path,gridsize,blockSize,smemsize