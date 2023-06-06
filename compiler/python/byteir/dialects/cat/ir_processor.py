from byteir import ir
from byteir.passmanager import PassManager

from pathlib import Path
import os

BYTEIR_CAT_ATTR = "__byteir_cat_fusion__"

class IRProcessor:
    def __init__(self, job_name, workdir):
        self.job_name = job_name
        self.workdir = workdir
        self.module = None

    def _get_builder(self, module, subgraph_name, backend="ait"):
        assert module != None
        if backend == "ait":
            from byteir.dialects.cat.ir_translator.ait_builder import ait_builder
            return ait_builder(module, workdir=self.workdir, subgraph_name=subgraph_name)
        else:
            raise RuntimeError(f"Unsupported runtime backend {backend}")

    def load_from_file(self, module_file):
        self.module = ir.Module.parse(Path(module_file).read_text())

    def _dump_ir(self, ir_name):
        ir_file = f'{self.workdir}/{ir_name}'
        with open(ir_file, "w") as f:
            f.write(self.module.operation.get_asm())

    def preprocess_pass(self, dump_ir=False):
        with self.module.context:
            pass_arg = "builtin.module(cat-preprocess)"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            if dump_ir:
                self._dump_ir("{}.mhlo.simplified.mlir".format(self.job_name))
        return self.module

    def hlo_opt_pass(self, dump_ir=False):
        with self.module.context:
            pass_arg = "builtin.module(hlo-opt{outline-cat-op})"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            if dump_ir:
                self._dump_ir("{}.hlo_opt.mlir".format(self.job_name))
        return self.module

    def cat_opt_pass(self, anchor_only=False, dump_ir=False):
        with self.module.context:
            if anchor_only:
                pass_arg = "builtin.module(cat-opt{anchor-only})"
            else:
                pass_arg = "builtin.module(cat-opt)"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            if dump_ir:
                self._dump_ir("{}.cat_opt.mlir".format(self.job_name))
        return self.module

    def ait_opt_pass(self, anchor_only=False, dump_ir=False):
        if not anchor_only:
            builder = self._get_builder(backend=backend)
            return self.module
        funcNameArg = ""
        aitLibPathArg = ""
        for func in self.module.body.operations:
            if BYTEIR_CAT_ATTR not in func.attributes:
                continue
            builder = self._get_builder(module=func, subgraph_name=func.name.value, backend="ait")
            builder.benchmark()
            funcNameArg += func.name.value + ","
            # TODO: builder.ait_module_path is relative to cur path, not to IR
            # in byre.computeOp, ait_lib_path is relative to IR
            # need to match these two paths
            aitLibPathArg += builder.ait_module_path + ","
        
        with self.module.context:
            pm = PassManager.parse("builtin.module(func.func(gen-ait-config{{func-names={} ait-lib-paths={}}}))".format(funcNameArg, aitLibPathArg))
            pm.run(self.module.operation)
            if dump_ir:
                self._dump_ir("{}.ait_opt.mlir".format(self.job_name))
        return self.module

    def bufferize_opt_pass(self, dump_ir=False):
        with self.module.context:
            pass_arg = "builtin.module(byteir-bufferize-opt)"
            pm = PassManager.parse(pass_arg)
            pm.run(self.module.operation)
            if dump_ir:
                self._dump_ir("{}.bufferize_opt.mlir".format(self.job_name))
        return self.module

    def execute(self, inputs, backend="ait"):
        module = self.module.body.operations[0]
        subgraph_name = module.name.value
        builder = self._get_builder(module=module, subgraph_name=subgraph_name, backend=backend)
        return builder.execute(inputs)

    def benchmark(self, backend="ait", num_trials=10):
        module = self.module.body.operations[0]
        subgraph_name = module.name.value
        builder = self._get_builder(module=module, subgraph_name=subgraph_name, backend=backend)
        builder.benchmark(num_trials)

    def profile(self, backend="ait"):
        pass
