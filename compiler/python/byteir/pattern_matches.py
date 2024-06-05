from byteir import ir
from byteir.dialects import pdl, _pdl_ops_gen
from byteir.passmanager import PassManager

from byteir import register_pdl_constraint_fn as _register_pdl_constraint_fn
from byteir import register_pdl_rewrite_fn as _register_pdl_rewrite_fn
from byteir import PDLValueKind

import tempfile
import functools
from collections.abc import Iterable

class RewritePatternBase:
    @property
    def op_name(self):
        raise NotImplementedError()

    def match(self, *args, **kwargs):
        raise NotImplementedError()

    def rewrite(self, *args, **kwargs):
        raise NotImplementedError()

class PDLPatternBase:
    def register_pdl_hooks(self, pdl_pattern_mgr):
        pass

    def emit_pdl(self):
        raise NotImplementedError()

class PDLPatternWrapper(PDLPatternBase):
    def __init__(self, pattern):
        self.pattern = pattern

    @functools.cached_property
    def op_name(self):
        return self.pattern.op_name

    @functools.cached_property
    def match(self):
        fn = self._match_fn

        @functools.wraps(fn)
        def builder(*args):
            return pdl.ApplyNativeConstraintOp(fn.__qualname__, args=args)

        return builder

    @functools.cached_property
    def rewrite(self):
        fn = self._rewrite_fn

        def get_pdl_type(value_kind):
            for k, v in PDLValueKind.__members__.items():
                if value_kind == v:
                    if k.endswith("Range"):
                        return pdl.RangeType.get(getattr(pdl, f"{k[:-5]}Type").get())
                    else:
                        return getattr(pdl, f"{k}Type").get()
            raise TypeError(f"unknown pdl type {value_kind}")

        @functools.wraps(fn)
        def builder(*args):
            return pdl.ApplyNativeRewriteOp([
                get_pdl_type(i) for i in fn.__annotations__["return"]
            ], fn.__qualname__, args=args)

        return builder

    def register_pdl_hooks(self, pdl_pattern_mgr):
        self._match_fn = pdl_pattern_mgr.register_pdl_constraint_fn(self.pattern.match)
        self._rewrite_fn = pdl_pattern_mgr.register_pdl_rewrite_fn(self.pattern.rewrite)

    def emit_pdl(self):
        with ir.InsertionPoint(pdl.PatternOp(0).body):
            rangeType = pdl.RangeType.get(pdl.TypeType.get())
            types = _pdl_ops_gen.TypesOp(rangeType)
            operands = pdl.OperandsOp()
            op = pdl.OperationOp(
                name=self.op_name,
                types=[types],
                args=[operands],
            )
            self.match(op)

            with ir.InsertionPoint(pdl.RewriteOp(op).add_body()):
                replOp = self.rewrite(op)
                if len(replOp.results) > 1:
                    raise NotImplementedError("replace op should return value ranges or operation")

                replType = replOp.results[0].type
                if pdl.RangeType.isinstance(replType) and pdl.ValueType.isinstance(pdl.RangeType(replType).element_type):
                    pdl.ReplaceOp(op, with_values=replOp)
                elif pdl.OperationType.isinstance(replType):
                    pdl.ReplaceOp(op, with_op=replOp)
                else:
                    raise NotImplementedError("replace op should return value ranges or operation")

class PDLPatternManager:
    def __init__(self):
        self._initialize()

    def _initialize(self):
        self.patterns = []
        self.constraints = []
        self.rewrites = []
        self.fn_uniquer = dict()

    def unique_function(self, fn):
        func_name = fn.__qualname__
        cnt = self.fn_uniquer.get(func_name, 0)
        if cnt != 0:
            fn.__qualname__ = f"{func_name}_{cnt}"
        self.fn_uniquer[func_name] = cnt + 1

    @classmethod
    def canonicalize_ty(cls, ty) -> PDLValueKind:
        mappings = {
            ir.Type : PDLValueKind.Type,
            ir.Value : PDLValueKind.Value,
            ir.Attribute : PDLValueKind.Attribute,
            ir.Operation : PDLValueKind.Operation,
        }
        if isinstance(ty, PDLValueKind):
            return ty
        elif ty in mappings:
            return mappings[ty]
        elif getattr(ty, '__origin__', None) == list:
            arg = ty.__args__[0]
            if arg == ir.Value:
                return PDLValueKind.ValueRange
            elif arg == ir.Type:
                return PDLValueKind.TypeRange
        raise NotImplementedError()


    def register_pdl_constraint_fn(self, fn):
        @functools.wraps(fn)
        def wrapper(args):
            return fn(*args)

        self.unique_function(wrapper)
        self.constraints.append(wrapper)
        return wrapper

    def register_pdl_rewrite_fn(self, fn):
        return_type = fn.__annotations__["return"]
        if isinstance(return_type, Iterable):
            return_type = [self.canonicalize_ty(i) for i in return_type]
        else:
            return_type = [self.canonicalize_ty(return_type),]

        @functools.wraps(fn)
        def wrapper(args):
            return fn(*args)

        wrapper.__annotations__["return"] = return_type
        self.unique_function(wrapper)
        self.rewrites.append(wrapper)
        return wrapper

    def attach_to_ctx(self, context):
        for fn in self.constraints:
            _register_pdl_constraint_fn(context, fn.__qualname__, fn)

        for fn in self.rewrites:
            _register_pdl_rewrite_fn(context, fn.__qualname__, fn, fn.__annotations__["return"])

    def add(self, *patterns):
        for pattern in patterns:
            if isinstance(pattern, Iterable):
                self.add(*pattern)
                continue

            if isinstance(pattern, RewritePatternBase):
                pattern = PDLPatternWrapper(pattern)

            assert isinstance(pattern, PDLPatternBase)

            self.patterns.append(pattern)
            pattern.register_pdl_hooks(self)

        return self

    def finalize(self, file):
        with ir.Context(), ir.Location.unknown():
            m = ir.Module.create()
            with ir.InsertionPoint(m.body):
                for pattern in self.patterns:
                    pattern.emit_pdl()
            m.operation.print(file=file)
            file.flush()

    def emit_pass(self, *, cleanup=False):
        self.f = tempfile.NamedTemporaryFile(mode='w+')
        self.finalize(self.f)

        if cleanup:
            self._initialize()

        return f'apply-pdl-patterns="pdl-file={self.f.name}"'

    def apply(self, module, *, nested=None):
        pass_and_arg = self.emit_pass().split("=", 1)
        pass_name = pass_and_arg[0]
        pass_arg = ""
        if len(pass_and_arg) > 1:
            pass_arg = eval(pass_and_arg[1])

        pm = PassManager("builtin.module")
        if nested:
            pm.add(f"{nested}({pass_name}{{{pass_arg}}})")
        else:
            pm.add(f"{pass_name}{{{pass_arg}}}")
        with module.context as ctx:
            self.attach_to_ctx(ctx)
            pm.run(module.operation)
