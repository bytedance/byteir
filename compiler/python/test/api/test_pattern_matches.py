from byteir import ir
from byteir.pattern_matches import PDLPatternManager, RewritePatternBase

class TestRewriteAPattern(RewritePatternBase):

    op_name = "test.test_op_A"

    def match(self, op):
        return "__rewrite__" in op.attributes

    def rewrite(self, op) -> ir.Operation:
        return ir.Operation.create(
            "test.test_op_B",
            results=[i.type for i in op.results],
            operands=op.operands,
        )

class TestRewriteBPattern(RewritePatternBase):

    op_name = "test.test_op_B"

    def match(self, op):
        return True

    def rewrite(self, op) -> ir.Operation:
        return ir.Operation.create(
            "test.test_op_C",
            results=[i.type for i in op.results],
            operands=op.operands,
        )

def test_simple_rewrite():
    ctx = ir.Context()
    ctx.enable_multithreading(False)
    mgr = PDLPatternManager().add(TestRewriteAPattern())
    with ctx:
        mod = ir.Module.parse(
R"""
func.func @foo(%arg0: index) -> index {
    %0 = "test.test_op_A"(%arg0) {__rewrite__} : (index) -> index
    %1 = "test.test_op_A"(%0) : (index) -> index
    return %1 : index
}
"""
        )
        mgr.apply(mod, nested="func.func")
        func = mod.body.operations[0]
        operations = func.body.blocks[0].operations
        assert operations[0].name == "test.test_op_B"
        assert operations[1].name == "test.test_op_A"

def test_rewrite_chain():
    ctx = ir.Context()
    ctx.enable_multithreading(False)
    mgr = PDLPatternManager().add(TestRewriteAPattern(), TestRewriteBPattern())
    with ctx:
        mod = ir.Module.parse(
R"""
func.func @foo(%arg0: index) -> index {
    %0 = "test.test_op_A"(%arg0) {__rewrite__} : (index) -> index
    return %0 : index
}
"""
        )
        mgr.apply(mod, nested="func.func")
        func = mod.body.operations[0]
        operations = func.body.blocks[0].operations
        assert operations[0].name == "test.test_op_C"
