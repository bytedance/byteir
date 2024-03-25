import os
import tempfile
import byteir
from byteir import ir

CUR_DIR = os.path.realpath(os.path.dirname(__file__))
TEST_ROOT_DIR = CUR_DIR + "/../../../test/"

temp_dir = tempfile.TemporaryDirectory()

def test_compile_mlp_inference():
    path = TEST_ROOT_DIR + "E2E/MLPInference/input.mlir"
    byteir.compile(path, temp_dir.name + "/test.mlir", entry_func="forward")

def test_translate_to_llvmbc():
    path = TEST_ROOT_DIR + "Pipelines/Host/E2E/Case0/03b_ToLLVMIR.mlir"
    context = ir.Context()
    with open(path, "r") as f:
        module = ir.Module.parse(f.read(), context)
        byteir.translate_to_llvmbc(module, temp_dir.name + "/test.ll.bc")

def test_serialize_byre():
    path = TEST_ROOT_DIR + "Dialect/Byre/Serialization/Compatibility/version_1_0_0.mlir"
    context = ir.Context()
    with open(path, "r") as f:
        module = ir.Module.parse(f.read(), context)
        byteir.serialize_byre(module, "1.0.0", temp_dir.name + "/test.mlir.bc")

def test_deserialize_byre():
    paths = [TEST_ROOT_DIR + "Dialect/Byre/Serialization/Compatibility/version_1_0_0.mlir.bc",
             TEST_ROOT_DIR + "Dialect/Byre/Serialization/Compatibility/version_1_0_0.mlir.bc.v0"]
    context = ir.Context()
    for path in paths:
        with open(path, "rb") as f:
            module = byteir.deserialize_byre(f.read(), context)
            print(module.operation.get_asm())
