import os
import tempfile
import byteir
from byteir import ir

CUR_DIR = os.path.realpath(os.path.dirname(__file__))
TEST_ROOT_DIR = CUR_DIR + "/../../../test/"
temp_dir = tempfile.gettempdir()

# ==============================================================================
# test byteir.compile

def test_compile_mlp_inference():
    path = TEST_ROOT_DIR + "E2E/CUDA/MLPInference/input.mlir"
    byteir.compile(path, temp_dir + "/test.mlir", entry_func="forward")

def test_compile_ccl_inference():
    path = TEST_ROOT_DIR + "E2E/CUDA/CclInference/input.mlir"
    byteir.compile(path, temp_dir + "/test_ccl.mlir", entry_func="forward")

def test_compile_ccl_inference_mlirbc():
    path = TEST_ROOT_DIR + "E2E/CUDA/CclInference/input.mlir"
    byteir.compile(path, temp_dir + "/test_ccl.mlirbc", entry_func="forward")

def test_compile_mlp_inference_cpu():
    path = TEST_ROOT_DIR + "E2E/Host/Case0/00_Input.mlir"
    byteir.compile(path, temp_dir + "/test_cpu.mlir", entry_func="main", target="cpu")

def test_compile_mlp_inference_cpu_mlirbc():
    path = TEST_ROOT_DIR + "E2E/Host/Case0/00_Input.mlir"
    byteir.compile(path, temp_dir + "/test_cpu.mlirbc", entry_func="main", target="cpu")

# ==============================================================================
# test merge two modules

def test_merge_two_modules():
    path0 = TEST_ROOT_DIR + "Utils/testMergeTwoModulesCase0.mlir"
    path1 = TEST_ROOT_DIR + "Utils/testMergeTwoModulesCase0_1.mlir"
    context = ir.Context()
    context.allow_unregistered_dialects = True
    module0 = ir.Module.parse(open(path0, 'r').read(), context)
    module1 = ir.Module.parse(open(path1, 'r').read(), context)
    module2 = byteir.merge_two_modules(module0, module1)
    print(module2.operation.get_asm())

    path0 = TEST_ROOT_DIR + "Utils/testMergeTwoModulesCase1.mlir"
    path1 = TEST_ROOT_DIR + "Utils/testMergeTwoModulesCase1_1.mlir"
    context = ir.Context()
    context.allow_unregistered_dialects = True
    module0 = ir.Module.parse(open(path0, 'r').read(), context)
    module1 = ir.Module.parse(open(path1, 'r').read(), context)
    module2 = byteir.merge_two_modules(module0, module1)
    print(module2.operation.get_asm())

def test_merge_two_modules_with_mapping():
    path0 = TEST_ROOT_DIR + "Utils/testMergeTwoModulesCase2.mlir"
    path1 = TEST_ROOT_DIR + "Utils/testMergeTwoModulesCase2_1.mlir"
    context = ir.Context()
    context.allow_unregistered_dialects = True
    module0 = ir.Module.parse(open(path0, 'r').read(), context)
    module1 = ir.Module.parse(open(path1, 'r').read(), context)
    module2 = byteir.merge_two_modules(module0, module1, [0, 1])
    print(module2.operation.get_asm())

    path0 = TEST_ROOT_DIR + "Utils/testMergeTwoModulesCase2.mlir"
    path1 = TEST_ROOT_DIR + "Utils/testMergeTwoModulesCase2_1.mlir"
    context = ir.Context()
    context.allow_unregistered_dialects = True
    module0 = ir.Module.parse(open(path0, 'r').read(), context)
    module1 = ir.Module.parse(open(path1, 'r').read(), context)
    module2 = byteir.merge_two_modules(module0, module1, [0, -1])
    print(module2.operation.get_asm())

# ==============================================================================
# test translate to llvm

def test_translate_to_llvmbc():
    path = TEST_ROOT_DIR + "E2E/Host/Case0/03b_ToLLVMIR.mlir"
    context = ir.Context()
    with open(path, "r") as f:
        module = ir.Module.parse(f.read(), context)
        byteir.translate_to_llvmbc(module, temp_dir + "/test.ll.bc")

def test_translate_to_llvmir():
    path = TEST_ROOT_DIR + "E2E/Host/Case0/03b_ToLLVMIR.mlir"
    context = ir.Context()
    with open(path, "r") as f:
        module = ir.Module.parse(f.read(), context)
        byteir.translate_to_llvmir(module, temp_dir + "/test.ll")

# ==============================================================================
# test byre serialization

def test_serialize_byre():
    path = TEST_ROOT_DIR + "Dialect/Byre/Serialization/Compatibility/version_1_0_0.mlir"
    context = ir.Context()
    with open(path, "r") as f:
        module = ir.Module.parse(f.read(), context)
        byteir.serialize_byre(module, "1.0.0", temp_dir + "/test.mlirbc")

def test_deserialize_byre():
    paths = [TEST_ROOT_DIR + "Dialect/Byre/Serialization/Compatibility/version_1_0_0.mlir.bc",
             TEST_ROOT_DIR + "Dialect/Byre/Serialization/Compatibility/version_1_0_0.mlir.bc.v0"]
    context = ir.Context()
    for path in paths:
        with open(path, "rb") as f:
            module = byteir.deserialize_byre(f.read(), context)
            print(module.operation.get_asm())
