import os
from torch_e2e_testing.registry import (
    GLOBAL_TORCH_TEST_REGISTRY,
    GLOBAL_TORCH_TEST_REGISTRY_NAMES,
)
from torch_e2e_testing.test_suite import register_all_torch_tests

register_all_torch_tests()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_test_files_from_dir(directory):
    test_files = []
    for filename in os.listdir(directory):
        if filename.startswith("."):
            continue
        if os.path.isfile(os.path.join(directory, filename)):
            test_files.append(filename)
    return test_files


##### CPU TEST SET #######
CPU_MLIR_TEST_DIR = os.path.join(CUR_DIR, "mlir_tests", "cpu_ops")
CPU_MLIR_TEST_SET = set(_get_test_files_from_dir(CPU_MLIR_TEST_DIR))
CPU_TORCH_TEST_SET = set()
CPU_XFAIL_SET = {
    "custom_call_tf_UpperBound.mlir",
    "rng.mlir",
}

CPU_ALL_SET = (CPU_MLIR_TEST_SET | CPU_TORCH_TEST_SET) - CPU_XFAIL_SET

##### CUDA TEST SET #######
CUDA_MLIR_TEST_DIR = os.path.join(CUR_DIR, "mlir_tests", "ops")
CUDA_MLIR_TEST_SET = set(_get_test_files_from_dir(CUDA_MLIR_TEST_DIR))
CUDA_TORCH_TEST_SET = set(GLOBAL_TORCH_TEST_REGISTRY_NAMES)
CUDA_XFAIL_SET = {
    "bmm_rcr.mlir",
    "bmm_rrc.mlir",
    "bmm_rrr_add_f16.mlir",
    "bmm_rrr_f16.mlir",
    "bmm_rrr_permute_f16.mlir",
    "bmm_rrr_permute_f32.mlir",
    "layernorm.mlir",
    "softmax.mlir",
    "transpose102.mlir",
    "transpose1023.mlir",
    "transpose120.mlir",
    "transpose1203.mlir",
    "transpose2013.mlir",
    "transpose120.mlir",
    "RngUniformModule_basic",
    "RngNormalModule_basic",
}

CUDA_ALL_SET = (CUDA_MLIR_TEST_SET | CUDA_TORCH_TEST_SET) - CUDA_XFAIL_SET

##### CUDA AIT TEST SET #######
CUDA_AIT_MLIR_TEST_SET = {
    "bmm_rcr.mlir",
    "bmm_rrc.mlir",
    "bmm_rrr_add_f16.mlir",
    "bmm_rrr_f16.mlir",
    "bmm_rrr_permute_f16.mlir",
    "bmm_rrr_permute_f32.mlir",
    "gemm_crr_f16.mlir",
    "gemm_rrr_f16.mlir",
    "gemm_rrr_f32.mlir",
    "layernorm.mlir",
    "softmax.mlir",
    "transpose2d.mlir",
    "transpose102.mlir",
    "transpose1023.mlir",
    "transpose120.mlir",
    "transpose1203.mlir",
    "transpose2013.mlir",
    "transpose120.mlir",
}
CUDA_AIT_TORCH_TEST_SET = {
    "MatmulF16Module_basic",
    "MatmulTransposeModule_basic",
    "MatmulF32Module_basic",
    "BatchMatmulF32Module_basic",
    "BatchMatmulAddF32Module_basic",
}
CUDA_AIT_SM80PLUS_SET = {
    "gemm_rrr_f32.mlir",
    "bmm_rrr_permute_f16.mlir",
    "bmm_rrr_permute_f32.mlir",
    "MatmulF32Module_basic",
    "BatchMatmulF32Module_basic",
    "BatchMatmulAddF32Module_basic",
}

CUDA_AIT_ALL_SET = CUDA_AIT_MLIR_TEST_SET | CUDA_AIT_TORCH_TEST_SET
