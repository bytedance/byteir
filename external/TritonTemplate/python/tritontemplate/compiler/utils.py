import subprocess

_TARGET2WARPSIZE={
    'cuda':32,
}

_DEVICE_MAX_SHARED_MEMORY={
    "NVIDIA H800": 227 * 1024,
    "NVIDIA H100": 227 * 1024,
    "NVIDIA A100": 164 * 1024,
    "NVIDIA A800": 164 * 1024,
    "NVIDIA V100": 96 * 1024,
    "NVIDIA T4": 64 * 1024,
}

def get_cuda_device_name(idx=0):
    cmd = "nvidia-smi --query-gpu=name --format=csv,noheader"
    result = subprocess.check_output(cmd, shell=True)
    gpu_names = result.decode().strip().split("\n")
    return gpu_names[idx]

def get_warpsize(target_name):
    try:
        return _TARGET2WARPSIZE[target_name]
    except KeyError:
        raise KeyError(f'target {target_name} not supported')
    
def get_device_max_shared_memory(target_name):
    try:
        return _DEVICE_MAX_SHARED_MEMORY[target_name]
    except KeyError:
        raise KeyError(f'target {target_name} not supported, please add max smem size info')
    