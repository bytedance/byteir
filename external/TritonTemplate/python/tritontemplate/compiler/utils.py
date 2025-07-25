import subprocess
import torch

_TARGET2WARPSIZE={
    'cuda':32,
}

_DEVICE_MAX_SHARED_MEMORY={
    "8.0" : 163*1024,
    "8.6" : 99*1024,
    "8.7" : 163*1024,
    "8.9" : 99*1024,
    "9." : 227*1024,
    "10." : 227*1024,
    "12." : 99*1024,
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
    
def get_cuda_device_max_shared_memory():
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] == 8:
        return _DEVICE_MAX_SHARED_MEMORY[str(compute_capability[0])+"."+str(compute_capability[1])]
    elif compute_capability[0] < 8:
        raise KeyError(f'cuda compute capability {compute_capability} does not support triton')
    return _DEVICE_MAX_SHARED_MEMORY[str(compute_capability[0])+"."]
