from typing import Callable, Dict, Optional

_BYTEIR_BACKENDS: Dict[str, Callable] = dict()
_BYTEIR_DEVICES: Dict[str, str] = dict()

def register_byteir_compiler_backend(
    compiler_fn: Optional[Callable] = None,
    target: Optional[str] = None,
    device: Optional[str] = None
):
    if compiler_fn is None or target is None:
        return None
    assert callable(compiler_fn), f"compiler_fn should be callable object"
    assert target not in _BYTEIR_BACKENDS, f"duplicate target name: {target}"
    compiler_fn.device = device
    _BYTEIR_DEVICES[target] = device
    _BYTEIR_BACKENDS[target] = compiler_fn

def list_backend_names():
    return list(_BYTEIR_BACKENDS.keys())

def get_backends():
    return list(_BYTEIR_BACKENDS.values())

def get_target_device(target):
    device = ''
    if target in _BYTEIR_DEVICES:
        device = _BYTEIR_DEVICES[target]
    return device

def look_up_backend(target: str):
    if target not in _BYTEIR_BACKENDS:
        raise RuntimeError(f"Unimplemented backend {target}")
    return _BYTEIR_BACKENDS[target]

