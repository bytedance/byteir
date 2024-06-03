from torch._dynamo import register_backend


@register_backend
def byteir(*args, **kwargs):
    from .compiler import byteir_compiler

    return byteir_compiler(*args, **kwargs)

def set_cache_dir(path: str):
    from .compilation_cache import ByteIRFxGraphCache

    ByteIRFxGraphCache.base_cache_dir = path
