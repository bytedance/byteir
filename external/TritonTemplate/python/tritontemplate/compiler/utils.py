
_TARGET2WARPSIZE={
    'cuda':32,
}

def get_warpsize(target_name):
    try:
        return _TARGET2WARPSIZE[target_name]
    except KeyError:
        raise KeyError(f'target {target_name} not supported')