import os

byteir_not_use_cache = os.environ.get("BYTEIR_NOT_USE_CACHE") == "1"

# TODO. default not save fx graph.
byteir_save_fxgraph = os.environ.get("BYTEIR_SAVE_FXGRAPH", "1") == "1"
