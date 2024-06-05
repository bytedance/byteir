from torch_frontend.extra_shape_fn import byteir_extra_library
from torch_mlir import torchscript

import os
import shutil

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

def _get_extra_library_file():
    extra_library = []
    for op in byteir_extra_library:
        extra_library += byteir_extra_library[op]
    return torchscript._canon_extra_library(extra_library)

def main():
    temp_file_path = _get_extra_library_file()
    shutil.copyfile(temp_file_path, os.path.join(CUR_DIR, "extra_fn.mlir"))

if __name__ == "__main__":
    main()
