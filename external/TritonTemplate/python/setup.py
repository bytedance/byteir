import os
import shutil

from setuptools import find_packages, setup

CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, "tritontemplate", "_libinfo.py")
libinfo = {}
with open(libinfo_py, "r") as f:
    exec(f.read(), libinfo)
__version__ = libinfo["__version__"]

def gen_file_list(srcs, f_cond):
    file_list = []
    for src in srcs:
        for root, _, files in os.walk(src):
            value = []
            for file in files:
                if f_cond(file):
                    path = os.path.join(root, file)
                    value.append(path.replace("tritontemplate/", ""))
            file_list.extend(value)
    return file_list

def gen_backend_common_file_list():
    srcs = ["tritontemplate/backend"]
    f_cond = lambda x: True if x.endswith(".py") else False
    return gen_file_list(srcs, f_cond)

def gen_utils_file_list():
    srcs = ["tritontemplate/utils"]
    f_cond = lambda x: True if x.endswith(".py") else False
    return gen_file_list(srcs, f_cond)

def gen_compiler_file_list():
    srcs = ["tritontemplate/compiler"]
    f_cond = lambda x: True if x.endswith(".py") else False
    return gen_file_list(srcs, f_cond)

setup_kwargs = {}
include_libs = True
wheel_include_libs = True

setup(
    name="tritontemplate",
    version=__version__,
    description="TritonTemplate: Make Flex Triton Templates for AI",
    zip_safe=True,
    install_requires=["torch>=2.1.0","triton"],
    packages=find_packages(),
    package_data={
        "tritontemplate": []
        + gen_utils_file_list()
        + gen_backend_common_file_list()
        + gen_compiler_file_list()
    },
    python_requires=">=3.7, <4",
    **setup_kwargs
)