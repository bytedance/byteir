#!/usr/bin/python3

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import shutil
import subprocess

def check_env_flag(name: str, default=None) -> bool:
    return str(os.getenv(name, default)).upper() in ["ON", "1", "YES", "TRUE", "Y"]

def get_git_commit(src_dir):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=src_dir).decode('ascii').strip()
    except Exception:
        return 'unknown'

def get_torch_frontend_version(version_txt_path):
    with open(version_txt_path) as f:
        version = f.readline().strip()
    return version

def get_torch_frontend_version_and_generate_versoin_file(input_version_txt_path, output_version_file_path, root_dir, *, enable_jitir=False):
    commit_id = get_git_commit(root_dir)
    torch_frontend_ver = get_torch_frontend_version(input_version_txt_path)

    if not enable_jitir:
        torch_frontend_ver += "+nojit"

    with open(output_version_file_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(torch_frontend_ver))
        f.write("git_version = {}\n".format(repr(commit_id)))

    return torch_frontend_ver

TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER = check_env_flag("TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER", True)
TORCH_FRONTEND_BUILD_DIR = os.getenv("TORCH_FRONTEND_BUILD_DIR", "build").split('/')[-1]

setup_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(setup_path + "/../../")

version_txt = os.path.join(setup_path, "version.txt")
version_file = os.path.join(setup_path, "torch_frontend", "version.py")
version = get_torch_frontend_version_and_generate_versoin_file(version_txt, version_file, root_path, enable_jitir=TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER)

maintainer = "ByteIR Team"
maintainer_email = "byteir@bytedance.com"
author = maintainer
author_email = maintainer_email
description = "use torch_frontend python interface"
long_description = """
torch_frontend
usage:
>>> import torch_frontend
>>> import torch_mlir
"""
install_requires = []
license = "LICENSE"
name = "torch-frontend"
platforms = ["linux_x86_64"]
url = ""
classifiers = [
    "Programming Language :: Python :: 3.6",
]

class TorchFrontendExtension(Extension):
  def __init__(self, name):
    Extension.__init__(self, name, sources=[])

class CustomBuild(build_ext):
  def run(self):
    super().run()
  
  def build_extension(self, ext):
    if isinstance(ext, TorchFrontendExtension):
      self.build_torch_mlir()
      return self.build_torch_frontend()
    
    return super().build_extension(ext)

  def build_torch_frontend(self):
    python_package_dir = os.path.join(
      root_path,
      TORCH_FRONTEND_BUILD_DIR,
      "python_packages",
      "torch_frontend",
    )
    ignore_pattern = shutil.ignore_patterns("*TorchFrontendMLIRAggregateCAPI.so")
    target_dir = os.path.join(self.build_lib, "torch_frontend")
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir, ignore_errors=False, onerror=None)
    shutil.copytree(python_package_dir, target_dir, symlinks=False, ignore=ignore_pattern)
    shutil.copyfile(version_file, os.path.join(target_dir, "version.py"))

  def build_torch_mlir(self):
    python_package_dir = os.path.join(
      root_path,
      TORCH_FRONTEND_BUILD_DIR,
      "torch_mlir_build",
      "python_packages",
      "torch_mlir",
      "torch_mlir",
    )
    target_dir = os.path.join(self.build_lib, "torch_mlir")
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir, ignore_errors=False, onerror=None)
    shutil.copytree(python_package_dir, target_dir, symlinks=False)

setup(
    author=author,
    author_email=author_email,
    description=description,
    license=license,
    long_description=long_description,
    install_requires=install_requires,
    maintainer=maintainer,
    name=name,
    packages=["torch_frontend", "torch_mlir"],
    package_data={"torch_mlir": ["extras/*.py"]},
    package_dir={"torch_frontend": setup_path+"/torch_frontend", "torch_mlir": root_path+"/third_party/torch-mlir/python/torch_mlir"},
    include_package_data=False,
    ext_modules=[TorchFrontendExtension("torch_frontend")],
    cmdclass={
      "build_ext": CustomBuild,
    },
    platforms=platforms,
    url=url,
    version=version,
    classifiers=classifiers,
)
