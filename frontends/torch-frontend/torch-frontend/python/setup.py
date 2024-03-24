#!/usr/bin/python3

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import shutil

from gen_version import get_torch_frontend_version_and_generate_versoin_file

setup_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(setup_path + "/../../")

version_txt = os.path.join(setup_path, "version.txt")
version_file = os.path.join(setup_path, "torch_frontend", "version.py")
version = get_torch_frontend_version_and_generate_versoin_file(version_txt, version_file, root_path, dev=False)

maintainer = ""
maintainer_email = ""
author = maintainer
author_email = maintainer_email
description = "use torch_frontend python interface"
long_description = """
torch_frontend
usage:
>>> import torch_frontend
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
      "build",
      "python_packages",
      "torch_frontend",
    )
    target_dir = os.path.join(self.build_lib, "torch_frontend")
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir, ignore_errors=False, onerror=None)
    shutil.copytree(python_package_dir, target_dir, symlinks=False)
    shutil.copyfile(version_file, os.path.join(target_dir, "version.py"))

  def build_torch_mlir(self):
    python_package_dir = os.path.join(
      root_path,
      "build",
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
