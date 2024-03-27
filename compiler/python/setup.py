#!/usr/bin/python3

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import shutil

from gen_version import get_byteir_version_and_generate_versoin_file

setup_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(setup_path + "/../")

version_txt = os.path.join(setup_path, "version.txt")
version_file = os.path.join(setup_path, "byteir", "version.py")
version = get_byteir_version_and_generate_versoin_file(version_txt, version_file, root_path, dev=False)

maintainer = "ByteIR Team"
maintainer_email = "byteir@bytedance.com"
author = maintainer
author_email = maintainer_email
description = "use byteir python interface"
long_description = """
byteir
usage:
>>> import byteir
"""
install_requires = []
license = "LICENSE"
name = "byteir"
platforms = ["linux_x86_64"]
url = ""
classifiers = [
    "Programming Language :: Python :: 3.6",
]

class ByteIRExtension(Extension):
  def __init__(self, name):
    Extension.__init__(self, name, sources=[])

class CustomBuild(build_ext):
  def run(self):
    super().run()
  
  def build_extension(self, ext):
    if isinstance(ext, ByteIRExtension):
      return self.build_byteir()
    
    return super().build_extension(ext)

  def build_byteir(self):
    python_package_dir = os.path.join(
      root_path,
      "build",
      "python_packages",
      "byteir",
      "byteir",
    )
    target_dir = os.path.join(self.build_lib, "byteir")
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir, ignore_errors=False, onerror=None)
    shutil.copytree(python_package_dir, target_dir, symlinks=False)
    shutil.copyfile(version_file, os.path.join(target_dir, "version.py"))

setup(
    author=author,
    author_email=author_email,
    description=description,
    license=license,
    long_description=long_description,
    install_requires=install_requires,
    maintainer=maintainer,
    name=name,
    packages=find_packages(where=setup_path, exclude=["build", "test", "__pycache__"]),
    package_dir={"": os.path.dirname(__file__)},
    include_package_data=False,
    ext_modules=[ByteIRExtension("byteir")],
    cmdclass={
      "build_ext": CustomBuild,
    },
    platforms=platforms,
    url=url,
    version=version,
    classifiers=classifiers,
)
