#!/usr/bin/python3

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os, errno
import shutil
import subprocess

setup_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(setup_path + "/../")

def get_git_commit(src_dir):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=src_dir).decode('ascii').strip()
    except Exception:
        return 'unknown'

def get_version_file(version_txt_path):
    with open(version_txt_path) as f:
        version = f.readline().strip()
    return version

def get_version_and_generate_version_file(input_version_txt_path, output_version_file_path):
    commit_id = get_git_commit(setup_path)
    version = get_version_file(input_version_txt_path)

    with open(output_version_file_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(commit_id)))
    return version


version_txt = os.path.join(setup_path, "../VERSION_NUMBER")
version_file = os.path.join(setup_path, "brt", "version.py")

version = get_version_and_generate_version_file(version_txt, version_file)

maintainer = ""
maintainer_email = ""
author = maintainer
author_email = maintainer_email
description = "brt python interface"
long_description = """
brt
usage:
>>> import brt
"""
install_requires = []
license = "LICENSE"
name = "brt"
platforms = ["linux_x86_64"]
url = ""
classifiers = [
    "Programming Language :: Python :: 3.6",
]

class BRTExtension(Extension):
  def __init__(self, name):
    Extension.__init__(self, name, sources=[])

class CustomBuild(build_ext):
  def run(self):
    super().run()
  
  def build_extension(self, ext):
    if isinstance(ext, BRTExtension):
      return self.build_brt(ext)

    return super().build_extension(ext)

  def build_brt(self, ext):
    brt_path = os.path.join(root_path, "build", "install", "lib", "libbrt.so")
    _brt_path = os.path.join(root_path, "build", "install", "_brt.so")
    brt_dst_path = os.path.join(self.build_lib, "brt", "lib", "libbrt.so")
    _brt_dst_path = self.get_ext_fullpath(ext.name)
    os.makedirs(os.path.dirname(brt_dst_path), exist_ok=True)
    os.makedirs(os.path.dirname(_brt_dst_path), exist_ok=True)
    shutil.copy(brt_path, brt_dst_path)
    shutil.copy(_brt_path, _brt_dst_path)

setup(
    author=author,
    author_email=author_email,
    description=description,
    license=license,
    long_description=long_description,
    install_requires=install_requires,
    maintainer=maintainer,
    name=name,
    packages=find_packages(where=setup_path, exclude=["examples", "src", "README.md"]),
    package_dir={"": setup_path},
    include_package_data=False,
    ext_modules=[
        BRTExtension("brt._brt"),
    ],
    cmdclass={
        "build_ext": CustomBuild,
    },
    platforms=platforms,
    url=url,
    version=version,
    classifiers=classifiers,
)
