# -*- Python -*-

import os
import sys
import re
import platform
import subprocess

import lit.util
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = 'ByteIR'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.byteir_test_build_dir

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.byteir_tools_dir,
    config.llvm_tools_dir
]
tool_names = [
    'byteir-opt',
    'byteir-stat',
    'byteir-translate'
]
tools = [ToolSubst(s, unresolved='ignore') for s in tool_names]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# where to find the thrid_party dir
config.substitutions.append(('%third_party_path',
                            os.path.join(config.test_source_root, '..', 'third_party')))

# for PTX tests
llvm_config.with_system_environment(['CUDA_HOME'])
