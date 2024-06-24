import os
import subprocess

def get_git_commit(src_dir):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=src_dir).decode('ascii').strip()
    except Exception:
        return 'unknown'

def get_byteir_version(version_txt_path):
    with open(version_txt_path) as f:
        version = f.readline().strip()
    return version

def get_byteir_version_and_generate_versoin_file(input_version_txt_path, output_version_file_path, root_dir, *, dev=False):
    commit_id = get_git_commit(root_dir)
    byteir_ver = get_byteir_version(input_version_txt_path)

    if dev:
        byteir_ver += ".dev0+{}".format(commit_id[:8])

    with open(output_version_file_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(byteir_ver))
        f.write("git_version = {}\n".format(repr(commit_id)))

    return byteir_ver

