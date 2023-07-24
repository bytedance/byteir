function apply_aitemplate_patches() {
  pushd $ROOT_PROJ_DIR/external/AITemplate
  git clean -fd .
  for patch in $ROOT_PROJ_DIR/external/patches/AITemplate/*; do
    git apply $patch
  done
  popd
}

function install_aitemplate() {
  pushd external/AITemplate/python
  python3 setup.py bdist_wheel
  python3 -m pip uninstall -y aitemplate
  python3 -m pip install dist/*.whl
  popd
}

function prepare_for_compiler() {
  git submodule update --init --recursive -f external/mlir-hlo external/AITemplate
  apply_aitemplate_patches
  install_aitemplate
}

function prepare_for_runtime() {
  git submodule update --init --recursive -f external/mlir-hlo external/cutlass external/date external/googletest external/pybind11
}
