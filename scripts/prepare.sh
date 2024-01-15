# note: need to apply mhlo patch with gcc8.3
function apply_mhlo_patches() {
  pushd $ROOT_PROJ_DIR/external/mlir-hlo
  git clean -fd .
  for patch in $ROOT_PROJ_DIR/external/patches/mlir-hlo/*; do
    git apply $patch
  done
  popd
}

function apply_aitemplate_patches() {
  pushd $ROOT_PROJ_DIR/external/AITemplate
  git clean -fd .
  for patch in $ROOT_PROJ_DIR/external/patches/AITemplate/*; do
    git apply $patch
  done
  popd
}

function install_aitemplate() {
  pushd $ROOT_PROJ_DIR/external/AITemplate/python
  python3 setup.py bdist_wheel
  python3 -m pip uninstall -y aitemplate
  python3 -m pip install dist/*.whl
  popd
}

function load_llvm_prebuilt() {
  LLVM_INSTALL_DIR="/data00/llvm_libraries/4592543a01609feb4b3c19e81a9d54743e15e329/llvm_build"
}

function load_pytorch_llvm_prebuilt() {
  TORCH_FRONTEND_LLVM_INSTALL_DIR="/data00/llvm_libraries/f7250179e22ce4aab96166493b27223fa28c2181/llvm_build"
}

function load_onnx_llvm_rtti_prebuilt() {
  ONNX_FRONTEND_LLVM_RTTI_INSTALL_DIR="/data00/llvm_libraries/d13da154a7c7eff77df8686b2de1cfdfa7cc7029/llvm_build_rtti"
}

function prepare_for_compiler() {
  git submodule update --init --recursive -f external/mlir-hlo external/AITemplate
  apply_aitemplate_patches
  install_aitemplate
  load_llvm_prebuilt
}

function prepare_for_runtime() {
  git submodule update --init --recursive -f external/mlir-hlo external/cutlass external/date external/googletest external/pybind11
  load_llvm_prebuilt
}
