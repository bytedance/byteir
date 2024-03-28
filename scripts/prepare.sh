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
  LLVM_INSTALL_DIR="/data00/llvm_libraries/b2cdf3cc4c08729d0ff582d55e40793a20bbcdcc/llvm_build"
}

function install_mhlo_tools() {
  python3 -m pip install /data00/mhlo_libraries/mhlo_tools-1.3.0-cp39-cp39-linux_x86_64.whl --force-reinstall
}

function copy_external_libs() {
  PREBUILT_FLASH_ATTN="/data00/external_libraries/libflash_attn.so"
  mkdir $ROOT_PROJ_DIR/external_libs/libs
  cp $PREBUILT_FLASH_ATTN $ROOT_PROJ_DIR/external_libs/libs
  mkdir $ROOT_PROJ_DIR/runtime/test/test_files/external_libs/
  cp $PREBUILT_FLASH_ATTN $ROOT_PROJ_DIR/runtime/test/test_files/external_libs/
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
  copy_external_libs
}
