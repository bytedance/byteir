#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

pushd $CUR_DIR

cp ../test/E2E/ResNet18/BW/host_output.mlir ../../runtime/test/test_files/resnet18_bw_host_cuda.mlir
sed -i 's/your_file/test\/test_files\/resnet18_bw_device.ptx/g' ../../runtime/test/test_files/resnet18_bw_host_cuda.mlir
cp ../test/E2E/ResNet18/BW/device_output.ptx ../../runtime/test/test_files/resnet18_bw_device.ptx

cp ../test/E2E/ResNet18/FW/host_output.mlir ../../runtime/test/test_files/resnet18_fw_host_cuda.mlir
sed -i 's/your_file/test\/test_files\/resnet18_fw_device.ptx/g' ../../runtime/test/test_files/resnet18_fw_host_cuda.mlir
cp ../test/E2E/ResNet18/FW/device_output.ptx ../../runtime/test/test_files/resnet18_fw_device.ptx

cp ../test/E2E/ResNet18/Whole/host_output.mlir ../../runtime/test/test_files/resnet18_fw_bw_host_cuda.mlir
sed -i 's/your_file/test\/test_files\/resnet18_fw_bw_device.ptx/g' ../../runtime/test/test_files/resnet18_fw_bw_host_cuda.mlir
cp ../test/E2E/ResNet18/Whole/device_output.ptx ../../runtime/test/test_files/resnet18_fw_bw_device.ptx

# cp ../test/E2E/BertTiny/host_output.mlir ../../runtime/test/test_files/bert_tiny_host_cuda.mlir
# sed -i 's/your_file/test\/test_files\/bert_tiny_device.ptx/g' ../../runtime/test/test_files/bert_tiny_host_cuda.mlir
# cp ../test/E2E/BertTiny/device_output.ptx ../../runtime/test/test_files/bert_tiny_device.ptx

popd