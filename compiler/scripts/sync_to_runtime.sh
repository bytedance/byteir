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

# cp ../test/E2E/MLPInference/host_output.mlir ../../runtime/test/test_files/mlp_inference_host_cuda.mlir
# sed -i 's/your_file/test\/test_files\/mlp_inference_device.ptx/g' ../../runtime/test/test_files/mlp_inference_host_cuda.mlir
# cp ../test/E2E/MLPInference/device_output.ptx ../../runtime/test/test_files/mlp_inference_device.ptx

cp ../test/Pipelines/Host/E2E/Case0/Output.mlir ../../runtime/test/test_files/LLJIT/Case0/entry.mlir
cp ../test/Pipelines/Host/E2E/Case0/Output.ll   ../../runtime/test/test_files/LLJIT/Case0/host_kernels.ll

cp ../test/Pipelines/Host/E2E/TypeCvt/Output.ll   ../../runtime/test/test_files/LLJIT/typecvt.ll

# cp ../test/E2E/BertTiny/host_output.mlir ../../runtime/test/test_files/bert_tiny_host_cuda.mlir
# sed -i 's/your_file/test\/test_files\/bert_tiny_device.ptx/g' ../../runtime/test/test_files/bert_tiny_host_cuda.mlir
# cp ../test/E2E/BertTiny/device_output.ptx ../../runtime/test/test_files/bert_tiny_device.ptx

popd