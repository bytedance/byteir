import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
BYTEIR_OPT = "byteir-opt"
BYTEIR_TRANSLATE = "byteir-translate"
test_dirs = [
    "test/E2E/BertTiny",
    "test/E2E/ResNet18/BW",
    "test/E2E/ResNet18/FW",
    "test/E2E/ResNet18/Whole",
]


def run_and_write_log(cmds: list):
    cmd = " ".join(cmds)
    print("[RUN]: " + cmd)
    ret = os.system(cmd)
    if ret != 0:
        print("[Error in]: " + cmd)
        exit(1)


def append_header(input: str, opt: str, option: str, check_str="func.func @main"):
    with open(input, "r") as f:
        content = f.read()
    with open(input, "w") as f:
        header = (
            "// RUN: "
            + os.path.basename(opt)
            + " %s "
            + option
            + " | FileCheck %s\n\n// CHECK-LABEL: "
            + check_str
            + "\n"
        )
        f.write(header + content)


def run(
    input: str,
    output: str,
    opt: str,
    option: str,
    check_str="func.func @main",
    redirect=False,
    append_header_on_input=True,
):
    if append_header_on_input:
        append_header(input, opt, option, check_str)
    cmds = []
    if redirect:
        cmds = [opt, option, input, ">", output]
    else:
        cmds = [opt, option, input, "-o", output]
    run_and_write_log(cmds)


if __name__ == "__main__":
    for dir in test_dirs:
        paths = {}
        for i in [
            "input",
            "1_hlo_opt",
            "2_linalg_tensor_opt",
            "3_bufferize_opt",
            "4_affine_opt",
            "4_alternative_scf_opt",
            "5_gpu_opt",
            "6_byre_opt",
            "7a_byre_host",
            "7b_nvvm_codegen",
            "8b_ptx_codegen",
            "host_output",
            "device_output",
        ]:
            if i == "device_output":
                paths[i] = os.path.join(REPO_ROOT, dir, i + ".ptx")
            else:
                paths[i] = os.path.join(REPO_ROOT, dir, i + ".mlir")

        run(
            paths["input"],
            paths["1_hlo_opt"],
            BYTEIR_OPT,
            option="",
            append_header_on_input=False,
        )
        run(
            paths["1_hlo_opt"],
            paths["2_linalg_tensor_opt"],
            BYTEIR_OPT,
            '-hlo-opt="outline-single-elemwise-op"',
        )
        run(
            paths["2_linalg_tensor_opt"],
            paths["3_bufferize_opt"],
            BYTEIR_OPT,
            "-linalg-tensor-opt",
        )
        run(
            paths["3_bufferize_opt"],
            paths["4_affine_opt"],
            BYTEIR_OPT,
            "-byteir-bufferize-opt",
        )
        # copy from 4_affine_opt.mlir to 4_alternative_scf_opt.mlir
        run_and_write_log(["cp", paths["4_affine_opt"], paths["4_alternative_scf_opt"]])
        append_header(paths["4_alternative_scf_opt"], BYTEIR_OPT, "-scf-opt")
        run(paths["4_affine_opt"], paths["5_gpu_opt"], BYTEIR_OPT, "-affine-opt")
        run(paths["5_gpu_opt"], paths["6_byre_opt"], BYTEIR_OPT, "-gpu-opt")
        run(
            paths["6_byre_opt"],
            paths["7a_byre_host"],
            BYTEIR_OPT,
            '-byre-opt="append-arg-types"',
        )
        # copy from 7a_byre_host.mlir to 7b_nvvm_codegen.mlir
        run_and_write_log(["cp", paths["7a_byre_host"], paths["7b_nvvm_codegen"]])
        run(
            paths["7a_byre_host"],
            paths["host_output"],
            BYTEIR_OPT,
            '-byre-host="device-file-name=your_file target=cuda"',
        )
        append_header(paths["host_output"], BYTEIR_OPT, option="")
        run(
            paths["7b_nvvm_codegen"],
            paths["8b_ptx_codegen"],
            BYTEIR_OPT,
            "-nvvm-codegen",
            check_str="gpu.module @unified",
        )
        run(
            paths["8b_ptx_codegen"],
            paths["device_output"],
            BYTEIR_TRANSLATE,
            "-gen-ptx -dump-ptx",
            check_str=".visible .entry Unknown",
            redirect=True,
        )
