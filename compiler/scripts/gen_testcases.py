import os
import pathlib
import collections
import functools
import subprocess
import argparse

CUR_DIR = pathlib.Path(__file__).absolute().parent
BIN_DIR = CUR_DIR.parent.joinpath("build", "bin")

Stage = collections.namedtuple("stage", ["identifier", "filename"])
Content = collections.namedtuple("content", ["stages", "content"])
Pipeline = collections.namedtuple("pipeline", ["executable", "cur_stage", "next_stages", "pipelines", "filecheck"])
OptPipeline = functools.partial(Pipeline, "byteir-opt")
TranslatePipeline = functools.partial(Pipeline, "byteir-translate")
Testcase = collections.namedtuple("testcase", ["contents", "pipelines"])

def composePipelines(pipelineCtors, cur_stage, next_stages):
    def fn(*args, **kwargs):
        underlyingPipelines = [ctor(*args, **kwargs) for ctor in pipelineCtors]
        executable = underlyingPipelines[0].executable
        pipelines = []
        pipelines += underlyingPipelines[0].pipelines
        filecheck = underlyingPipelines[0].filecheck
        cur_executable = executable
        for i in underlyingPipelines[1:]:
            if i.executable != cur_executable:
                pipelines += ['|', i.executable]
                cur_executable = i.executable
            pipelines += i.pipelines
        return Pipeline(executable, cur_stage, next_stages, pipelines, filecheck)
    return fn

# for host pipelines
class HostPipelineCollections:
    # stages
    Input    = Stage("input",     "00_Input.mlir")
    HostOpt  = Stage("host_opt",  "01_HostOpt.mlir")
    ByreHost = Stage("byre_host", "02a_ByreHost.mlir")
    ToLLVM   = Stage("to_llvm",   "02b_ToLLVM.mlir")
    ToLLVMIR = Stage("to_llvmir", "03b_ToLLVMIR.mlir")
    ByreOut  = Stage("byre_out",  "Output.mlir")
    LLVMOut  = Stage("llvm_out",  "Output.ll")
    E2E      = Stage("e2e",       "TotalPipeline.mlir")

    # pipelines
    InputPipeline = functools.partial(OptPipeline, Input, [HostOpt], [
        "--hlo-graph-opt --hlo-fusion-opt=\"target=CPU\"",
        "--linalg-tensor-opt=\"target=CPU\"",
        "--byre-tensor-opt=\"entry-func=main append-arg-types\"",
        "--byteir-bufferize-opt",
        "--linalg-memref-opt",
        "--scf-opt=\"target=CPU\"",
    ])
    HostOptPipeline = functools.partial(OptPipeline, HostOpt, [ByreHost, ToLLVM], [
        "--host-opt",
        "-set-op-space=\"entry-func=main space=cpu\"",
        "-set-arg-space=\"entry-func=main all-space=cpu\"",
        "--byre-opt",
    ])
    ToLLVMPipeline = functools.partial(OptPipeline, ToLLVM, [ToLLVMIR], [
        "--to-llvm",
    ])
    ToLLVMIRPipeline = functools.partial(TranslatePipeline, ToLLVMIR, [LLVMOut], [
        "--mlir-to-llvmir",
    ])
    ByreHostPipeline = functools.partial(OptPipeline, ByreHost, [ByreOut], [
        "-byre-host=\"device-file-name=your_file target=cpu\"",
    ])
    ByreOutPipeline = functools.partial(OptPipeline, ByreOut, [], [])
    TotalPipeline = composePipelines([InputPipeline, HostOptPipeline, ToLLVMPipeline, ToLLVMIRPipeline], E2E, [])

# for host pipelines
class HostPipelineBytecodeCollections:
    # stages
    Input       = Stage("input",     "00_Input.mlir")
    HostOpt     = Stage("host_opt",  "01_HostOpt.mlir")
    ByreHost    = Stage("byre_host", "02a_ByreHost.mlir")
    ByreSerial  = Stage("byre_out",  "03a_ByreSerial.mlir")
    ToLLVM      = Stage("to_llvm",   "02b_ToLLVM.mlir")
    ToLLVMBC    = Stage("to_llvmbc", "03b_ToLLVMBC.mlir")
    MLIROut     = Stage("mlir_out",  "Output.mlirbc")
    LLVMOut     = Stage("llvm_out",  "Output.bc")

    # pipelines
    InputPipeline = functools.partial(OptPipeline, Input, [HostOpt], [
        "--hlo-graph-opt --hlo-fusion-opt=\"target=CPU\"",
        "--linalg-tensor-opt=\"target=CPU\"",
        "--byre-tensor-opt=\"entry-func=main append-arg-types\"",
        "--byteir-bufferize-opt",
        "--linalg-memref-opt",
        "--scf-opt=\"target=CPU\"",
    ])
    HostOptPipeline = functools.partial(OptPipeline, HostOpt, [ByreHost, ToLLVM], [
        "--host-opt=\"file-name=host_kernels.bc\"", "--byre-opt",
    ])
    ToLLVMPipeline = functools.partial(OptPipeline, ToLLVM, [ToLLVMBC], [
        "--to-llvm",
    ])
    ToLLVMBCPipeline = functools.partial(TranslatePipeline, ToLLVMBC, [LLVMOut], [
        "--mlir-to-llvmbc",
    ])
    ByreHostPipeline = functools.partial(OptPipeline, ByreHost, [ByreSerial], [
        "--collect-func=\"anchor-attr=byre.entry_point\"",
        "-set-op-space=\"entry-func=main space=cpu\"",
        "-set-arg-space=\"entry-func=main all-space=cpu\"",
    ])
    ByreSerialPipeline = functools.partial(OptPipeline, ByreSerial, [], [
        "--dump-byre=\"file-name={}\"".format(MLIROut.filename),
    ])

# for e2e
class E2ECollections:
    Input           = Stage("input",               "input.mlir")
    HloOpt          = Stage("hlo_opt",             "1_hlo_opt.mlir")
    LinalgTensorOpt = Stage("linalg_tensor_opt",   "2_linalg_tensor_opt.mlir")
    ByreTensorOpt   = Stage("byre_tensor_opt",     "3_byre_tensor_opt.mlir")
    BufferizeOpt    = Stage("bufferize_opt",       "4_bufferize_opt.mlir")
    AffineOpt       = Stage("affine_opt",          "5_affine_opt.mlir")
    SCFOpt          = Stage("alternative_scf_opt", "5_alternative_scf_opt.mlir")
    GPUOpt          = Stage("gpu_opt",             "6_gpu_opt.mlir")
    SetSpaceOpt     = Stage("set_space_opt",       "7_set_space_opt.mlir")
    ByreOpt         = Stage("byre_opt",            "8_byre_opt.mlir")
    ByreHost        = Stage("byre_host",           "9a_byre_host.mlir")
    NVVMCodegen     = Stage("nvvm_codegen",        "9b_nvvm_codegen.mlir")
    PTXCodegen      = Stage("ptx_codegen",         "10b_ptx_codegen.mlir")
    HostOutput      = Stage("host_output",         "host_output.mlir")
    DeviceOutput    = Stage("device_output",       "device_output.ptx")

    InputPipeline = functools.partial(OptPipeline, Input, [HloOpt], [])
    HloOptPipeline = functools.partial(OptPipeline, HloOpt, [LinalgTensorOpt], [
        "-hlo-graph-opt -hlo-fusion-opt=\"outline-single-elemwise-op\"",
    ])
    LinalgTensorOptPipeline = functools.partial(OptPipeline, LinalgTensorOpt, [ByreTensorOpt], [
        "-linalg-tensor-opt",
    ])
    def ByreTensorOptPipeline(filecheck, *, entryFunc="main"):
        return OptPipeline(E2ECollections.ByreTensorOpt, [E2ECollections.BufferizeOpt], ["-byre-tensor-opt=\"append-arg-types entry-func={}\"".format(entryFunc)], filecheck)
    BufferizeOptPipeline = functools.partial(OptPipeline, BufferizeOpt, [AffineOpt, SCFOpt], [
        "-byteir-bufferize-opt",
    ])
    AffineOptPipeline = functools.partial(OptPipeline, AffineOpt, [GPUOpt], [
        "-affine-opt",
    ])
    SCFOptPipeline = functools.partial(OptPipeline, SCFOpt, [GPUOpt], [
        "-scf-opt",
    ])
    GPUOptPipeline = functools.partial(OptPipeline, GPUOpt, [SetSpaceOpt], [
        "-gpu-opt"
    ])
    def SetSpaceOptPipeline(filecheck, *, entryFunc="main"):
        return OptPipeline(E2ECollections.SetSpaceOpt, [E2ECollections.ByreOpt], [
            "-remove-func-body=\"anchor-attr=__byteir_elementwise_fusion__\"",
            "--inline",
            "--gpu-launch-func-to-byre",
            "-set-op-space=\"entry-func={} space=cuda\"".format(entryFunc),
            "-set-arg-space=\"entry-func={} all-space=cuda\"".format(entryFunc)
        ], filecheck)
    def ByreOptPipeline(filecheck, *, entryFunc="main"):
        return OptPipeline(E2ECollections.ByreOpt, [E2ECollections.ByreHost, E2ECollections.NVVMCodegen], ["-byre-opt=\"append-arg-types entry-func={}\"".format(entryFunc)], filecheck)
    def ByreHostPipeline(filecheck, *, entryFunc="main"):
        return OptPipeline(E2ECollections.ByreHost, [E2ECollections.HostOutput], ["-byre-host=\"device-file-name=your_file target=cuda entry-func={}\"".format(entryFunc)], filecheck)
    HostOutputPipeline = functools.partial(OptPipeline, HostOutput, [], [])
    NVVMCodegenPipeline = functools.partial(OptPipeline, NVVMCodegen, [PTXCodegen], [
        "-nvvm-codegen"
    ])
    PTXCodegenPipeline = functools.partial(TranslatePipeline, PTXCodegen, [DeviceOutput], [
        "-gen-ptx", "-o-ptx", DeviceOutput.filename[:-4], "-dump-ptx"
    ])

def render(content, pipeline):
    if pipeline.filecheck is None:
        return "// RUN: {0} %s {1}\n\n{2}".format(
            pipeline.executable,
            " ".join(pipeline.pipelines),
            content.strip()
        ).strip()
    else:
        return "// RUN: {0} %s {1} | FileCheck %s\n\n{2}\n\n{3}".format(
            pipeline.executable,
            " ".join(pipeline.pipelines),
            pipeline.filecheck.strip(),
            content.strip()
        ).strip()

def emitSingleTestcase(workdir, testcase):
    print("===- start processing {} -===".format(workdir))
    for i in testcase.contents:
        assert isinstance(i, Content), "item in testcase.contents must be a Content"
        for s in i.stages:
            with workdir.joinpath(s.filename).open("w") as f:
                f.write(i.content)

    for pipeline in testcase.pipelines:
        assert isinstance(pipeline, Pipeline), "item in testcase.pipelines must be a Pipeline"
        output = subprocess.check_output(' '.join([
            pipeline.executable, pipeline.cur_stage.filename, *pipeline.pipelines
        ]), text=False, shell=True, env=dict(os.environ, PATH=BIN_DIR), cwd=workdir)
        for next_stage in pipeline.next_stages:
            with workdir.joinpath(next_stage.filename).open("wb") as f:
                f.write(output)

        print("  emit {}".format(pipeline.cur_stage))
        with workdir.joinpath(pipeline.cur_stage.filename).open("r+") as f:
            input = f.read()
            f.seek(0)
            f.write(render(input, pipeline))
            f.truncate()

def processSingleTestcase(templateFile, env):
    with templateFile.open() as f:
        testcase = eval(f.read(), env)
        emitSingleTestcase(templateFile.parent, testcase)

def processAllTestcases(top_dir, template_file, collections):
    collections = {k: v for k, v in collections.__dict__.items() if not k.startswith("__")}
    env = dict(collections,
        Stage=Stage,
        Pipeline=Pipeline,
        Testcase=Testcase,
        Content=Content
    )
    for templateFile in pathlib.Path(top_dir).glob("**/{}".format(template_file)):
        processSingleTestcase(templateFile, env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ByteIR testcases generator for consecutive pipelines")
    parser.add_argument("--top-dir", required=True, type=str, help="Top-level dir to search testcases")
    parser.add_argument("--template-file", type=str, default="template.py", help="Filename of the testcase template")
    parser.add_argument("--category", type=str, choices=["E2E", "HostPipeline", "HostPipelineBytecode"], help="Category of testcases")

    args = parser.parse_args()

    if args.category == "HostPipeline":
        collections = HostPipelineCollections
    elif args.category == "HostPipelineBytecode":
        collections = HostPipelineBytecodeCollections
    elif args.category == "E2E":
        collections = E2ECollections
    else:
        raise RuntimeError("Unknown testcase category")

    processAllTestcases(args.top_dir, args.template_file, collections)
