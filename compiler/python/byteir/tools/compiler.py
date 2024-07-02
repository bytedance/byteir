#!/usr/bin/python3

import byteir
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("input_mlir_path")
    parser.add_argument("-o", "--output_host_mlir_path", type=str, required=True, help="output host mlir path")
    parser.add_argument("--entry_func", type=str, default="main", help="entry function name")
    parser.add_argument("--target",
                        type=str,
                        default="cuda",
                        choices=["cuda", "cuda_with_ait", "cuda_with_ait_aggressive", "cpu"],
                        help="target device name")
    parser.add_argument("--gpu_arch",
                        type=str,
                        default="local",
                        choices=["local", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
                        help="specify target gpu arch: 'local' for detecting by nvidia-smi")
    parser.add_argument("--cpu_arch",
                        type=str,
                        default="x86_64",
                        choices=["x86_64", "aarch64"],
                        help="specify target cpu arch")
    parser.add_argument("--serial_version", type=str, default="1.0.0", help="byre serialize version")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    # gpu options
    parser.add_argument("--enable_tf32", default=False, action="store_true")
    parser.add_argument("--ait_parallelism", type=int, default=1, help="number of processes to compile ait op")
    parser.add_argument("--disable_byteir_cache", default=False, action="store_true")

    args = parser.parse_args()
    byteir.compile(args.input_mlir_path,
                   args.output_host_mlir_path,
                   entry_func=args.entry_func,
                   target=args.target,
                   gpu_arch=args.gpu_arch,
                   cpu_arch=args.cpu_arch,
                   byre_serial_version=args.serial_version,
                   verbose=args.verbose,
                   enable_tf32=args.enable_tf32,
                   parallelism=args.ait_parallelism,
                   disable_byteir_ait_cache=args.disable_byteir_cache)
