#!/usr/bin/python3

import byteir
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("input_mlir_path")
    parser.add_argument("-o", "--output_host_mlir_path", type=str, help="output host mlir path")
    parser.add_argument("--entry_func", type=str, default="main", help="entry function name")
    parser.add_argument("--target", type=str, default="cuda", help="target device name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--ait_parallelism", type=int, default=1, help="number of processes to compile ait op")
    parser.add_argument("--disable_byteir_cache", action="store_true")

    args = parser.parse_args()
    byteir.compile(args.input_mlir_path, 
                   args.output_host_mlir_path, 
                   args.entry_func, 
                   args.target, 
                   args.verbose, 
                   args.ait_parallelism,
                   args.disable_byteir_cache)


