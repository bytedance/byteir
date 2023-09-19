import argparse

parser = argparse.ArgumentParser(description="Parse catir executor options.")
parser.add_argument("--name", type=str, default="model", help="model name")
parser.add_argument("--mhlo_path", type=str, required=True, help="path to mhlo.ir")
parser.add_argument(
    "--bypass-byteir", action="store_true", help="whether use backend directly without ByteIR lowering"
)
parser.add_argument("--dump_ir", action="store_true", help="dump ir to files")
parser.add_argument(
    "--workdir", type=str, default="./workspace", help="workspace directory"
)
parser.add_argument(
    "--backend", type=str, choices=["ait"], default="ait", help="runtime backend"
)
args = parser.parse_args()

from byteir import ir
from byteir.dialects.cat import IRProcessor

from pathlib import Path
import os

if args.dump_ir:
    os.makedirs(args.workdir, exist_ok=True)

if __name__ == "__main__":
    with ir.Context() as context:

        processor = IRProcessor(args.name, args.workdir)
        processor.load_from_file(args.mhlo_path)
        
        if args.bypass_byteir:
            processor.cat_opt_pass(anchor_only=True, dump_ir=args.dump_ir)
            if not args.dump_ir:
                processor.benchmark(backend=args.backend, num_trials=100)
        else:
            processor.preprocess_pass(args.dump_ir)
            # convert mhlo to cat
            processor.cat_opt_pass(anchor_only=False, dump_ir=args.dump_ir)
            # clustering
            processor.hlo_opt_pass(dump_ir=args.dump_ir)
            # generate ait .so for subgraphs
            processor.ait_opt_pass(anchor_only=True, dump_ir=args.dump_ir)
            processor.bufferize_opt_pass(dump_ir=args.dump_ir)
            
