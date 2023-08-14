from formatCheck.check import *
import argparse

# parse directory path
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, help="path to directory")
args = parser.parse_args()

format_check(args.dir)
