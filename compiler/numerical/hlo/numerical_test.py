import os
import mhlo_tools
import numpy as np
import argparse

def get_config(config: str):
    if config == "":
        return {}
    raise NotImplementedError("No such config name {}".format(config))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before_pass_file")
    parser.add_argument("after_pass_file")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--decimal", type=str, default="")
    parser.add_argument("--black_list", type=str, default="")
    args = parser.parse_args()

    if args.config != "":
      mhlo_tools.testing.pass_numerical_test(
          args.before_pass_file, args.after_pass_file, **get_config(args.config)
      )
    else:
      inputs_dict = {}
      decimal_dict = {}
      black_list = []
      if args.decimal != "":
        decimal_list = args.decimal.split(",")
        for i in decimal_list:
          decimal_dict[i.split(":")[0]] = int(i.split(":")[1])
      if args.black_list != "":
        black_list = args.black_list.split(",")
      mhlo_tools.testing.pass_numerical_test(
          args.before_pass_file, args.after_pass_file, inputs_dict=inputs_dict, decimal_dict=decimal_dict, black_list=black_list
      )
