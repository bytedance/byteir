import os
import os.path as osp
import sys

ONNX_FRONTEND_PATH = osp.join(os.environ["ONNX_FRONTEND_ROOT"], "build/onnx-frontend/src/onnx-frontend")
LARGE_MODEL_PATH = os.environ["LARGE_MODEL_PATH"]
