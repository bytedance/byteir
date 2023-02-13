import os
import shutil
import contextlib
import subprocess
import numpy as np
import tensorflow.compat.v1 as tf

from models.resnet50 import ResNet50
from tensorflow.python.framework import graph_util
from tf_mlir_ext import parse_and_evaluate_simple_module

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
WORKSPACE = "{}/.workspace".format(CUR_DIR)

def translate_from_tf_graph(model_path, output_names, output_path, batch_size, logfile_path):
    with open(logfile_path, "w+") as f:
        cmd_opts = ["tf-frontend"]
        cmd_opts += [model_path]
        cmd_opts += ["-batch-size={}".format(batch_size)]
        cmd_opts += ["-tf-output-arrays={}".format(','.join(output_names))]
        cmd_opts += ["-mlir-print-debuginfo"]
        cmd_opts += ["-o"]
        cmd_opts += [output_path]
        subprocess.check_call(cmd_opts, stdout=f, stderr=f)


def calc_mhlo_golden(input_model_path, output_model_path, golden_dir, logfile_path):
    with open(logfile_path, "w+") as f:
        os.makedirs(golden_dir, exist_ok=True)
        cmd_opts = ["tf-calc-golden"]
        cmd_opts.append(input_model_path)
        cmd_opts.append(output_model_path)
        cmd_opts.append(golden_dir)
        cmd_opts.append("-mlir-print-debuginfo")
        subprocess.check_call(cmd_opts, stdout=f, stderr=f)

def eval_mhlo_module(mlir_model_path, inputs, logfile_path):
    with open(mlir_model_path, "r") as f:
        with open(logfile_path, "w+") as logfile:
            with contextlib.redirect_stdout(logfile), contextlib.redirect_stderr(logfile):
                return parse_and_evaluate_simple_module(f.read(), inputs)


def test_resnet50():
    workdir = "{}/resnet50".format(WORKSPACE)

    tf_model_path = "{}/model.pb".format(workdir)
    mlir_model_path = "{}/model.mlir".format(workdir)
    golden_dir = "{}/golden".format(workdir)
    logfile_path = "{}/log.txt".format(workdir)

    batch_size = 2

    graph = tf.Graph()
    with graph.as_default():
        images = tf.placeholder(tf.float32, (batch_size, 224, 224, 3))
        model = ResNet50("channels_last")
        predictions = model(images, training=False)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            output_name = predictions.name.split(':')[0]
            graph_def = graph_util.convert_variables_to_constants(
                sess, sess.graph_def, output_node_names=[output_name,])
            tf.io.write_graph(graph_def, workdir, "model.pb", False)

            translate_from_tf_graph(tf_model_path, [output_name,], mlir_model_path, batch_size, logfile_path)
            calc_mhlo_golden(mlir_model_path, mlir_model_path, golden_dir, logfile_path)

            with open("{}/file_list".format(golden_dir), "r") as f:
                file_list = f.read().splitlines()

            golden_input = np.fromfile("{}/{}".format(golden_dir, file_list[0]), dtype=np.float32).reshape(batch_size, 224, 224, 3)
            golden_output = np.fromfile("{}/{}".format(golden_dir, file_list[-1]), dtype=np.float32).reshape(batch_size, 1000)

            tf_raw_output = sess.run(predictions, feed_dict={images: golden_input})
            python_eval_output, = eval_mhlo_module(mlir_model_path, [golden_input], logfile_path)

            np.testing.assert_allclose(tf_raw_output, golden_output, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(tf_raw_output, python_eval_output, rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    os.makedirs(WORKSPACE, exist_ok=True)
    test_resnet50()
    shutil.rmtree(WORKSPACE)
