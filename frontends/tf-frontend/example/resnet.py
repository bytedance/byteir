import os
import shutil
import contextlib
import subprocess
import numpy as np
import argparse
import tensorflow.compat.v1 as tf

from resnet50_model import ResNet50
from tensorflow.python.framework import graph_util

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


def test_resnet50(also_convert):
    workdir = "{}/resnet50".format(WORKSPACE)

    tf_model_path = "{}/model.pb".format(workdir)
    mlir_model_path = "{}/model.mlir".format(workdir)
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

            if also_convert:
                translate_from_tf_graph(
                    tf_model_path, [output_name,], mlir_model_path, batch_size, logfile_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    description='tf-frontend for resnet50')
    parser.add_argument('-c', '--also_convert',
                    action='store_true')
    args = parser.parse_args()
    os.makedirs(WORKSPACE, exist_ok=True)
    test_resnet50(args.also_convert)
