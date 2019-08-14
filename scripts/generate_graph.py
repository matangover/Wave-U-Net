"""
Run this script:
    python generate_graph.py with name=006 train/006.json

Then view the graph in TensorBoard:
    tensorboard --logdir=logs/graphs
"""

from sacred import Experiment
from Config import config_ingredient
import visualize
import tensorflow as tf
import time

ex = Experiment('Waveunet', ingredients=[config_ingredient])

@ex.automain
def main(cfg, name):
    model_config = cfg["model_config"]
    graph = tf.Graph()
    with graph.as_default():
        separator_input = visualize.init_graph(model_config)
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = 'logs/graphs/%s' % name
        with tf.summary.FileWriter(log_dir) as writer:
            writer.add_graph(graph)
