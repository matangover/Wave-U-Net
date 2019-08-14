from sacred import Experiment
from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os

import Datasets
import Utils
import Models.UnetAudioSeparator

ex = Experiment('Waveunet Preprocessing', ingredients=[config_ingredient])

@ex.config
# Executed for training, sets the seed value to the Sacred config so that Sacred fixes the Python and Numpy RNG to the same state everytime.
def set_seed():
    seed = 1337

@config_ingredient.capture
def preprocess(model_config, dataset):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of input
    separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))

    tiny = 'tiny' in dataset
    Datasets.preprocess_dataset(model_config, sep_input_shape, sep_output_shape, tiny)

@ex.automain
def run(cfg):
    preprocess()
    print("Preprocessing finished.")