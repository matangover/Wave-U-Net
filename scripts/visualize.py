import Models
import numpy as np
import tensorflow as tf
import sacred.initialize
import Predict
import Models.UnetAudioSeparator

def init_graph(model_config):
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    separator = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    sep_input_shape, sep_output_shape = separator.get_padding(np.array(disc_input_shape))

    batch_size = 1
    sep_input_shape[0] = batch_size
    sep_output_shape[0] = batch_size

    mix_ph = tf.placeholder(tf.float32, sep_input_shape, 'input_mix')
    scores = {
        source_name + '_score': tf.placeholder(tf.uint8, sep_input_shape, source_name + '_score')
        for source_name in model_config["separator_source_names"]}
    _separator_sources = separator.get_output(mix_ph, training=False, return_spectrogram=False, reuse=False, scores=scores)
    return mix_ph

def get_config():
    return sacred.initialize.create_run(Predict.ex, Predict.ex.default_command).config
