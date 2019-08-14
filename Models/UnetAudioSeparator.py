import tensorflow as tf

import Models.InterpolationLayer
import Utils
from Utils import LeakyReLU
import numpy as np
import Models.OutputLayer
import scipy.signal
from typing import Callable

class UnetAudioSeparator:
    '''
    U-Net separator network for singing voice separation.
    Uses valid convolutions, so it predicts for the centre part of the input - only certain input and output shapes are therefore possible (see getpadding function)
    '''

    def __init__(self, model_config):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        '''
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.filter_size = model_config["filter_size"]
        self.merge_filter_size = model_config["merge_filter_size"]
        self.input_filter_size = model_config["input_filter_size"]
        self.output_filter_size = model_config["output_filter_size"]
        self.upsampling = model_config["upsampling"]
        self.downsampling = model_config["downsampling"]
        self.output_type = model_config["output_type"]
        self.context = model_config["context"]
        self.padding = "valid" if model_config["context"] else "same"
        self.source_names = model_config["separator_source_names"]
        self.num_channels = 1 if model_config["mono_downmix"] else 2
        self.output_activation = model_config["output_activation"]
        self.score_informed = model_config["score_informed"]
        self.model_config = model_config

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape 
        :return: Input_shape, output_shape, where each is a list [batch_size, time_steps, channels]
        '''

        if self.context:
            # Check if desired shape is possible as output shape - go from output shape towards lowest-res feature map
            rem = float(shape[1]) # Cut off batch size number and channel

            # Output filter size
            rem = rem - self.output_filter_size + 1

            # Upsampling blocks
            for i in range(self.num_layers):
                rem = rem + self.merge_filter_size - 1
                rem = (rem + 1.) / 2.# out = in + in - 1 <=> in = (out+1)/

            # Round resulting feature map dimensions up to nearest integer
            x = np.asarray(np.ceil(rem),dtype=np.int64)
            assert(x >= 2)

            # Compute input and output shapes based on lowest-res feature map
            output_shape = x
            input_shape = x

            # Extra conv
            input_shape = input_shape + self.filter_size - 1

            # Go from centre feature map through up- and downsampling blocks
            for i in range(self.num_layers):
                output_shape = 2*output_shape - 1 #Upsampling
                output_shape = output_shape - self.merge_filter_size + 1 # Conv

                input_shape = 2*input_shape - 1 # Decimation
                if i < self.num_layers - 1:
                    input_shape = input_shape + self.filter_size - 1 # Conv
                else:
                    input_shape = input_shape + self.input_filter_size - 1

            # Output filters
            output_shape = output_shape - self.output_filter_size + 1

            input_shape = np.concatenate([[shape[0]], [input_shape], [self.num_channels]])
            output_shape = np.concatenate([[shape[0]], [output_shape], [self.num_channels]])

            return input_shape, output_shape
        else:
            return [shape[0], shape[1], self.num_channels], [shape[0], shape[1], self.num_channels]

    def get_output(self, input_mix, training, return_spectrogram=False, reuse=True, scores=None):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch of mixtures, 3D tensor [batch_size, num_samples, num_channels]
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :return: U-Net output: List of source estimates. Each item is a 3D tensor [batch_size, num_out_samples, num_channels]
        '''
        with tf.variable_scope("separator", reuse=reuse):
            processed_scores = None
            if self.score_informed:
                with tf.variable_scope("%s_scores" % self.model_config['score_type']):
                    processed_scores = {
                        source: get_score(scores[source + '_score'], self.model_config['score_type'], self.model_config)
                        for source in self.source_names
                    }

            mix_and_score = None
            if self.model_config['score_input_concat']:
                input_channels = mix_and_score = self.concat_score(input_mix, processed_scores)
            else:
                input_channels = input_mix

            current_layer, enc_outputs = self.get_downsampling_layers(input_channels)
            current_layer = self.get_upsampling_layers(current_layer, enc_outputs)  # out = in - filter + 1

            with tf.variable_scope("concat_signal"):
                if self.model_config['score_featuremap_concat']:
                    signal_to_concat = self.concat_score(input_mix, processed_scores) if mix_and_score is None else mix_and_score
                else:
                    signal_to_concat = input_mix
                
                current_layer = Utils.crop_and_concat(signal_to_concat, current_layer, match_feature_dim=False)

            scores_for_output_layer = processed_scores if self.model_config['score_per_source_concat'] else None
            return self.get_output_layer(input_channels, current_layer, training, scores_for_output_layer)
            
    def get_downsampling_layers(self, input):
        enc_outputs = list()        
        current_layer = input

        # Down-convolution: Repeat strided conv
        for i in range(self.num_layers + 1):
            scope_name = "layer%s_downsampling" % i if i < self.num_layers else "bottleneck"
            with tf.variable_scope(scope_name):
                num_filters = self.num_initial_filters + self.model_config["additional_filters_per_layer"] * i
                current_layer = tf.layers.conv1d(current_layer, num_filters, self.filter_size, activation=LeakyReLU, padding=self.padding) # out = in - filter + 1
                if i < self.num_layers:
                    enc_outputs.append(current_layer)
                    with tf.variable_scope("decimation"):
                        if self.downsampling == 'naive':
                            current_layer = current_layer[:,::2,:] # Decimate by factor of 2 # out = (in-1)/2 + 1
                        else:
                            dims = current_layer.shape.dims
                            shape_after_decimation = (dims[0], dims[1] // 2, dims[2])
                            current_layer = tf.py_func(UnetAudioSeparator.decimate, [current_layer], tf.float32, stateful=False, name='decimate')
                            current_layer = tf.ensure_shape(current_layer, shape_after_decimation)
        
        return current_layer, enc_outputs


    @staticmethod
    def decimate(layer):
        return scipy.signal.decimate(layer, 2, axis=1, ftype='fir').astype(np.float32)

    @staticmethod
    def interpolate(layer):
        return scipy.signal.resample_poly(layer, up=2, down=1, axis=1).astype(np.float32)
        
    def get_upsampling_layers(self, current_layer, enc_outputs):
        # Upconvolution
        for i in range(self.num_layers):
            with tf.variable_scope("layer%s_upsampling" % (self.num_layers - i - 1)):
                with tf.variable_scope("upsampling"):
                    #UPSAMPLING
                    if self.upsampling == 'learned':
                        current_layer = tf.expand_dims(current_layer, axis=1)
                        # Learned interpolation between two neighbouring time positions by using a convolution filter of width 2, and inserting the responses in the middle of the two respective inputs
                        current_layer = Models.InterpolationLayer.learned_interpolation_layer(current_layer, self.padding, i)
                        current_layer = tf.squeeze(current_layer, axis=1)
                    elif self.upsampling == 'filter':
                        dims = current_layer.shape.dims
                        shape_after_upsampling = (dims[0], dims[1] * 2, dims[2])
                        current_layer = tf.py_func(UnetAudioSeparator.interpolate, [current_layer], tf.float32, stateful=False, name='interpolate')
                        current_layer = tf.ensure_shape(current_layer, shape_after_upsampling)
                    else:
                        current_layer = tf.expand_dims(current_layer, axis=1)                        
                        if self.context:
                            current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)
                        else:
                            current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2]) # out = in + in - 1
                        current_layer = tf.squeeze(current_layer, axis=1)
                    # UPSAMPLING FINISHED

                assert(enc_outputs[-i-1].get_shape().as_list()[1] == current_layer.get_shape().as_list()[1] or self.context) #No cropping should be necessary unless we are using context
                with tf.variable_scope("crop_and_concat"):
                    current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer, match_feature_dim=False)
                num_filters = self.num_initial_filters + (self.model_config["additional_filters_per_layer"] * (self.num_layers - i - 1))
                current_layer = tf.layers.conv1d(current_layer, num_filters, self.merge_filter_size,
                                                activation=LeakyReLU,
                                                padding=self.padding)  # out = in - filter + 1
        return current_layer
    
    def get_output_layer(self, input_mix, current_layer, training, scores):
        # Determine output activation function
        if self.output_activation == "tanh":
            out_activation = tf.tanh # type: Callable
        elif self.output_activation == "linear":
            out_activation = lambda x: Utils.AudioClip(x, training)
        else:
            raise NotImplementedError

        if self.output_type == "direct":
            return Models.OutputLayer.independent_outputs(current_layer, self.source_names, self.num_channels, self.output_filter_size, self.padding, out_activation, scores)
        elif self.output_type == "difference":
            assert scores is None # Unsupported with score-informed for now
            return Models.OutputLayer.difference_output(input_mix, current_layer, self.source_names, self.num_channels, self.output_filter_size, self.padding, out_activation, training)
        else:
            raise NotImplementedError


    def get_embedding(self, input_mix, reuse=True):
        with tf.variable_scope("separator", reuse=reuse):
            current_layer, enc_outputs = self.get_downsampling_layers(input_mix)
            return current_layer

    def concat_score(self, input_mix, scores):
        with tf.variable_scope('concat_score'):
            scores = [scores[source] for source in self.source_names]
            return tf.concat([input_mix] + scores, axis=2)

# Score pitch range: C2 (MIDI pitch 36) to C6 (84).
min_score_pitch = 36
max_score_pitch = 84
score_pitch_count = max_score_pitch - min_score_pitch + 1

def get_score(score, score_type, model_config):
    if score_type == 'one-hot':
        return get_one_hot_score(score)
    elif score_type == 'midi_pitch':
        return tf.cast(score, tf.float32)
    elif score_type == 'midi_pitch_normalized':
        return get_normalized_score(score)
    elif score_type == 'pitch_and_amplitude':
        return get_pitch_and_amplitude_score(score)
    elif score_type == 'pure_tone_synth':
        return get_pure_tone_synth_score(score, model_config['expected_sr'])
    else:
        raise ValueError('Invalid score_type: ' + score_type)

def get_one_hot_score(score):
        score = tf.squeeze(score, axis=-1)
        return tf.one_hot(score - min_score_pitch, score_pitch_count, dtype=tf.float32)

def get_normalized_score(score):
    """
    Return score with pitch normalized to range [0, 1], with -1
    in places where no pitch is present (0 in original score).
    """
    score_float = tf.cast(score, tf.float32)
    normalized_score_values = (score_float - min_score_pitch) / score_pitch_count
    zero_locations = tf.equal(score_float, 0)
    normalized_score = tf.where(zero_locations, tf.fill(tf.shape(score_float), -1.0), normalized_score_values)
    return tf.clip_by_value(normalized_score, -1, 1)

def get_pitch_and_amplitude_score(score):
    """
    Return score in two tracks:
        - pitch normalized to range [-1, 1]
        - amplitude -- 1 when something is playing, 0 when nothing is playing
    """
    score_float = tf.cast(score, tf.float32)
    normalized_score = (score_float - min_score_pitch) / score_pitch_count * 2 - 1
    pitch = tf.clip_by_value(normalized_score, -1, 1)
    amplitude = tf.cast(score_float > 0, tf.float32)
    return tf.concat([pitch, amplitude], axis=2)

def get_pure_tone_synth_score(score, sample_rate):
    # TODO: Prevent clicks? Does it matter?
    score_float = tf.squeeze(tf.cast(score, tf.float32), -1)
    frequency = 440.0 * (2.0 ** ((score_float - 69.0)/12.0))
    num_samples = score_float.shape[1]
    time = tf.range(int(num_samples), dtype=tf.float32) / sample_rate
    synth_score = tf.math.sin(2 * np.pi * frequency * time)
    zero_locations = tf.equal(score_float, 0)
    synth_score = tf.where(zero_locations, tf.zeros_like(score_float), synth_score)
    return tf.expand_dims(synth_score, -1)
